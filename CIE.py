from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Any
import tempfile
import zipfile
import shutil
import os
import asyncio
import subprocess
import json

app = FastAPI()

# --- Dummy auth dependency and models (replace with your real ones) ---
class User(BaseModel):
    id: int
    username: str

async def get_current_user() -> User:
    # Replace this with your real authentication/dependency
    return User(id=1, username="testuser")

# --- Utility functions ---

def _is_zip_file_bytes(data: bytes) -> bool:
    """Check whether the given bytes represent a valid ZIP archive by trying to open it."""
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(data)
            tmp.flush()
            tmp_path = tmp.name
        with zipfile.ZipFile(tmp_path, 'r') as zf:
            # try a quick read of namelist to validate the archive structure
            _ = zf.namelist()
        return True
    except zipfile.BadZipFile:
        return False
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


async def _clone_repo_async(git_url: str, dest_dir: Optional[str] = None, timeout: int = 60) -> str:
    """
    Clone the repository into a temporary directory using subprocess in a thread so it doesn't block the event loop.
    Returns the path to cloned repository on success, raises an exception on failure.
    """
    if dest_dir is None:
        dest_dir = tempfile.mkdtemp(prefix="repo_clone_")

    # Use git clone --depth 1 to reduce data transferred and time
    cmd = ["git", "clone", "--depth", "1", git_url, dest_dir]

    try:
        # run in a thread so FastAPI's event loop isn't blocked
        proc = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            # include stderr for debugging
            raise RuntimeError(f"git clone failed: {proc.stderr.strip()}")
        return dest_dir
    except subprocess.TimeoutExpired:
        # cleanup
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise RuntimeError("git clone timed out")
    except Exception as exc:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise


async def _validate_git_url(git_url: str) -> bool:
    """
    Basic validation for git URL shape. This is intentionally permissive; ultimate validation happens by attempting to clone or run `git ls-remote`.
    """
    if not isinstance(git_url, str) or not git_url.strip():
        return False
    # quick blacklist: avoid local paths or suspicious input
    if git_url.startswith("/") or git_url.startswith("file://"):
        return False
    return True


# --- Endpoint ---
@app.post("/submit-project")
async def submit_project(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    git_url: Optional[str] = Form(None),
    personas_json: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
):
    """
    Accept either exactly one of: a ZIP file upload or a Git repository URL.
    personas_json - optional JSON string representing personas (e.g., '["SDE","PM"]')

    Returns JSON describing the submission and starts background work for long-running tasks (e.g., cloning).
    """

    # --- Normalize and parse personas if provided ---
    personas: Optional[List[Any]] = None
    if personas_json:
        try:
            personas = json.loads(personas_json)
            if not isinstance(personas, list):
                raise ValueError("personas must be a JSON array")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid personas JSON: {str(exc)}")

    # --- Require exactly one submission method ---
    if (file is None and (git_url is None or git_url.strip() == "")):
        raise HTTPException(status_code=400, detail="You must provide a file or a Git URL.")
    if file is not None and git_url and git_url.strip() != "":
        raise HTTPException(status_code=400, detail="Provide only one submission method: either a file or a Git URL, not both.")

    # --- Handle file upload ---
    if file is not None:
        filename = file.filename or ""
        if not filename.lower().endswith('.zip'):
            raise HTTPException(status_code=400, detail="Only .zip files are accepted based on filename.")

        # read contents (careful about very large uploads)
        contents = await file.read()

        # verify ZIP integrity by attempting to open it
        if not _is_zip_file_bytes(contents):
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid ZIP archive.")

        # Optionally save to permanent storage. Here we'll save it to a temp dir and return its path for demo.
        saved_dir = tempfile.mkdtemp(prefix="uploaded_project_")
        saved_path = os.path.join(saved_dir, filename)
        with open(saved_path, "wb") as f:
            f.write(contents)

        # If you have any background processing (e.g., unzipping, scanning), do it via background_tasks
        def _background_process_uploaded_zip(path: str):
            try:
                # Example: unzip to folder named <saved_dir>/unzipped
                unzip_to = path + "_unzipped"
                os.makedirs(unzip_to, exist_ok=True)
                with zipfile.ZipFile(path, 'r') as zf:
                    zf.extractall(unzip_to)
                # (Add scanning, indexing, storing to DB/cloud etc.)
            except Exception:
                pass

        background_tasks.add_task(_background_process_uploaded_zip, saved_path)

        return {
            "type": "file",
            "filename": filename,
            "saved_path": saved_path,
            "uploaded_by": current_user.username,
            "uploaded_by_id": current_user.id,
            "personas": personas,
        }

    # --- Handle git_url submission ---
    git_url = git_url.strip() if isinstance(git_url, str) else git_url
    if not await _validate_git_url(git_url):
        raise HTTPException(status_code=400, detail="Invalid Git URL format")

    # We'll validate by attempting a shallow clone in a background task so we don't block the response.
    clone_dest = tempfile.mkdtemp(prefix="repo_clone_request_")

    async def _background_clone_and_process(url: str, dest: str):
        try:
            await _clone_repo_async(url, dest_dir=dest, timeout=120)
            # (Add indexing, scanning, build steps, etc.)
        except Exception as exc:
            # Log the error to your monitoring/logging system
            # If you need synchronous notification to user, store status in DB for later retrieval.
            pass

    # Fire-and-forget clone (background). You can also await it if you want a synchronous response.
    background_tasks.add_task(lambda: asyncio.run(_background_clone_and_process(git_url, clone_dest)))

    return {
        "type": "git_url",
        "git_url": git_url,
        "clone_path": clone_dest,
        "submitted_by": current_user.username,
        "submitted_by_id": current_user.id,
        "personas": personas,
    }


# If you want to run the app directly for local testing:
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
