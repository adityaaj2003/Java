from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Any, Dict
import tempfile
import zipfile
import shutil
import os
import asyncio
import subprocess
import json

app = FastAPI()

# --- Configuration ---
MAX_UPLOAD_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB
GIT_CLONE_TIMEOUT_SECONDS = 60
ALLOWED_ZIP_EXT = ('.zip',)
CODE_FILE_EXTS = {'.py', '.java', '.js', '.ts', '.c', '.cpp', '.go', '.rs', '.php', '.html', '.css', '.swift', '.kt', '.m', '.r'}
DOCUMENT_EXTS = {'.md', '.txt', '.rst', '.pdf', '.docx'}
BINARY_EXTS = {'.exe', '.dll', '.so', '.bin', '.dat'}

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
            _ = zf.namelist()
        return True
    except zipfile.BadZipFile:
        return False
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def _detect_archive_type_bytes(data: bytes) -> Optional[str]:
    """Detect some common archive signatures by magic bytes for clearer errors."""
    if data.startswith(b'PK\x03\x04'):
        return 'zip'
    if data.startswith(b'Rar!') or data[:7].startswith(b'Rar!'):
        return 'rar'
    if data.startswith(b'7z\xBC\xAF\x27\x1C'):
        return '7z'
    return None


async def _run_subprocess(cmd: List[str], timeout: int = 60) -> subprocess.CompletedProcess:
    """Run a subprocess in a thread to avoid blocking the event loop."""
    return await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True, timeout=timeout)


async def _clone_repo_shallow(git_url: str, dest_dir: Optional[str] = None, timeout: int = GIT_CLONE_TIMEOUT_SECONDS) -> str:
    """
    Attempt a shallow clone (depth=1) and return the destination path. Raises RuntimeError with clear messages on failure.
    """
    if dest_dir is None:
        dest_dir = tempfile.mkdtemp(prefix="repo_clone_")

    cmd = ["git", "clone", "--depth", "1", git_url, dest_dir]

    try:
        proc = await _run_subprocess(cmd, timeout=timeout)
        if proc.returncode != 0:
            stderr = (proc.stderr or "").lower()
            # Provide specific, user-friendly messages for common git failures
            if 'repository not found' in stderr or 'not found' in stderr:
                raise RuntimeError("Repository not found (404) or the URL is malformed.")
            if 'access denied' in stderr or 'authentication failed' in stderr or 'permission denied' in stderr:
                raise RuntimeError("Access denied or private repository — authentication required.")
            if 'could not resolve host' in stderr or 'name or service not known' in stderr:
                raise RuntimeError("Could not resolve host — check the repository URL or your network.")
            raise RuntimeError(f"git clone failed: {proc.stderr.strip()}")
        return dest_dir
    except subprocess.TimeoutExpired:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise RuntimeError("git clone timed out")
    except Exception:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise


def _scan_repo_for_code(path: str) -> Dict[str, Any]:
    """
    Scan the checked-out repository for code files, documentation, binary heavy repos, and return a small summary.
    """
    code_files = []
    doc_files = []
    binary_files = []
    total_files = 0

    for root, dirs, files in os.walk(path):
        # skip .git folder content
        if '.git' in root.split(os.sep):
            continue
        for f in files:
            total_files += 1
            _, ext = os.path.splitext(f.lower())
            if ext in CODE_FILE_EXTS:
                code_files.append(os.path.join(root, f))
            elif ext in DOCUMENT_EXTS:
                doc_files.append(os.path.join(root, f))
            elif ext in BINARY_EXTS:
                binary_files.append(os.path.join(root, f))
            else:
                # Heuristic: treat unknowns as possible code if small and text-like
                p = os.path.join(root, f)
                try:
                    if os.path.getsize(p) > 5 * 1024 * 1024:  # >5MB likely binary
                        binary_files.append(p)
                    else:
                        # try to sample to see if it's text
                        with open(p, 'rb') as fh:
                            sample = fh.read(1024)
                            if b'\0' in sample:
                                binary_files.append(p)
                            else:
                                # assume text-like; could be code or docs
                                # Use extension-less heuristics
                                if ext == '':
                                    # check for shebang or common code tokens
                                    if sample.startswith(b'#!') or b'import ' in sample or b'class ' in sample:
                                        code_files.append(p)
                                    else:
                                        doc_files.append(p)
                                else:
                                    doc_files.append(p)
                except Exception:
                    # If we can't read, treat as binary to be safe
                    binary_files.append(p)

    summary = {
        'total_files': total_files,
        'code_files_count': len(code_files),
        'doc_files_count': len(doc_files),
        'binary_files_count': len(binary_files),
        'sample_code_files': code_files[:10],
        'sample_doc_files': doc_files[:10],
    }
    return summary


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
    This implementation provides clear, actionable error messages for common failure modes such as corrupted archives,
    wrong file formats (RAR/7z), oversized uploads, empty/non-code repositories, malformed or private GitHub URLs, etc.
    """

    # --- Parse personas if provided ---
    personas: Optional[List[Any]] = None
    if personas_json:
        try:
            personas = json.loads(personas_json)
            if not isinstance(personas, list):
                raise ValueError("personas must be a JSON array")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid personas JSON: {str(exc)}")

    # --- Require exactly one submission method ---
    if (file is None and (git_url is None or str(git_url).strip() == "")):
        raise HTTPException(status_code=400, detail="You must provide a file or a Git URL.")
    if file is not None and git_url and str(git_url).strip() != "":
        raise HTTPException(status_code=400, detail="Provide only one submission method: either a file or a Git URL, not both.")

    # --- Handle file upload ---
    if file is not None:
        filename = file.filename or ""
        ext = os.path.splitext(filename.lower())[1]

        # Quick extension check
        if ext not in ALLOWED_ZIP_EXT:
            # attempt to read beginning of file to give better guidance (in case extension lies)
            first_chunk = await file.read(4096)
            arch_type = _detect_archive_type_bytes(first_chunk)
            if arch_type == 'rar':
                raise HTTPException(status_code=400, detail="RAR archives are not supported. Please provide a ZIP file.")
            if arch_type == '7z':
                raise HTTPException(status_code=400, detail="7z archives are not supported. Please provide a ZIP file.")
            raise HTTPException(status_code=400, detail="Only .zip files are accepted based on filename.")

        # Read file in chunks while enforcing size limit
        await file.seek(0)
        bytes_read = 0
        chunks = []
        while True:
            chunk = await file.read(1024 * 1024)  # 1MB
            if not chunk:
                break
            bytes_read += len(chunk)
            if bytes_read > MAX_UPLOAD_SIZE_BYTES:
                raise HTTPException(status_code=400, detail=f"Uploaded file is too large. Maximum allowed size is {MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB.")
            chunks.append(chunk)
        contents = b"".join(chunks)

        # Validate ZIP integrity
        if not _is_zip_file_bytes(contents):
            # If we detected another archive type, return that information
            arch_type = _detect_archive_type_bytes(contents[:4096])
            if arch_type in ('rar', '7z'):
                raise HTTPException(status_code=400, detail=f"Uploaded archive appears to be {arch_type.upper()}, which is not supported. Please upload a ZIP.")
            raise HTTPException(status_code=400, detail="Uploaded file appears to be a corrupted or incomplete ZIP archive.")

        # Save to permanent storage (demo: temp dir)
        saved_dir = tempfile.mkdtemp(prefix="uploaded_project_")
        saved_path = os.path.join(saved_dir, filename)
        with open(saved_path, "wb") as f:
            f.write(contents)

        # Optionally perform background processing such as unzipping and lightweight scanning
        def _background_process_uploaded_zip(path: str):
            try:
                unzip_to = path + "_unzipped"
                os.makedirs(unzip_to, exist_ok=True)
                with zipfile.ZipFile(path, 'r') as zf:
                    zf.extractall(unzip_to)
                # Scan for code files and update DB/state if needed
                # (This is synchronous background work — keep it light.)
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
            "note": "File accepted and saved. Integrity checked (ZIP). Background processing started.",
        }

    # --- Handle git_url submission ---
    git_url = str(git_url).strip() if git_url is not None else None
    if not git_url:
        raise HTTPException(status_code=400, detail="Git URL is empty or invalid.")

    # Basic shape validation — permissive. For GitHub/GitLab, you could add stricter regexes.
    if git_url.startswith('/') or git_url.startswith('file://'):
        raise HTTPException(status_code=400, detail="Local file paths are not allowed for repository URL.")

    # Attempt to shallow clone synchronously but without blocking event loop
    clone_dest = tempfile.mkdtemp(prefix='repo_clone_request_')
    try:
        dest = await _clone_repo_shallow(git_url, dest_dir=clone_dest, timeout=GIT_CLONE_TIMEOUT_SECONDS)
    except RuntimeError as rexc:
        # Provide the user a clear 400 with reason
        raise HTTPException(status_code=400, detail=str(rexc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error while cloning repository: {str(exc)}")

    # If clone succeeded, scan repository for code vs docs vs binaries
    try:
        scan = _scan_repo_for_code(dest)
    except Exception as exc:
        # cleanup on failure
        shutil.rmtree(dest, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to scan repository: {str(exc)}")

    # Assess and provide clear feedback
    if scan['total_files'] == 0:
        shutil.rmtree(dest, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Repository is empty.")

    if scan['code_files_count'] == 0:
        # No recognizable code files. Clean up clone and inform user.
        shutil.rmtree(dest, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Repository contains no recognizable code files (may be documentation-only or binary-only).")

    # If we get here, we have at least some code files. Kick off any heavy background processing (CI, indexing) asynchronously
    async def _background_process_repo(path: str):
        try:
            # Example: run static analyzers, indexing, packaging, etc.
            # Keep this robust and idempotent. If long, consider queuing externally.
            pass
        finally:
            # Optionally keep or remove the clone depending on your retention policy.
            # shutil.rmtree(path, ignore_errors=True)
            return

    # Use FastAPI background tasks to continue processing without blocking the response
    background_tasks.add_task(lambda: asyncio.run(_background_process_repo(dest)))

    return {
        "type": "git_url",
        "git_url": git_url,
        "clone_path": dest,
        "submitted_by": current_user.username,
        "submitted_by_id": current_user.id,
        "personas": personas,
        "repo_summary": scan,
        "note": "Repository cloned (shallow). Background processing started. If private repository, ensure server has access or provide credentials.",
    }


# If you want to run the app directly for local testing:
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
