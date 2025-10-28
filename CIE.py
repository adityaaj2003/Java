from enum import Enum
from typing import Optional
from fastapi import File, Form, UploadFile, BackgroundTasks, HTTPException, Depends
import tempfile, os

# Persona Enum for dropdown
class Persona(str, Enum):
    SDE = "SDE"
    PM = "PM"
    Both = "Both"

MAX_UPLOAD_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB

@app.post("/submit-project")
async def submit_project(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None, description="Upload a ZIP file if not using Git URL"),
    git_url: Optional[str] = Form(None, description="Provide Git repository URL if not uploading a file"),
    persona: Persona = Form(..., description="Select persona"),  # required dropdown in Swagger
    current_user: User = Depends(get_current_user),
):
    # Require exactly one submission method
    if (file is None or getattr(file, "filename", None) in (None, "")) and (git_url is None or not str(git_url).strip()):
        raise HTTPException(status_code=400, detail="Provide either a file or Git URL.")
    if file is not None and git_url and str(git_url).strip():
        raise HTTPException(status_code=400, detail="Provide only one submission type (file OR Git URL).")

    # -------- Handle file upload --------
    if file is not None and getattr(file, "filename", "") != "":
        filename = file.filename or ""
        if not filename.lower().endswith(".zip"):
            raise HTTPException(status_code=400, detail="Only .zip files are accepted.")

        # Read in chunks to enforce size limit
        await file.seek(0)
        total = 0
        chunks = []
        while True:
            chunk = await file.read(1024 * 1024)  # 1 MB
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_SIZE_BYTES:
                raise HTTPException(status_code=400, detail="File too large. Limit 100MB.")
            chunks.append(chunk)
        contents = b"".join(chunks)

        # Validate ZIP integrity by trying to open it (helper _is_zip_file_bytes assumed present)
        if not _is_zip_file_bytes(contents):
            raise HTTPException(status_code=400, detail="Corrupted or invalid ZIP file.")

        # Save to temp dir
        saved_dir = tempfile.mkdtemp(prefix="uploaded_project_")
        saved_path = os.path.join(saved_dir, filename)
        with open(saved_path, "wb") as out_f:
            out_f.write(contents)

        # Optionally add background task for processing
        def _bg_process(path: str):
            try:
                # e.g., unzip, scan, index
                pass
            except Exception:
                pass

        background_tasks.add_task(_bg_process, saved_path)

        return {
            "type": "file",
            "filename": filename,
            "persona": persona.value,
            "uploaded_by": current_user.username,
            "message": "File accepted and saved.",
            "saved_path": saved_path,
        }

    # -------- Handle git_url --------
    git_url = str(git_url).strip() if git_url is not None else ""
    if not git_url:
        raise HTTPException(status_code=400, detail="Git URL is empty or invalid.")

    try:
        dest = await _clone_repo_shallow(git_url)  # use your existing helper that raises RuntimeError on known failures
        scan = _scan_repo_for_code(dest)           # use your repo scanner helper
        if scan["total_files"] == 0 or scan["code_files_count"] == 0:
            # cleanup clone if desired
            shutil.rmtree(dest, ignore_errors=True)
            raise HTTPException(status_code=400, detail="Repository has no recognizable code files.")
    except RuntimeError as rexc:
        raise HTTPException(status_code=400, detail=str(rexc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error cloning/scanning repo: {exc}")

    # background processing example (do not block response)
    async def _bg_repo_process(path: str):
        try:
            # long processing here
            pass
        finally:
            return

    background_tasks.add_task(lambda: asyncio.run(_bg_repo_process(dest)))

    return {
        "type": "git_url",
        "git_url": git_url,
        "persona": persona.value,
        "submitted_by": current_user.username,
        "repo_summary": scan,
        "message": "Repository cloned and scanned successfully."
    }
