"""
Smart Preprocessing Starter for Repository Intelligence
- Purpose: identify repo type/structure, detect important files, extract metadata,
  parse config files, detect entrypoints, generate embeddings, and index to pgvector.
- Tech stack assumptions: FastAPI, WebSockets, LangGraph orchestration, pgvector (Postgres),
  Azure OpenAI for embeddings, Langfuse for observability. External MCP servers are used for
  heavy file operations / knowledge-base tools. This file provides a runnable starter scaffold
  with clear TODOs for production wiring.

Notes:
- Replace placeholders like AZURE_*, POSTGRES_DSN, LANGFUSE_KEY, MCP_SERVER_URL with real values.
- This file aims to be modular: each component (repo intelligence, file processing, indexing)
  can be swapped or extended.

Requirements (pip):
fastapi
uvicorn[standard]
httpx
gitpython
python-multipart
pydantic
aiofiles
asyncpg
sqlalchemy
psycopg2-binary
openai  # or azure-openai SDK wrapper depending on your infra
pgvector
langgraph-sdk  # hypothetical; adapt to your orchestration lib
langfuse-sdk   # hypothetical
"""

from typing import List, Optional, Dict, Any
import os
import re
import shutil
import hashlib
import tempfile
import subprocess
import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, WebSocket
from pydantic import BaseModel
import asyncio

# --- Configuration (replace with env vars in production) ---
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "<REPLACE>")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "<REPLACE>")
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://user:pass@localhost:5432/codevec")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://mcp-server.local")
LANGFUSE_API_KEY = os.getenv("LANGFUSE_API_KEY", "<REPLACE>")

# --- Helper utilities ---

def sha1_of_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


async def run_cmd(cmd: List[str], cwd: Optional[str] = None) -> Dict[str, Any]:
    """Run a shell command asynchronously and return stdout/stderr/returncode."""
    proc = await asyncio.create_subprocess_exec(*cmd, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await proc.communicate()
    return {"returncode": proc.returncode, "stdout": out.decode(errors="ignore"), "stderr": err.decode(errors="ignore")}


# --- Repository Intelligence ---
class RepoSummary(BaseModel):
    repo_type: Optional[str]
    language_breakdown: Dict[str, int]
    important_files: List[str]
    entrypoints: List[str]
    detected_frameworks: List[str]
    config_files: List[str]


COMMON_ENTRYPOINT_NAMES = ["main.py", "app.py", "index.js", "server.js", "package.json", "pyproject.toml", "manage.py"]
CONFIG_FILENAMES = ["package.json", "pyproject.toml", "requirements.txt", "Pipfile", "setup.py", "pom.xml", "build.gradle", "Dockerfile", "Procfile"]


def detect_language_by_ext(path: Path) -> Optional[str]:
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".cpp": "cpp",
        ".c": "c",
        ".php": "php",
        ".html": "html",
        ".css": "css",
    }
    return mapping.get(path.suffix.lower())


def analyze_repo_local(path: str) -> RepoSummary:
    """Scan a local repository directory for high-level intelligence.
    - language breakdown by file count
    - detect well-known config files and entrypoints
    - pick 'important' files heuristically (README, Dockerfile, CI config, LICENSE, major src dirs)
    """
    root = Path(path)
    file_counts: Dict[str, int] = {}
    important_files: List[str] = []
    entrypoints: List[str] = []
    detected_frameworks: List[str] = []
    config_files_found: List[str] = []

    for p in root.rglob("*"):
        if p.is_file():
            lang = detect_language_by_ext(p)
            if lang:
                file_counts[lang] = file_counts.get(lang, 0) + 1
            name = p.name.lower()
            if name in (n.lower() for n in COMMON_ENTRYPOINT_NAMES):
                entrypoints.append(str(p.relative_to(root)))
            if name in (c.lower() for c in CONFIG_FILENAMES):
                config_files_found.append(str(p.relative_to(root)))
            # heuristics for important files
            if p.name.lower() in ["readme.md", "readme", "license", "dockerfile"] or p.suffix in [".yml", ".yaml"] and "ci" in p.stem.lower():
                important_files.append(str(p.relative_to(root)))

    # quick framework detection
    if "package.json" in [os.path.basename(f) for f in config_files_found]:
        # try to detect frameworks by scanning package.json
        try:
            pkg_json_path = root / "package.json"
            if pkg_json_path.exists():
                data = json.loads(pkg_json_path.read_text(encoding="utf-8"))
                deps = list((data.get("dependencies") or {}).keys()) + list((data.get("devDependencies") or {}).keys())
                for d in deps:
                    if d.lower().startswith("react"):
                        detected_frameworks.append("react")
                    if d.lower().startswith("express"):
                        detected_frameworks.append("express")
        except Exception:
            pass

    # infer repo type
    repo_type = None
    if "python" in file_counts:
        repo_type = "python"
    elif "javascript" in file_counts:
        repo_type = "javascript"

    return RepoSummary(
        repo_type=repo_type,
        language_breakdown=file_counts,
        important_files=important_files,
        entrypoints=entrypoints,
        detected_frameworks=list(set(detected_frameworks)),
        config_files=config_files_found,
    )


# --- Smart File Processing ---
class FileMetadata(BaseModel):
    path: str
    size: int
    sha1: str
    language: Optional[str]
    summary: Optional[str]


def extract_file_metadata(root: str, relpath: str) -> FileMetadata:
    p = Path(root) / relpath
    text = p.read_text(encoding="utf-8", errors="ignore")
    return FileMetadata(
        path=relpath,
        size=p.stat().st_size,
        sha1=sha1_of_text(text),
        language=detect_language_by_ext(p),
        summary=(text[:400] + "...") if len(text) > 400 else text,
    )


def detect_entrypoints_by_heuristics(root: str) -> List[str]:
    rootp = Path(root)
    found = []
    for name in COMMON_ENTRYPOINT_NAMES:
        if (rootp / name).exists():
            found.append(name)
    # fallback scanning: look for if __name__ == '__main__' pattern in py files
    for p in rootp.rglob("*.py"):
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
            if "if __name__ == '__main__'" in t or 'if __name__ == "__main__"' in t:
                found.append(str(p.relative_to(rootp)))
        except Exception:
            pass
    return list(set(found))


# --- Embedding & Indexing (pgvector) ---
# NOTE: This is a simple example: in production use batching, rate-limits, retries, and Langfuse tracing.
async def embed_texts_azure(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using Azure OpenAI embeddings endpoint.
    Replace with your preferred client.
    """
    # Pseudo-code to show expected shape; implement with your SDK (openai/azure-ai-openai) in prod.
    # For now, raise if credentials not set.
    if AZURE_OPENAI_API_KEY.startswith("<REPLACE>"):
        raise RuntimeError("Azure OpenAI credentials not configured")
    # TODO: implement actual call to Azure OpenAI embeddings here
    # return [[0.0]*1536 for _ in texts]
    raise NotImplementedError("Implement Azure OpenAI embeddings call")


async def index_embeddings_postgres(records: List[Dict[str, Any]]):
    """Index into a Postgres table with pgvector. The table schema assumed:
    CREATE TABLE code_vectors (
        id TEXT PRIMARY KEY,
        path TEXT,
        repo TEXT,
        metadata JSONB,
        embedding VECTOR(1536)
    );
    """
    # Implement with asyncpg or SQLAlchemy. This is a placeholder.
    raise NotImplementedError("Implement indexing to Postgres/pgvector")


# --- LangGraph Orchestration (pseudo) ---
# This is a scaffold showing how you'd break the pipeline into tasks. Replace 'langgraph' calls with your orchestrator.
async def orchestrate_analysis(repo_path: str, repo_name: str) -> Dict[str, Any]:
    """Top-level orchestrator that would be driven by LangGraph multi-agent flows.
    Steps:
      1. Repo analysis
      2. Detect entrypoints and important files
      3. For each important file: extract metadata, chunk, embed, index
      4. Trigger downstream knowledge-base updates (MCP server)
    """
    summary = analyze_repo_local(repo_path)
    entrypoints = detect_entrypoints_by_heuristics(repo_path)
    # choose interesting files (for demo: README + all .py/.js under src/ or root)
    candidates = set(summary.important_files)
    for ext in (".py", ".js", ".ts"):
        for p in Path(repo_path).rglob(f"*{ext}"):
            candidates.add(str(p.relative_to(repo_path)))

    records_to_index = []
    for rel in candidates:
        try:
            md = extract_file_metadata(repo_path, rel)
            # chunking strategy: simple fixed-size window on characters (replace with code-aware chunker)
            text = Path(repo_path).joinpath(rel).read_text(encoding="utf-8", errors="ignore")
            chunk_size = 2000
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                record_id = f"{repo_name}:{rel}:{i}:{sha1_of_text(chunk)}"
                records_to_index.append({
                    "id": record_id,
                    "path": rel,
                    "repo": repo_name,
                    "text": chunk,
                    "metadata": {"file_sha1": md.sha1, "offset": i},
                })
        except Exception as e:
            # log and continue
            print("error extracting", rel, e)

    # next: generate embeddings in batches
    # embeddings = await embed_texts_azure([r['text'] for r in records_to_index])
    # for rec, emb in zip(records_to_index, embeddings): rec['embedding'] = emb
    # await index_embeddings_postgres(records_to_index)

    # notify MCP server (placeholder)
    # await notify_mcp_indexed(repo_name, len(records_to_index))

    return {"repo_summary": summary.dict(), "indexed": len(records_to_index)}


# --- MCP server hooks (placeholders) ---
async def notify_mcp_indexed(repo: str, count: int):
    async with httpx.AsyncClient() as client:
        try:
            await client.post(f"{MCP_SERVER_URL}/indexed", json={"repo": repo, "count": count})
        except Exception as e:
            print("Failed to notify MCP", e)


# --- FastAPI server ---
app = FastAPI(title="Repo Intelligence / Smart Preprocessing API")


class AnalyzeRequest(BaseModel):
    git_url: Optional[str] = None
    repo_name: Optional[str] = None


@app.post("/analyze-repo")
async def analyze_repo(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    """Accepts either a git_url (public) or expects repo already mounted on disk by MCP.
    This endpoint triggers orchestration in background and returns an immediate job token.
    """
    if not req.git_url:
        return {"error": "provide git_url"}

    # clone to a temp dir (synchronous for simplicity) -- in prod, perform on MCP or worker
    tmp = tempfile.mkdtemp(prefix="repo_clone_")
    cmd = ["git", "clone", "--depth", "1", req.git_url, tmp]
    res = await run_cmd(cmd)
    if res["returncode"] != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        return {"error": "git clone failed", "details": res}

    repo_name = req.repo_name or Path(req.git_url).stem
    # schedule background orchestration
    background_tasks.add_task(orchestrate_and_cleanup, tmp, repo_name)
    return {"status": "started", "repo_name": repo_name}


async def orchestrate_and_cleanup(path: str, repo_name: str):
    try:
        out = await orchestrate_analysis(path, repo_name)
        # optionally send results to MCP
        await notify_mcp_indexed(repo_name, out.get("indexed", 0))
    except Exception as e:
        print("Orchestration failed", e)
    finally:
        shutil.rmtree(path, ignore_errors=True)


@app.websocket("/ws/analysis")
async def websocket_analysis(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("action") == "status":
                await ws.send_json({"status": "ok", "message": "no-op"})
            else:
                await ws.send_json({"error": "unknown action"})
    except Exception:
        await ws.close()


# --- Langfuse tracing stubs (observability) ---
def langfuse_trace(event: str, payload: Dict[str, Any]):
    # placeholder: wire to Langfuse SDK with API key and send trace
    print(f"[langfuse] {event}", payload)


# --- Example CLI usage ---
if __name__ == "__main__":
    import uvicorn
    print("Starting server on http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
