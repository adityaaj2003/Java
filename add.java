#!/usr/bin/env python3
"""
summarize_repo.py

Walk a directory, feed each file to an LLM in chunks, and save summaries.

Usage:
    export OPENAI_API_KEY="sk-..."
    python summarize_repo.py /path/to/repo --out summaries.json
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import List, Dict

import openai  # pip install openai

# ---------- Config ----------
MODEL = "gpt-4o-mini"            # change to your preferred model
MAX_TOKENS_PER_REQUEST = 2400    # keep response + prompt tokens under model cap
CHUNK_SIZE = 2500                # approx characters per chunk (simple heuristic)
TEMPERATURE = 0.0                # deterministic summaries
# -----------------------------

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

IGNORED_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}


def discover_files(root: str, exts: List[str] = None) -> List[Path]:
    """Return list of file paths under root, optionally filtered by extensions."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # skip ignored dirs
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        for fname in filenames:
            fp = Path(dirpath) / fname
            if exts:
                if fp.suffix.lower() in exts:
                    files.append(fp)
            else:
                files.append(fp)
    return files


def read_file_text(path: Path) -> str:
    """Read text file as UTF-8; if fails, return empty string."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"[WARN] Couldn't read {path}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """
    Naive chunker by characters that tries to split by newline boundaries when possible.
    Returns list of chunks (strings).
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        # try to expand end to next newline for a cleaner split (up to +200 chars)
        if end < n:
            extra_end = min(n, end + 200)
            newline_idx = text.rfind("\n", start, extra_end)
            if newline_idx > start:
                end = newline_idx + 1
        chunks.append(text[start:end])
        start = end
    return chunks


def build_system_prompt(file_path: str, language_hint: str = None) -> str:
    return (
        "You are a helpful code summarization assistant. "
        "Given a code snippet, produce a short summary (3-8 sentences) that includes:\n"
        "- What the code/file does (purpose)\n"
        "- Key functions/classes and their responsibilities\n"
        "- Important implementation details or caveats\n"
        "- Suggested tests or things a reviewer should look for (brief)\n\n"
        f"File path: {file_path}\n"
        + (f"Language hint: {language_hint}\n" if language_hint else "")
        + "Return a JSON object with keys: 'summary' (string), 'highlights' (list of short strings).\n"
        "ONLY return a JSON object with those keys (no extra text)."
    )


def call_llm_system_messages(system_prompt: str, user_content: str) -> dict:
    """
    Call the LLM with a system + user message.
    Return the parsed JSON from the model's response.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # Synchronous API call
    resp = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS_PER_REQUEST,
    )

    text = resp["choices"][0]["message"]["content"].strip()

    # The system instructed a JSON object; try to parse it robustly
    try:
        parsed = json.loads(text)
        return parsed
    except json.JSONDecodeError:
        # try to extract a JSON substring
        import re

        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        # fallback: wrap the whole reply as a summary
        return {"summary": text, "highlights": []}


def summarize_file(path: Path) -> Dict:
    """Create a summary for a single file by summarizing chunks and then combining."""
    text = read_file_text(path)
    if not text.strip():
        return {"path": str(path), "summary": "", "highlights": [], "note": "empty or binary file"}

    # language hint from extension
    ext = path.suffix.lower()
    lang_map = {
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".java": "Java",
        ".cpp": "C++",
        ".c": "C",
        ".cs": "C#",
        ".html": "HTML",
        ".css": "CSS",
        ".go": "Go",
        ".rs": "Rust",
        ".md": "Markdown",
        ".sh": "Shell",
        ".json": "JSON",
    }
    language_hint = lang_map.get(ext, None)

    chunks = chunk_text(text)
    chunk_summaries = []
    chunk_highlights = []

    for i, chunk in enumerate(chunks):
        system_prompt = build_system_prompt(str(path) + f" [chunk {i+1}/{len(chunks)}]", language_hint)
        user_content = (
            "Here is the code chunk:\n\n"
            "-----BEGIN CHUNK-----\n"
            + chunk
            + "\n-----END CHUNK-----\n\n"
            "Summarize as requested."
        )
        parsed = call_llm_system_messages(system_prompt, user_content)
        summary = parsed.get("summary") or ""
        highlights = parsed.get("highlights") or []
        chunk_summaries.append(summary)
        chunk_highlights.extend(highlights)

    # Combine chunk summaries into a single file summary by asking the model to condense them
    combined_prompt = (
        "You were given the following chunk summaries for the same file. "
        "Produce a concise unified summary (3-6 sentences) and a short list of the top 4 highlights.\n\n"
        "Chunk summaries:\n"
    )
    for idx, s in enumerate(chunk_summaries, 1):
        combined_prompt += f"\n--- chunk {idx} ---\n{s}\n"

    system_prompt_final = (
        "You are an assistant that merges and distills multiple chunk summaries into a single, concise file-level summary. "
        "Return a JSON object with keys: 'summary' and 'highlights' (list of up to 4 strings)."
    )
    final_parsed = call_llm_system_messages(system_prompt_final, combined_prompt)

    return {
        "path": str(path),
        "summary": final_parsed.get("summary", "").strip(),
        "highlights": final_parsed.get("highlights", chunk_highlights)[:4],
        "chunks": len(chunks),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize code files using an LLM.")
    parser.add_argument("root", help="Repository root directory to summarize")
    parser.add_argument("--out", default="summaries.json", help="Output JSON filename")
    parser.add_argument("--ext", nargs="*", help="Optional list of file extensions to include (e.g. .py .js)")
    args = parser.parse_args()

    root = args.root
    out_file = args.out
    exts = [e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.ext] if args.ext else None

    files = discover_files(root, exts)
    print(f"Found {len(files)} files to summarize.")

    all_summaries = []
    for idx, path in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] Summarizing: {path} ...")
        try:
            file_summary = summarize_file(path)
            all_summaries.append(file_summary)
        except Exception as e:
            print(f"[ERROR] Failed summarizing {path}: {e}")
            all_summaries.append({"path": str(path), "error": str(e)})

    # Save to output JSON
    out_obj = {"root": os.path.abspath(root), "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z", "summaries": all_summaries}
    Path(out_file).write_text(json.dumps(out_obj, indent=2, ensure_ascii=False))
    print(f"Saved summaries to {out_file}")


if __name__ == "__main__":
    main()
   
