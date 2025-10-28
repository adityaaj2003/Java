"""
repo_intel_with_system_user_state.py

- Uses a robust SYSTEM prompt + USER prompt pattern for LLM calls.
- Maintains per-repo analysis state in a JSON file.
- Uses PGVector retriever, DuckDuckGo for lightweight web context,
  and ChatOpenAI for the LLM.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# LangChain imports (match your environment; adjust if module paths differ)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# -------------------------
# Conversation / analysis state store
# -------------------------
class RepoStateStore:
    """A lightweight JSON-backed store to persist repo analyses and conversation history."""

    def __init__(self, path: str = "repo_analyses.json"):
        self.path = path
        self._data: Dict[str, Any] = {"repos": {}}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception as e:
                logger.warning("Failed to load state file '%s': %s. Starting fresh.", self.path, e)
                self._data = {"repos": {}}

    def _save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp, self.path)

    def get_repo(self, repo_id: str) -> Dict[str, Any]:
        return self._data["repos"].get(repo_id, {})

    def upsert_repo(self, repo_id: str, payload: Dict[str, Any]):
        now = time.time()
        entry = self._data["repos"].get(repo_id, {"created_at": now, "runs": []})
        entry["updated_at"] = now
        # append run metadata
        run = {"timestamp": now, "payload": payload}
        entry.setdefault("runs", []).append(run)
        # store summary top-level convenience field
        entry["last_summary"] = payload.get("summary") or payload
        self._data["repos"][repo_id] = entry
        self._save()


# -------------------------
# Default sophisticated system prompt
# -------------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are RepoIntelGPT — an expert system for analyzing code repositories, their structure, "
    "entry points, dependencies and frameworks. Follow these instructions strictly:\n"
    "1) Output a valid JSON object (no surrounding markdown) with keys: "
    "repo_type, structure, important_files, entry_point, dependencies, framework, summary.\n"
    "2) Keep 'summary' to 2-4 sentences. Other fields can be arrays/strings as appropriate.\n"
    "3) If something is ambiguous or missing from the provided context, include an 'assumptions' key "
    "listing them.\n"
    "4) If you provide file paths or commands, ensure they are relative to repository root and platform-agnostic.\n"
    "5) If multiple valid options exist, provide them as an array with short pros/cons.\n"
    "6) Avoid hallucinating filenames or dependencies — indicate when information is not present in context.\n"
)


# -------------------------
# LLM wrapper with retries
# -------------------------
class LLMInvoker:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2):
        # ChatOpenAI wrapper from langchain
        self.llm = ChatOpenAI(model=model, temperature=temperature)

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3),
           retry=retry_if_exception_type(Exception))
    def call_with_system_user(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
        """
        Build a chat prompt with explicit system & user messages then call the ChatOpenAI LLM.
        Returns assistant text.
        """
        system_t = SystemMessagePromptTemplate.from_template(system_prompt)
        user_t = HumanMessagePromptTemplate.from_template(user_prompt)
        prompt = ChatPromptTemplate.from_messages([system_t, user_t])

        # Format messages -> depending on LangChain version, you may call format_messages
        messages = prompt.format_messages({})
        # ChatOpenAI expects either prompt or messages; using messages directly is safer across versions.
        # Some ChatOpenAI clients accept messages param or .generate(messages=...). We'll attempt .generate() if available.
        try:
            # .generate returns object with generations -> get text
            result = self.llm.generate(messages)
            # extract text from result (LangChain shapes vary; we try a few fallbacks)
            if hasattr(result, "generations"):
                gen = result.generations[0][0]
                assistant_text = gen.text
            else:
                assistant_text = str(result)
        except Exception:
            # fallback: call .predict with joined templates (less structured but works)
            joined = "\n\n".join([system_prompt, user_prompt])
            assistant_text = self.llm.predict(joined, max_tokens=max_tokens)

        assistant_text = assistant_text.strip()
        return assistant_text


# -------------------------
# Main repo intelligence function
# -------------------------
def see_repository_intelligence(
    repo_id: str,
    postgres_url: str,
    collection_name: str,
    *,
    user_query: Optional[str] = None,
    system_prompt: Optional[str] = None,
    state_store_path: str = "repo_analyses.json",
    retriever_k: int = 8,
) -> Dict[str, Any]:
    """
    Retrieve stored repo chunks from pgvector, optionally augment with web search,
    and use a system+user prompt LLM call to build structured repo intelligence.
    """

    system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    user_query = user_query or "Analyze the repository and detect its type, entry points, dependencies, and framework."

    # 1) load previous state
    store = RepoStateStore(path=state_store_path)
    prev = store.get_repo(repo_id)
    logger.info("Loaded previous state for repo %s: %s", repo_id, "found" if prev else "none")

    # 2) Connect to vector store and retriever
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = PGVector(
        connection_string=postgres_url,
        collection_name=collection_name,
        embedding_function=embeddings,
        use_jsonb=True,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": retriever_k})

    # 3) Fetch relevant repo docs (safely handle if none present)
    try:
        retrieved_docs = retriever.get_relevant_documents(user_query)
    except Exception as e:
        logger.exception("Failed to retrieve documents from pgvector: %s", e)
        retrieved_docs = []

    context_pieces: List[str] = []
    for d in retrieved_docs:
        # Each doc may be a dict-like object, ensure page_content exists.
        text = getattr(d, "page_content", None) or d.get("page_content") if isinstance(d, dict) else None
        if text:
            # Keep some light trimming to avoid over-long prompts
            context_pieces.append(text[:4000])

    context = "\n\n---\n\n".join(context_pieces) if context_pieces else "NO_LOCAL_CONTEXT_FOUND"

    # 4) Get web augmentation
    try:
        web_search = DuckDuckGoSearchResults()
        web_results = web_search.run(f"{user_query} repository type and framework details")
        # compact web info to a string
        web_info = "\n".join([r.get("snippet") or str(r) for r in (web_results or [])])[:4000]
    except Exception as e:
        logger.info("Web search failed or not available: %s", e)
        web_info = "NO_WEB_RESULTS"

    # 5) Compose the user prompt combining the dynamic pieces
    composed_user_prompt = (
        f"Repository ID: {repo_id}\n\n"
        f"Local Context (from pgvector retriever):\n{context}\n\n"
        f"Web Augmentation:\n{web_info}\n\n"
        f"User Task:\n{user_query}\n\n"
        "Produce a strict JSON object with the keys described in the system prompt. "
        "If something is missing in the context, say so under the 'assumptions' key."
    )

    # 6) Call LLM with system + user prompt
    llm_invoker = LLMInvoker()
    try:
        assistant_text = llm_invoker.call_with_system_user(system_prompt=system_prompt, user_prompt=composed_user_prompt)
    except Exception as e:
        logger.exception("LLM invocation failed: %s", e)
        assistant_text = json.dumps({"error": "LLM invocation failed", "detail": str(e)})

    # 7) Try to parse assistant output as JSON (some LLMs might include explanation — we try to extract JSON)
    structured_output: Dict[str, Any] = {}
    assistant_text_stripped = assistant_text.strip()
    # attempt direct JSON parse
    try:
        structured_output = json.loads(assistant_text_stripped)
    except Exception:
        # attempt to find first JSON object inside text
        import re

        match = re.search(r"(\{(?:.|\n)*\})", assistant_text_stripped)
        if match:
            try:
                structured_output = json.loads(match.group(1))
            except Exception:
                logger.warning("Failed to parse JSON inside LLM output; returning raw assistant text.")
                structured_output = {"summary": assistant_text_stripped}
        else:
            structured_output = {"summary": assistant_text_stripped}

    # enrich with meta
    structured_output.setdefault("repo_id", repo_id)
    structured_output.setdefault("collection_name", collection_name)
    # store run metadata into persistent state
    run_payload = {
        "user_query": user_query,
        "system_prompt_used": system_prompt[:2000],
        "assistant_raw": assistant_text_stripped[:16000],
        "result": structured_output,
    }
    store.upsert_repo(repo_id, run_payload)

    return structured_output


# -------------------------
# Quick demo when executed directly
# -------------------------
if __name__ == "__main__":
    # Replace with your real connection string and collection name
    postgres_url = os.getenv("POSTGRES_URL", "postgresql+psycopg://postgres:password@localhost:5432/vector_db")
    collection_name = os.getenv("COLLECTION_NAME", "repo_a1b2c3d4")
    repo_id = "a1b2c3d4"

    result = see_repository_intelligence(
        repo_id=repo_id,
        postgres_url=postgres_url,
        collection_name=collection_name,
        user_query="Analyze the repository and list its entry points, main dependencies and frameworks used."
    )
    print(json.dumps(result, indent=2))
