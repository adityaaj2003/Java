import os
import zipfile
import tempfile
from langchain_community.vectorstores.pgvector import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ---------------------
# ⚙️ CONFIGURATION
# ---------------------
CONNECTION = "postgresql+psycopg2://postgres:your_password@localhost:5432/repo_intelligence"
COLLECTION_NAME = "repo_chunks"

embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key="YOUR_OPENAI_API_KEY"
)

# ---------------------
# 🧠 SMART FILTERING
# ---------------------
def should_skip_file(file_path: str) -> bool:
    """Skip irrelevant or non-source files for Python, Java, JS projects."""
    path_lower = file_path.lower()

    skip_dirs = [
        "node_modules", ".git", "__pycache__", "venv", "env",
        "dist", "build", ".idea", ".vscode", "target", "out",
        ".gradle", ".mvn", "test-results", "migrations", "cache", "logs"
    ]

    skip_exts = [
        ".png", ".jpg", ".jpeg", ".gif", ".exe", ".zip", ".tar", ".gz",
        ".rar", ".dll", ".so", ".class", ".jar", ".lock", ".log", ".db",
        ".sqlite", ".env", ".pyc", ".pdf", ".docx", ".csv", ".xml", ".iml"
    ]

    include_exts = [
        ".py", ".java", ".js", ".ts", ".jsx", ".tsx",
        ".html", ".css", ".json", ".yaml", ".yml", ".md"
    ]

    # Skip by directory name
    if any(skip in path_lower for skip in skip_dirs):
        return True

    # Skip by extension
    ext = os.path.splitext(path_lower)[1]
    if ext in skip_exts:
        return True

    # Only process included extensions
    if ext and ext not in include_exts:
        return True

    return False


# ---------------------
# 📦 ZIP PROCESSING
# ---------------------
def process_zip_repo(zip_path: str):
    """Extract ZIP → Filter files → Chunk → Store in PGVector."""
    temp_dir = tempfile.mkdtemp()
    repo_name = os.path.basename(zip_path).replace(".zip", "")

    # Step 1: Extract ZIP
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)
    print(f"✅ Extracted: {repo_name}")

    # Step 2: Collect all relevant files
    useful_files = []
    for root, _, files in os.walk(temp_dir):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, temp_dir)
            if not should_skip_file(file_path):
                useful_files.append(rel_path)

    if not useful_files:
        print("⚠️ No relevant files found.")
        return None

    print(f"📄 Found {len(useful_files)} source files in {repo_name}")

    # Step 3: Chunk and prepare documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = []

    for rel_path in useful_files:
        abs_path = os.path.join(temp_dir, rel_path)
        try:
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            print(f"⚠️ Skipping unreadable file {rel_path}: {e}")
            continue

        if not content.strip():
            continue

        chunks = splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "repo_name": repo_name,
                    "file_name": rel_path,
                    "chunk_index": i + 1,
                    "total_chunks": len(chunks),
                    "all_files": useful_files,
                },
            )
            documents.append(doc)

    print(f"✅ Created {len(documents)} total chunks from {repo_name}")

    # Step 4: Store in PGVector
    try:
        vectorstore = PGVector.from_documents(
            documents=documents,
            embedding=embedding,
            connection_string=CONNECTION,
            collection_name=COLLECTION_NAME,
        )
        print(f"🎯 Stored {len(documents)} chunks for '{repo_name}' in PGVector.")
        return vectorstore
    except Exception as e:
        print(f"❌ Error storing in PGVector: {e}")
        return None


# ---------------------
# 🔍 QUERY FUNCTION
# ---------------------
def query_repo(question: str, top_k: int = 3, file_name: str = None, repo_name: str = None):
    """Semantic query across stored repo data."""
    vectorstore = PGVector(
        connection_string=CONNECTION,
        embedding_function=embedding,
        collection_name=COLLECTION_NAME,
    )

    filters = {}
    if file_name:
        filters["file_name"] = {"$eq": file_name}
    if repo_name:
        filters["repo_name"] = {"$eq": repo_name}

    try:
        results = vectorstore.similarity_search(question, k=top_k, filter=filters or None)
    except Exception as e:
        print(f"❌ Query error: {e}")
        return []

    if not results:
        print("⚠️ No relevant results found.")
        return []

    for doc in results:
        meta = doc.metadata
        print(f"\n📄 {meta['file_name']} ({meta['repo_name']}) | Chunk {meta['chunk_index']}/{meta['total_chunks']}")
        print("-" * 60)
        print(doc.page_content[:400], "...\n")

    return results


# ---------------------
# 🧪 USAGE
# ---------------------
if __name__ == "__main__":
    zip_path = "/path/to/your/project.zip"
    process_zip_repo(zip_path)

    # Example query
    # query_repo("Explain login logic", repo_name="project-name")
