import os
import tempfile
import subprocess
from langchain_community.vectorstores import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# --- CONFIG ---
CONNECTION_STRING = "postgresql+psycopg2://postgres:your_password@localhost:5432/repo_intelligence"
COLLECTION_NAME = "repo_chunks"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key="YOUR_OPENAI_API_KEY")

def clone_repo(github_url):
    """Clone repo to a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    repo_name = github_url.split("/")[-1].replace(".git", "")
    repo_path = os.path.join(temp_dir, repo_name)
    subprocess.run(["git", "clone", "--depth", "1", github_url, repo_path], check=True)
    return repo_name, repo_path


def process_files(repo_name, repo_path):
    """Recursively read all files and chunk them with metadata."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []

    for root, _, files in os.walk(repo_path):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, repo_path)

            # Skip binaries or unnecessary files
            if any(file.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".exe", ".zip", ".gif", ".lock", ".pyc"]):
                continue

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception as e:
                print(f"⚠️ Error reading {rel_path}: {e}")
                continue

            # Chunking
            chunks = splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                metadata = {
                    "repo_name": repo_name,
                    "file_name": rel_path,
                    "chunk_index": i + 1,
                    "total_chunks": len(chunks),
                }
                docs.append({"page_content": chunk, "metadata": metadata})

    print(f"✅ {len(docs)} chunks created for repo '{repo_name}'")
    return docs


def store_in_pgvector(docs):
    """Store all chunks in PostgreSQL using PGVector."""
    vectorstore = PGVector.from_documents(
        documents=docs,
        embedding=embeddings,
        connection_string=CONNECTION_STRING,
        collection_name=COLLECTION_NAME,
    )
    print("✅ Chunks stored successfully in PGVector DB.")
    return vectorstore


def process_repo(github_url):
    repo_name, repo_path = clone_repo(github_url)
    docs = process_files(repo_name, repo_path)
    store_in_pgvector(docs)


# --- Example Run ---
if __name__ == "__main__":
    github_url = "https://github.com/psf/requests.git"
    process_repo(github_url)
