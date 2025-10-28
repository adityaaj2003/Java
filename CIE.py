# file: services/repo_analyzer.py

import os
import uuid
from langchain_community.document_loaders import GitLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain.schema import Document


def clone_chunk_store_repo(
    github_url: str,
    postgres_url: str,
    local_base: str = "./repos",
    branch: str = "main",
    collection_name: str | None = None,
):
    """
    Clone a GitHub repo, chunk its files, enrich metadata, and store them in a pgvector DB.

    Args:
        github_url (str): Repository URL.
        postgres_url (str): PostgreSQL connection string with pgvector enabled.
                            Example: "postgresql+psycopg://user:password@localhost:5432/mydb"
        local_base (str): Directory to store cloned repos.
        branch (str): Branch name to clone.
        collection_name (str | None): Optional custom name for the vector collection.

    Returns:
        dict: summary of operation with repo_path and number of chunks stored.
    """

    # ------------------ STEP 1: CLONE REPO ------------------
    os.makedirs(local_base, exist_ok=True)
    repo_id = uuid.uuid4().hex[:8]
    repo_path = os.path.join(local_base, repo_id)

    loader = GitLoader(
        clone_url=github_url,
        repo_path=repo_path,
        branch=branch,
    )
    print(f"🌀 Cloning repo from {github_url} ...")
    raw_docs = loader.load()
    print(f"✅ Repo cloned: {len(raw_docs)} files loaded")

    # ------------------ STEP 2: CHUNK FILES ------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    chunked_docs = splitter.split_documents(raw_docs)

    for doc in chunked_docs:
        source_path = doc.metadata.get("source", "")
        file_ext = os.path.splitext(source_path)[1]
        doc.metadata.update({
            "repo_id": repo_id,
            "repo_url": github_url,
            "repo_path": repo_path,
            "branch": branch,
            "filename": os.path.basename(source_path),
            "file_directory": os.path.dirname(source_path),
            "file_extension": file_ext,
            "chunk_length": len(doc.page_content),
        })

    print(f"🧩 Chunking complete: {len(chunked_docs)} chunks created")

    # ------------------ STEP 3: STORE IN PGVECTOR ------------------
    collection = collection_name or f"repo_{repo_id}"

    print(f"💾 Connecting to PostgreSQL vector store: {collection}")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = PGVector.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        connection_string=postgres_url,
        collection_name=collection,
        use_jsonb=True,  # metadata stored in JSONB
    )
print(f"💾 Storing chunks in batches of {batch_size} ...")
    total = len(chunked_docs)
    for i in range(0, total, batch_size):
        batch = chunked_docs[i:i + batch_size]
        try:
            vector_store.add_documents(batch)
            print(f"✅ Stored batch {i // batch_size + 1} ({i + len(batch)} / {total})")
        except Exception as e:
            print(f"⚠️ Error storing batch {i // batch_size + 1}: {e}")
            time.sleep(2)  # small delay for rate limits

    print(f"🎉 Done! {total} chunks stored in collection '{collection}'")


    print(f"✅ Stored {len(chunked_docs)} chunks in collection '{collection}'")

    # ------------------ STEP 4: RETURN SUMMARY ------------------
    return {
        "repo_id": repo_id,
        "repo_path": repo_path,
        "collection_name": collection,
        "chunks_stored": len(chunked_docs),
    }

if __name__ == "__main__":
    github_url = "https://github.com/openai/openai-cookbook"
    postgres_url = "postgresql+psycopg://postgres:password@localhost:5432/vector_db"

    summary = clone_chunk_store_repo(github_url, postgres_url)
    print("\n--- Summary ---")
    print(summary)
