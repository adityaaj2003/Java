# file: services/file_processing.py

import os
import uuid
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from langchain.document_loaders import DirectoryLoader, NotebookLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain.schema import Document
from utils import clean_and_tokenize


# ------------------------------------------------------
# 1️⃣ Clone GitHub repository
# ------------------------------------------------------
def clone_github_repo(github_url, base_path="./repos"):
    try:
        repo_id = str(uuid.uuid4())
        local_path = os.path.join(base_path, repo_id)
        os.makedirs(local_path, exist_ok=True)

        subprocess.run(['git', 'clone', github_url, local_path], check=True)
        print(f"✅ Repository cloned at {local_path}")
        return repo_id, local_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to clone repository: {e}")
        return None, None


# ------------------------------------------------------
# 2️⃣ Load, split, and store documents (into pgvector)
# ------------------------------------------------------
def load_and_index_files(repo_id, repo_path, postgres_url):
    extensions = [
        'txt', 'md', 'markdown', 'rst', 'py', 'js', 'java', 'c', 'cpp', 'cs', 'go', 'rb',
        'php', 'scala', 'html', 'htm', 'xml', 'json', 'yaml', 'yml', 'ini', 'toml',
        'cfg', 'conf', 'sh', 'bash', 'css', 'scss', 'sql', 'gitignore', 'dockerignore',
        'editorconfig', 'ipynb'
    ]

    file_type_counts = {}
    documents_dict = {}

    for ext in extensions:
        glob_pattern = f'**/*.{ext}'
        try:
            if ext == 'ipynb':
                loader = NotebookLoader(
                    str(repo_path),
                    include_outputs=True,
                    max_output_length=20,
                    remove_newline=True
                )
            else:
                loader = DirectoryLoader(repo_path, glob=glob_pattern)

            loaded_documents = loader.load() if callable(loader.load) else []
            if loaded_documents:
                file_type_counts[ext] = len(loaded_documents)
                for doc in loaded_documents:
                    file_path = doc.metadata['source']
                    relative_path = os.path.relpath(file_path, repo_path)
                    file_id = str(uuid.uuid4())
                    doc.metadata.update({
                        "source": relative_path,
                        "file_id": file_id,
                        "repo_id": repo_id,
                        "repo_path": repo_path,
                        "extension": ext,
                    })
                    documents_dict[file_id] = doc
        except Exception as e:
            print(f"⚠️ Error loading files with pattern '{glob_pattern}': {e}")
            continue

    # --- Split into smaller chunks ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    split_documents = []
    for file_id, original_doc in documents_dict.items():
        chunks = text_splitter.split_documents([original_doc])
        for c in chunks:
            c.metadata.update(original_doc.metadata)
        split_documents.extend(chunks)

    # --- Store in pgvector for semantic search ---
    if split_documents:
        print(f"🧠 Storing {len(split_documents)} chunks for repo {repo_id} in pgvector...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = PGVector.from_documents(
            documents=split_documents,
            embedding=embeddings,
            connection_string=postgres_url,
            collection_name=f"repo_{repo_id}",
            use_jsonb=True
        )

    # --- Create BM25 + TF-IDF hybrid index (in memory) ---
    tokenized_docs = [clean_and_tokenize(doc.page_content) for doc in split_documents]
    index = BM25Okapi(tokenized_docs)

    return {
        "repo_id": repo_id,
        "index": index,
        "documents": split_documents,
        "file_type_counts": file_type_counts,
        "sources": [doc.metadata["source"] for doc in split_documents]
    }


# ------------------------------------------------------
# 3️⃣ Search within a specific repo (hybrid + vector)
# ------------------------------------------------------
def search_documents(query, repo_id, postgres_url, bm25_index, documents, n_results=5):
    query_tokens = clean_and_tokenize(query)
    bm25_scores = bm25_index.get_scores(query_tokens)

    # --- TF-IDF cosine similarity ---
    tfidf_vectorizer = TfidfVectorizer(
        tokenizer=clean_and_tokenize,
        lowercase=True,
        stop_words='english',
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True
    )

    tfidf_matrix = tfidf_vectorizer.fit_transform([doc.page_content for doc in documents])
    query_tfidf = tfidf_vectorizer.transform([query])
    cosine_sim_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

    # --- Combine BM25 + TF-IDF ---
    combined_scores = bm25_scores * 0.5 + cosine_sim_scores * 0.5
    top_indices = combined_scores.argsort()[::-1][:n_results]
    local_results = [documents[i] for i in top_indices]

    # --- Also retrieve from pgvector for deep semantic matches ---
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = PGVector(
        connection_string=postgres_url,
        collection_name=f"repo_{repo_id}",
        embedding_function=embeddings,
        use_jsonb=True
    )
    vector_results = vector_store.similarity_search(query, k=5)

    # --- Combine hybrid + vector results ---
    all_results = {r.metadata["source"]: r for r in local_results + vector_results}
    final = list(all_results.values())[:n_results]

    return final
# file: services/file_processing.py

def main():
    """
    Demonstrates full repository flow:
    1. Clone GitHub repository
    2. Index and store chunks in pgvector
    3. Perform hybrid + semantic search within that repo
    """

    # --- 🧩 Step 1: Config ---
    GITHUB_URL = "https://github.com/tiangolo/fastapi.git"  # you can replace this
    POSTGRES_URL = "postgresql+psycopg://postgres:password@localhost:5432/vector_db"

    print("🚀 Starting repository processing pipeline...\n")

    # --- 🌀 Step 2: Clone the repository ---
    repo_id, repo_path = clone_github_repo(GITHUB_URL)
    if not repo_id:
        print("❌ Repository cloning failed. Exiting.")
        return

    print(f"✅ Repo cloned successfully.\nRepo ID: {repo_id}\nRepo Path: {repo_path}\n")

    # --- 🧠 Step 3: Index and push to pgvector ---
    repo_data = load_and_index_files(
        repo_id=repo_id,
        repo_path=repo_path,
        postgres_url=POSTGRES_URL
    )

    print(f"📊 Indexed {len(repo_data['documents'])} chunks from {len(repo_data['file_type_counts'])} file types.")
    print(f"🧠 Data pushed to pgvector under collection: repo_{repo_id}")

    # --- 🔍 Step 4: Search query ---
    print("\n🔍 Performing sample search...")
    query = "authentication middleware"
    results = search_documents(
        query=query,
        repo_id=repo_id,
        postgres_url=POSTGRES_URL,
        bm25_index=repo_data["index"],
        documents=repo_data["documents"],
        n_results=5
    )

    # --- 📜 Step 5: Display results ---
    print(f"\n💬 Query: {query}")
    print(f"📂 Found {len(results)} relevant code/document sections:\n")

    for i, doc in enumerate(results, start=1):
        meta = doc.metadata
        print(f"🔹 {i}. {meta.get('source', 'unknown file')}")
        print(f"   📄 File ID: {meta.get('file_id')}")
        print(f"   📁 Repo ID: {meta.get('repo_id')}")
        print(f"   ✏️  Snippet:\n{doc.page_content[:300]}...\n")

    print("✅ Search completed successfully.")


# Entry point
if __name__ == "__main__":
    main()
