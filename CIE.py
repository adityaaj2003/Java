import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.runnables import RunnablePassthrough


def see_repository_intelligence(
    repo_id: str,
    postgres_url: str,
    collection_name: str,
    query: str = "Analyze the repository and detect its type, entry points, dependencies, and framework.",
):
    """
    Retrieve stored repo chunks from pgvector, analyze using LLM + web search,
    and return structured repository intelligence.

    Args:
        repo_id (str): ID of the repository stored.
        postgres_url (str): PostgreSQL + pgvector connection string.
        collection_name (str): Vector collection name for this repo.
        query (str): Query for the analysis (default: full repo understanding).

    Returns:
        dict: Structured analysis summary.
    """

    # --- 1️⃣ Connect to vector store ---
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = PGVector(
        connection_string=postgres_url,
        collection_name=collection_name,
        embedding_function=embeddings,
        use_jsonb=True,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 8})

    # --- 2️⃣ Prepare web search + LLM ---
    web_search = DuckDuckGoSearchResults()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    # --- 3️⃣ Get context + web info ---
    retrieved_docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    web_results = web_search.run(f"{query} repository type and framework details")

    # --- 4️⃣ Modern LCEL-style prompt ---
    prompt = ChatPromptTemplate.from_template(
        """You are a repository intelligence system.
        Using the following repository context and web info, analyze the codebase.

        Context:
        {context}

        Web Info:
        {web}

        Task:
        {query}

        Return your answer in JSON with keys:
        repo_type, structure, important_files, entry_point, dependencies, framework, summary.
        """
    )

    # --- 5️⃣ Build LCEL chain (retriever + prompt + LLM) ---
    chain = (
        {
            "context": lambda _: context,  # static repo context
            "query": RunnablePassthrough(),
            "web": lambda _: web_results,  # static web info
        }
        | prompt
        | llm
    )

    # --- 6️⃣ Run the chain ---
    response = chain.invoke(query)

    # --- 7️⃣ Parse JSON output ---
    try:
        structured_output = json.loads(response.content)
    except Exception:
        structured_output = {"summary": response.content}

    structured_output["repo_id"] = repo_id
    structured_output["collection_name"] = collection_name

    return structured_output


if __name__ == "__main__":
    postgres_url = "postgresql+psycopg://postgres:password@localhost:5432/vector_db"

    result = see_repository_intelligence(
        repo_id="a1b2c3d4",
        postgres_url=postgres_url,
        collection_name="repo_a1b2c3d4"
    )

    print(json.dumps(result, indent=2))

