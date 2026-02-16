from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DB_PATH = "ingestion_pipeline/faiss_index"

def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

    return vectordb.as_retriever(search_kwargs={"k": 3})
