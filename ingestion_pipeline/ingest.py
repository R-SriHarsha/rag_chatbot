from loader import load_pdfs
from splitter import split_documents
from embedder import get_embeddings
from vector_store import create_vector_store

PDF_FOLDER = "all_pdfs"

def run_ingestion():
    print("Loading PDFs...")
    docs = load_pdfs(PDF_FOLDER)

    print("Splitting documents...")
    chunks = split_documents(docs)

    print("Creating embeddings...")
    embeddings = get_embeddings()

    print("Storing in vector DB...")
    create_vector_store(chunks, embeddings)

    print("Ingestion completed successfully!")

if __name__ == "__main__":
    run_ingestion()