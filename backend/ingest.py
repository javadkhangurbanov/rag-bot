import os

from dotenv import load_dotenv

from .rag_store import ingest_folder

if __name__ == "__main__":
    load_dotenv()
    folder = os.path.join(os.path.dirname(__file__), "..", "knowledge")
    folder = os.path.abspath(folder)
    count = ingest_folder(folder)
    print(f"Ingested {count} chunks from {folder}")
