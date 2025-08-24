import os
import glob
from typing import Iterable, List, Dict, Any

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

from .embeddings import embed_texts

load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma")
COLLECTION_NAME = "kb_main"

def _get_client():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    print(">> CHROMA_DIR:", os.path.abspath(CHROMA_DIR))
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    return client

def get_collection():
    client = _get_client()
    try:
        # Prefer get_or_create to avoid race/first-run errors
        col = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception:
        # Fallback, though get_or_create should handle it
        col = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return col

def simple_chunk(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """
    Robust fixed-size chunker.
    - If the text is short, return a single chunk.
    - Uses a positive 'step' so we never get stuck when len(tokens) <= overlap.
    """
    tokens = text.split()
    if not tokens:
        return []
    if len(tokens) <= chunk_size:
        return [" ".join(tokens)]

    step = max(1, chunk_size - overlap)  # ensure forward progress
    chunks: list[str] = []
    start = 0
    n = len(tokens)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(" ".join(tokens[start:end]))
        if end >= n:
            break
        start += step
    return chunks

def load_files_from_folder(folder: str) -> List[Dict[str, Any]]:
    """
    Reads .txt/.md files and returns [{"doc_id":..., "text":..., "metadata":{...}}, ...]
    """
    out = []
    for path in glob.glob(os.path.join(folder, "**/*"), recursive=True):
        if os.path.isdir(path):
            continue
        if not (path.endswith(".txt") or path.endswith(".md")):
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        chunks = simple_chunk(raw)
        for i, ch in enumerate(chunks):
            out.append({
                "doc_id": f"{os.path.basename(path)}::{i}",
                "text": ch,
                "metadata": {"source": os.path.basename(path), "chunk": i, "path": path},
            })
    return out

def ingest_folder(folder: str) -> int:
    """
    Chunks files, embeds, and upserts into Chroma.
    Returns number of chunks ingested.
    """
    docs = load_files_from_folder(folder)
    if not docs:
        return 0

    col = get_collection()
    ids = [d["doc_id"] for d in docs]
    texts = [d["text"] for d in docs]
    metas = [d["metadata"] for d in docs]

    vectors = embed_texts(texts)
    # Use upsert (add with existing ids will throw)
    col.upsert(ids=ids, documents=texts, embeddings=vectors, metadatas=metas)
    return len(docs)

def retrieve(query: str, k: int = 4) -> list[dict]:
    col = get_collection()
    qv = embed_texts([query])[0]
    results = col.query(
        query_embeddings=[qv],
        n_results=k,
        include=["documents", "metadatas", "distances"],  # <- remove "ids"
    )
    out = []
    if results and results.get("documents"):
        # ids are still returned in results["ids"] even if not listed in include
        ids = results.get("ids", [[]])[0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        for i in range(len(docs)):
            out.append({
                "id": ids[i] if i < len(ids) else None,
                "text": docs[i],
                "metadata": metas[i],
                "distance": dists[i],
            })
    return out


def format_context(snippets: List[Dict[str, Any]]) -> str:
    """
    Builds a compact context block for the LLM.
    """
    lines = []
    for s in snippets:
        src = s["metadata"].get("source")
        lines.append(f"[{src}] {s['text']}")
    return "\n\n".join(lines)
