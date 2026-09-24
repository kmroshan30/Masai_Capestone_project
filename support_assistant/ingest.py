"""Ingestion stage: load -> chunk -> embed -> store in ChromaDB.

- Chunks: one chunk per document (docs are short policy paragraphs), ids
  doc_01 ... doc_08.
- Embeddings: sentence-transformers all-MiniLM-L6-v2 (local, free, keyless).
- Store: persistent ChromaDB collection `zepto_policies` with cosine space.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = Path(__file__).parent / "docs"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "zepto_policies"
MODEL_NAME = "all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None


def load_and_chunk() -> list[dict]:
    """Load the 8 policy documents and chunk them (1 chunk per document)."""
    chunks: list[dict] = []
    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        # simple per-document chunking: each short policy doc is one chunk
        chunks.append(
            {
                "id": path.stem,  # e.g. doc_01
                "text": text,
                "metadata": {"source": path.name},
            }
        )
    if len(chunks) != 8:
        raise RuntimeError(f"expected 8 documents, found {len(chunks)}")
    return chunks


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def build_collection(force: bool = False) -> chromadb.Collection:
    """Embed all chunks and upsert them into ChromaDB (idempotent)."""
    col = get_collection()
    if force:
        try:
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            client.delete_collection(COLLECTION_NAME)
            col = get_collection()
        except Exception:
            pass
    if col.count() >= 8 and not force:
        return col

    chunks = load_and_chunk()
    model = get_model()
    embeddings = model.encode(
        [c["text"] for c in chunks],
        convert_to_numpy=True,
        show_progress_bar=False,
    ).tolist()
    col.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        embeddings=embeddings,
    )
    print(f"[ingest] indexed {col.count()} chunks into '{COLLECTION_NAME}'")
    return col


def retrieve(query: str, k: int = 3) -> list[dict]:
    """Embed the query and retrieve top-k chunks by cosine similarity.

    Runs for real in BOTH mock and real-LLM modes (no API key needed).
    """
    col = build_collection()
    model = get_model()
    q_emb = model.encode([query], convert_to_numpy=True)[0].tolist()
    res = col.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    out: list[dict] = []
    for i in range(len(res["ids"][0])):
        distance = res["distances"][0][i]
        out.append(
            {
                "id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                # cosine space: distance in [0, 2]; similarity = 1 - distance
                "score": 1.0 - float(distance),
            }
        )
    return out


if __name__ == "__main__":
    build_collection(force=True)
    print("[ingest] done -", [c["id"] for c in load_and_chunk()])
