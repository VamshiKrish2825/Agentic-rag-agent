"""
retriever.py
------------
Document ingestion and retrieval using FAISS + HuggingFace embeddings.

Why FAISS (not Qdrant / Chroma)?
  - Zero external services to run — works fully offline.
  - Perfect for a portfolio project: one pip install and it just works.
  - Easy to swap for Qdrant or Chroma in production by changing ~5 lines.

Ingestion pipeline
------------------
  PDF / TXT files in data/
      ↓  UnstructuredFileLoader / TextLoader
  Raw documents
      ↓  RecursiveCharacterTextSplitter
  Chunks (512 tokens, 64 overlap)
      ↓  HuggingFaceEmbeddings (all-MiniLM-L6-v2)
  FAISS index saved to data/faiss_index/
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.getenv("DATA_DIR", "data/sample_docs"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", "data/faiss_index"))

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
TOP_K = int(os.getenv("TOP_K", "4"))

# ---------------------------------------------------------------------------
# Embedding model (cached after first call)
# ---------------------------------------------------------------------------

_embeddings: HuggingFaceEmbeddings | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embeddings


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_documents(force: bool = False) -> FAISS:
    """
    Load documents from DATA_DIR, chunk them, embed, and build a FAISS index.

    If the index already exists on disk it is loaded directly (skip re-ingestion)
    unless force=True is passed.

    Args:
        force: Re-ingest even if a saved index exists.

    Returns:
        A FAISS vector store ready for similarity search.
    """
    if INDEX_DIR.exists() and not force:
        print(f"[retriever] Loading existing FAISS index from {INDEX_DIR}")
        return FAISS.load_local(
            str(INDEX_DIR),
            _get_embeddings(),
            allow_dangerous_deserialization=True,
        )

    print(f"[retriever] Ingesting documents from {DATA_DIR} …")
    docs = []

    for path in DATA_DIR.iterdir():
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        elif path.suffix.lower() in {".txt", ".md"}:
            loader = TextLoader(str(path), encoding="utf-8")
        else:
            continue  # skip unknown formats
        docs.extend(loader.load())

    if not docs:
        raise FileNotFoundError(
            f"No supported documents found in {DATA_DIR}. "
            "Add .pdf or .txt files there first."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"[retriever] {len(docs)} documents → {len(chunks)} chunks")

    db = FAISS.from_documents(chunks, _get_embeddings())
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    db.save_local(str(INDEX_DIR))
    print(f"[retriever] Index saved to {INDEX_DIR}")
    return db


def get_retriever():
    """
    Return a LangChain retriever interface over the FAISS index.

    Automatically triggers ingestion on first run if no index exists.
    """
    db = ingest_documents()
    return db.as_retriever(search_kwargs={"k": TOP_K})
