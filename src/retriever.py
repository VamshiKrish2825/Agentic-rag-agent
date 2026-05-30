"""
retriever.py
------------
Document ingestion and retrieval using FAISS + HuggingFace embeddings.

Why FAISS (not Qdrant / Chroma)?
  - Zero external services to run - works fully offline.
  - Perfect for a portfolio project: one pip install and it just works.
  - Easy to swap for Qdrant or Chroma in production by changing ~5 lines.

Ingestion pipeline
------------------
  PDF / TXT files in data/
      -> PyPDFLoader / TextLoader
  Raw documents
      -> RecursiveCharacterTextSplitter
  Chunks (512 tokens, 64 overlap)
      -> HuggingFaceEmbeddings (all-MiniLM-L6-v2)
  FAISS index saved to data/faiss_index/
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _data_dir() -> Path:
    return Path(os.getenv("DATA_DIR", "data/sample_docs"))


def _index_dir() -> Path:
    return Path(os.getenv("INDEX_DIR", "data/faiss_index"))


def _embed_model() -> str:
    return os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def _chunk_size() -> int:
    return int(os.getenv("CHUNK_SIZE", "512"))


def _chunk_overlap() -> int:
    return int(os.getenv("CHUNK_OVERLAP", "64"))


def _top_k() -> int:
    return int(os.getenv("TOP_K", "4"))

# ---------------------------------------------------------------------------
# Embedding model (cached after first call)
# ---------------------------------------------------------------------------

_embeddings: HuggingFaceEmbeddings | None = None
_embeddings_model_name: str | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings, _embeddings_model_name
    model_name = _embed_model()
    if _embeddings is None or _embeddings_model_name != model_name:
        _embeddings = HuggingFaceEmbeddings(model_name=model_name)
        _embeddings_model_name = model_name
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
    data_dir = _data_dir()
    index_dir = _index_dir()

    if index_dir.exists() and not force:
        print(f"[retriever] Loading existing FAISS index from {index_dir}")
        return FAISS.load_local(
            str(index_dir),
            _get_embeddings(),
            allow_dangerous_deserialization=True,
        )

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Document directory not found: {data_dir}. "
            "Create it and add .pdf, .txt, or .md files."
        )

    print(f"[retriever] Ingesting documents from {data_dir} ...")
    docs = []

    for path in sorted(data_dir.iterdir()):
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        elif path.suffix.lower() in {".txt", ".md"}:
            loader = TextLoader(str(path), encoding="utf-8")
        else:
            continue  # skip unknown formats
        docs.extend(loader.load())

    if not docs:
        raise FileNotFoundError(
            f"No supported documents found in {data_dir}. "
            "Add .pdf, .txt, or .md files there first."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_chunk_size(),
        chunk_overlap=_chunk_overlap(),
    )
    chunks = splitter.split_documents(docs)
    print(f"[retriever] {len(docs)} documents -> {len(chunks)} chunks")

    db = FAISS.from_documents(chunks, _get_embeddings())
    index_dir.mkdir(parents=True, exist_ok=True)
    db.save_local(str(index_dir))
    print(f"[retriever] Index saved to {index_dir}")
    return db


def get_retriever():
    """
    Return a LangChain retriever interface over the FAISS index.

    Automatically triggers ingestion on first run if no index exists.
    """
    db = ingest_documents()
    return db.as_retriever(search_kwargs={"k": _top_k()})
