"""
ingest.py
---------
One-time script to build (or rebuild) the FAISS vector index.

Usage:
    python ingest.py              # build index from data/sample_docs/
    python ingest.py --force      # force rebuild even if index exists
    python ingest.py --data-dir /path/to/my/docs

Run this once before starting the app.  After that, the app loads the
saved index from disk — no re-embedding needed.
"""

import argparse
import os
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from src.retriever import ingest_documents


def main():
    parser = argparse.ArgumentParser(description="Build the FAISS vector index.")
    parser.add_argument(
        "--data-dir",
        default="data/sample_docs",
        help="Directory containing .pdf / .txt / .md files (default: data/sample_docs)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the index even if it already exists.",
    )
    args = parser.parse_args()

    os.environ["DATA_DIR"] = args.data_dir

    print("=" * 60)
    print("  Agentic RAG — Document Ingestion")
    print("=" * 60)
    db = ingest_documents(force=args.force)
    count = db.index.ntotal  # FAISS index size
    print(f"\n✅  Done!  {count} vectors stored in the FAISS index.")
    print("   Start the app with:  python app.py")


if __name__ == "__main__":
    main()
