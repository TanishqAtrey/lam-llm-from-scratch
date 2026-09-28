"""
scripts/build_index.py
Builds a FAISS index from a directory of .txt documents.

Run:
    python scripts/build_index.py --docs data/raw/ --output data/embeddings/
"""

import os
import argparse
from kanha.rag.chunker import Chunker
from kanha.rag.retriever import Retriever
from kanha.utils.logging import get_logger

log = get_logger("build_index")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs",   required=True, help="Directory of .txt files")
    parser.add_argument("--output", default="data/embeddings/")
    args = parser.parse_args()

    # Load documents
    docs = []
    for fname in os.listdir(args.docs):
        if fname.endswith(".txt"):
            fpath = os.path.join(args.docs, fname)
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                docs.append({"id": fname, "text": f.read()})

    log.info(f"Loaded {len(docs)} documents.")

    # Chunk + embed + index
    chunker   = Chunker()
    chunks    = chunker.chunk_documents(docs)
    log.info(f"Created {len(chunks)} chunks.")

    retriever = Retriever()
    retriever.build_index_from_chunks(chunks, save_dir=args.output)
    log.info(f"Index saved to {args.output}")