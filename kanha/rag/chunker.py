"""
kanha/rag/chunker.py
Text chunker for RAG pipeline.

Splits documents into overlapping chunks of a specified word count.
"""

from typing import List, Dict
from kanha.utils.config import cfg


class Chunker:
    """
    Splits documents into fixed-size word chunks with overlap.

    Args:
        chunk_size    : number of words per chunk
        chunk_overlap : number of overlapping words between consecutive chunks
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or cfg.rag.chunk_size
        self.chunk_overlap = chunk_overlap or cfg.rag.chunk_overlap

    def chunk_document(self, doc_id: str, text: str) -> List[Dict]:
        """
        Splits a single document into chunks.

        Returns:
            List of {"id": ..., "text": ..., "source": ...} dicts
        """
        words = text.split()
        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        idx = 0

        for start in range(0, len(words), step):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            if chunk_text.strip():
                chunks.append({
                    "id": f"{doc_id}_chunk{idx}",
                    "text": chunk_text,
                    "source": doc_id,
                })
                idx += 1
            if end >= len(words):
                break

        return chunks

    def chunk_documents(self, docs: List[Dict]) -> List[Dict]:
        """
        Chunks multiple documents.

        Args:
            docs : list of {"id": ..., "text": ...} dicts

        Returns:
            Flat list of all chunks from all documents
        """
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc["id"], doc["text"]))
        return all_chunks
