"""
kanha/rag/vector_store.py
FAISS-based vector store for RAG.
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional

from kanha.utils.logging import get_logger

log = get_logger("vector_store")


class VectorStore:
    """
    Stores document vectors in a FAISS index for fast similarity search.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = None
        self.metadata: List[Dict] = []

        try:
            import faiss
            self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine on normalized vecs)
            self._faiss = faiss
        except ImportError:
            log.warning("FAISS not installed. Run: pip install faiss-cpu")

    @property
    def size(self) -> int:
        if self.index is None:
            return 0
        return self.index.ntotal

    def add(self, vectors: np.ndarray, metadata: List[Dict]):
        """
        Adds vectors and their metadata to the store.

        Args:
            vectors  : (N, dim) float32 array (should be L2-normalized)
            metadata : list of dicts with at least {"text": ..., "source": ...}
        """
        if self.index is None:
            raise RuntimeError("FAISS not available")

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index.add(vectors)
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Searches for the top-k most similar vectors.

        Args:
            query_vector : (1, dim) float32 array
            top_k        : number of results

        Returns:
            List of metadata dicts with added "score" field
        """
        if self.index is None or self.size == 0:
            return []

        query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        top_k = min(top_k, self.size)
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            result = dict(self.metadata[idx])
            result["score"] = float(scores[0][i])
            results.append(result)

        return results

    def save(self, save_dir: str):
        """Saves the index and metadata to disk."""
        os.makedirs(save_dir, exist_ok=True)
        if self.index is not None:
            self._faiss.write_index(self.index, os.path.join(save_dir, "faiss.index"))
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(self.metadata, f)
        log.info(f"VectorStore saved ({self.size} vectors) → {save_dir}")

    def load(self, load_dir: str):
        """Loads the index and metadata from disk."""
        index_path = os.path.join(load_dir, "faiss.index")
        meta_path = os.path.join(load_dir, "metadata.json")

        if os.path.exists(index_path):
            self.index = self._faiss.read_index(index_path)
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)
        log.info(f"VectorStore loaded ({self.size} vectors) from {load_dir}")
