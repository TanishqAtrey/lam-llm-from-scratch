"""
kanha/rag/retriever.py
RAG retriever — embeds queries and retrieves relevant chunks.
"""

import os
import numpy as np
from typing import List, Dict, Optional

from kanha.rag.vector_store import VectorStore
from kanha.utils.config import cfg
from kanha.utils.logging import get_logger

log = get_logger("retriever")


class SentenceEmbedder:
    """
    Wraps sentence-transformers for generating embeddings.
    Falls back gracefully if not installed.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or cfg.rag.embedding_model
        self.model = None
        self.dim = 384  # default for all-MiniLM-L6-v2

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.dim = self.model.get_sentence_embedding_dimension()
            log.info(f"Embedder loaded: {self.model_name} (dim={self.dim})")
        except ImportError:
            log.warning("sentence-transformers not installed. Run: pip install sentence-transformers")

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embeds a list of texts. Returns (N, dim) float32 array."""
        if self.model is None:
            raise RuntimeError("sentence-transformers not installed")
        embeddings = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embeds a single query. Returns (1, dim) float32 array."""
        return self.embed([text])


class Retriever:
    """
    RAG retriever — embeds queries, searches the vector store,
    and returns relevant text chunks.
    """

    def __init__(self, embedder=None, store=None):
        self.embedder = embedder or SentenceEmbedder()
        self.store = store or VectorStore(dim=self.embedder.dim)

    def build_index_from_chunks(self, chunks: List[Dict], save_dir: str = None):
        """
        Embeds and indexes a list of text chunks.

        Args:
            chunks   : list of {"id": ..., "text": ..., "source": ...}
            save_dir : if provided, saves the index to disk
        """
        texts = [c["text"] for c in chunks]
        log.info(f"Embedding {len(texts)} chunks...")
        vectors = self.embedder.embed(texts)

        self.store.add(vectors, chunks)
        log.info(f"Index built with {self.store.size} vectors")

        if save_dir:
            self.store.save(save_dir)

    def load_index(self, load_dir: str):
        """Loads a pre-built index from disk."""
        self.store.load(load_dir)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieves the top-k most relevant chunks for a query.

        Args:
            query : the search query string
            top_k : number of results

        Returns:
            List of chunk dicts with "score" field added
        """
        query_vec = self.embedder.embed_query(query)
        results = self.store.search(query_vec, top_k=top_k)
        return results
