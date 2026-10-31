"""Method A: embed each page's text layer and rank pages by cosine similarity.

One vector per page (no chunking), so the retrieval unit is the same as in
method C. Pages longer than the model's max_seq_length are truncated by the
encoder; scripts/run_eval.py reports how many.

The encoder is passed in rather than created here, so the ranking logic can be
tested with a small fake encoder and no model download.
"""

from pathlib import Path
from typing import Protocol

import numpy as np

from multimodal_rag.retrieval.base import PageHit, top_k_hits


class TextEncoder(Protocol):
    """The two SentenceTransformer methods this module uses."""

    def encode_document(self, sentences: list[str], **kwargs) -> np.ndarray: ...

    def encode_query(self, sentences: list[str], **kwargs) -> np.ndarray: ...


def rank_by_similarity(page_vectors: np.ndarray, query_vector: np.ndarray, page_ids: list[str], k: int) -> list[PageHit]:
    """Top-k pages by dot product, best first.

    Vectors are unit length, so the dot product is the cosine similarity.
    """
    return top_k_hits(page_vectors @ query_vector, page_ids, k)


def load_encoder(config: dict):
    """Load the SentenceTransformer described by configs/method_a.toml onto the GPU.

    Imported here, not at the top, so tests and other methods don't pay for
    loading torch and transformers.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        config["model"],
        revision=config["revision"],
        device="cuda",  # ROCm builds of torch also use the "cuda" device name
        model_kwargs={"torch_dtype": getattr(torch, config["torch_dtype"])},
    )
    model.max_seq_length = config["max_seq_length"]
    return model


class TextDenseRetriever:
    def __init__(self, encoder: TextEncoder, batch_size: int = 8):
        self.encoder = encoder
        self.batch_size = batch_size
        self.page_ids: list[str] = []
        self.page_vectors: np.ndarray | None = None

    def index(self, pages: list[dict]) -> None:
        self.page_ids = [p["page_id"] for p in pages]
        self.page_vectors = self.encoder.encode_document(
            [p["text"] for p in pages],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)

    def save_index(self, path: Path) -> None:
        np.savez(path, page_ids=np.array(self.page_ids), vectors=self.page_vectors)

    def load_index(self, path: Path) -> None:
        """Reuse vectors saved by save_index instead of re-encoding every page."""
        saved = np.load(path)
        page_ids, vectors = [str(p) for p in saved["page_ids"]], saved["vectors"]
        if len(page_ids) != len(vectors):
            raise ValueError("page_ids and vectors differ in length")
        self.page_ids = page_ids
        self.page_vectors = vectors.astype(np.float32)

    def search(self, query: str, k: int) -> list[PageHit]:
        if self.page_vectors is None:
            raise RuntimeError("call index() or load_index() before search()")
        query_vector = self.encoder.encode_query(
            [query], normalize_embeddings=True, convert_to_numpy=True,
        )[0].astype(np.float32)
        return rank_by_similarity(self.page_vectors, query_vector, self.page_ids, k)
