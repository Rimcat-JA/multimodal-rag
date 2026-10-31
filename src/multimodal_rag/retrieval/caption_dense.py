"""Method B2: embed each figure description and rank pages by their best figure.

The descriptions come from ingest/captions.py (a vision LLM reading the whole
page image). Each figure is one retrieval unit, embedded with the same BGE-M3
settings as method A, so A and B2 differ only in the text that is embedded.
A page's score is the highest score among its figures; pages with no figures
are never returned.
"""

from pathlib import Path

import numpy as np

from multimodal_rag.retrieval.base import PageHit, top_k_hits
from multimodal_rag.retrieval.text_dense import TextEncoder


def best_unit_per_page(unit_scores: np.ndarray, unit_pages: np.ndarray, n_pages: int) -> np.ndarray:
    """Max score per page index; -inf for pages without units.

    unit_pages[i] is the index (into the page list) of the page unit i belongs to.
    """
    page_scores = np.full(n_pages, -np.inf)
    np.maximum.at(page_scores, unit_pages, unit_scores)
    return page_scores


class CaptionDenseRetriever:
    def __init__(self, encoder: TextEncoder, batch_size: int = 8):
        self.encoder = encoder
        self.batch_size = batch_size
        self.page_ids: list[str] = []
        self.unit_pages: np.ndarray | None = None  # page index of each unit
        self.unit_vectors: np.ndarray | None = None

    def index(self, pages: list[dict]) -> None:
        """pages: page records with an added "captions" list (one text per figure)."""
        self.page_ids = [p["page_id"] for p in pages]
        units = [(i, text) for i, p in enumerate(pages) for text in p["captions"]]
        self.unit_pages = np.array([i for i, _ in units], dtype=np.int64)
        self.unit_vectors = self.encoder.encode_document(
            [text for _, text in units],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)

    def save_index(self, path: Path) -> None:
        np.savez(path, page_ids=np.array(self.page_ids), unit_pages=self.unit_pages, vectors=self.unit_vectors)

    def load_index(self, path: Path) -> None:
        saved = np.load(path)
        if len(saved["unit_pages"]) != len(saved["vectors"]):
            raise ValueError("unit_pages and vectors differ in length")
        self.page_ids = [str(p) for p in saved["page_ids"]]
        self.unit_pages = saved["unit_pages"]
        self.unit_vectors = saved["vectors"].astype(np.float32)

    def search(self, query: str, k: int) -> list[PageHit]:
        if self.unit_vectors is None:
            raise RuntimeError("call index() or load_index() before search()")
        query_vector = self.encoder.encode_query(
            [query], normalize_embeddings=True, convert_to_numpy=True,
        )[0].astype(np.float32)
        page_scores = best_unit_per_page(self.unit_vectors @ query_vector, self.unit_pages, len(self.page_ids))
        has_units = np.isfinite(page_scores)
        ids = [pid for pid, keep in zip(self.page_ids, has_units) if keep]
        return top_k_hits(page_scores[has_units], ids, k)
