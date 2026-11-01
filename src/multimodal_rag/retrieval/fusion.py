"""Method D: combine the rankings of other retrievers with Reciprocal Rank Fusion.

RRF (Cormack et al., 2009) scores a page by summing 1 / (k + rank) over the
rankings it appears in (rank starts at 1). Only ranks are used, not scores,
so methods whose scores live on different scales (cosine similarity for A,
MaxSim for C) can be combined without normalising them. k = 60 is the value
from the original paper; a larger k flattens the difference between ranks.
"""

import numpy as np

from multimodal_rag.retrieval.base import PageHit, top_k_hits


def reciprocal_rank_fusion(rankings: list[list[str]], k: int) -> dict[str, float]:
    """page_id -> sum of 1 / (k + rank) over the rankings that contain it."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, page_id in enumerate(ranking, start=1):
            scores[page_id] = scores.get(page_id, 0.0) + 1.0 / (k + rank)
    return scores


class FusionRetriever:
    """Fuses already-indexed retrievers; it has no index of its own."""

    def __init__(self, retrievers: dict[str, object], k: int):
        page_ids = [r.page_ids for r in retrievers.values()]
        if any(p != page_ids[0] for p in page_ids):
            raise ValueError("all fused retrievers must index the same pages in the same order")
        self.retrievers = retrievers
        self.k = k
        self.page_ids: list[str] = page_ids[0]

    def search(self, query: str, k: int) -> list[PageHit]:
        # Each retriever ranks every page it can, so no page is cut off before fusion.
        rankings = [[h.page_id for h in r.search(query, k=len(self.page_ids))] for r in self.retrievers.values()]
        fused = reciprocal_rank_fusion(rankings, self.k)
        scores = np.array([fused.get(p, 0.0) for p in self.page_ids])
        # Pages no retriever returned (score 0) are left out, like B2's pages without figures.
        return [h for h in top_k_hits(scores, self.page_ids, k) if h.score > 0]
