import pytest

from multimodal_rag.eval.runner import check_hits
from multimodal_rag.retrieval.base import PageHit
from multimodal_rag.retrieval.fusion import FusionRetriever, reciprocal_rank_fusion


class FixedRetriever:
    """Returns a fixed ranking, whatever the query."""

    def __init__(self, page_ids, ranking):
        self.page_ids = page_ids
        self.ranking = ranking

    def search(self, query, k):
        return [PageHit(p, 1.0 / (i + 1)) for i, p in enumerate(self.ranking[:k])]


def test_rrf_by_hand():
    # k = 60. Ranks: a: (1, 3), b: (2, 1), c: (3, 2)
    # a: 1/61 + 1/63 = 0.016393 + 0.015873 = 0.032266
    # b: 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522  <- best
    # c: 1/63 + 1/62 = 0.015873 + 0.016129 = 0.032002
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]], k=60)
    assert scores == pytest.approx({"a": 1 / 61 + 1 / 63, "b": 1 / 62 + 1 / 61, "c": 1 / 63 + 1 / 62})


def test_page_missing_from_one_ranking_gets_only_the_other_term():
    # B2 never returns pages without figures: "c" is in one list only.
    scores = reciprocal_rank_fusion([["a", "c"], ["a"]], k=60)
    assert scores == pytest.approx({"a": 2 / 61, "c": 1 / 62})


def test_fusion_retriever_ranks_and_breaks_ties_in_page_order():
    pages = ["a", "b", "c", "d"]
    one = FixedRetriever(pages, ["a", "b", "c", "d"])
    two = FixedRetriever(pages, ["b", "a", "d", "c"])
    # a and b tie (1/61 + 1/62 each), as do c and d: ties keep page order.
    fused = FusionRetriever({"one": one, "two": two}, k=60)
    hits = fused.search("q", k=4)
    assert [h.page_id for h in hits] == ["a", "b", "c", "d"]
    assert hits[0].score == pytest.approx(1 / 61 + 1 / 62)
    check_hits(hits, k=4, page_ids=set(pages))
    assert [h.page_id for h in fused.search("q", k=2)] == ["a", "b"]


def test_fusion_retriever_requires_same_pages():
    with pytest.raises(ValueError):
        FusionRetriever({"one": FixedRetriever(["a"], ["a"]), "two": FixedRetriever(["b"], ["b"])}, k=60)
