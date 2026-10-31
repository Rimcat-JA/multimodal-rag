import numpy as np
import pytest

from multimodal_rag.eval.runner import check_hits
from multimodal_rag.retrieval.text_dense import TextDenseRetriever, rank_by_similarity

# Three unit vectors in 2D, so similarities are easy to compute by hand.
S = np.sqrt(0.5)
VECTORS = np.array([[1.0, 0.0], [0.0, 1.0], [S, S]], dtype=np.float32)
IDS = ["p1", "p2", "p3"]


def test_rank_by_similarity_by_hand():
    # query (1, 0): p1 -> 1.0, p2 -> 0.0, p3 -> 0.707
    hits = rank_by_similarity(VECTORS, np.array([1.0, 0.0], dtype=np.float32), IDS, k=3)
    assert [h.page_id for h in hits] == ["p1", "p3", "p2"]
    assert [h.score for h in hits] == pytest.approx([1.0, S, 0.0])


def test_rank_by_similarity_cuts_at_k():
    hits = rank_by_similarity(VECTORS, np.array([0.0, 1.0], dtype=np.float32), IDS, k=1)
    assert [h.page_id for h in hits] == ["p2"]


def test_ties_keep_page_order():
    # query (1, 1)/sqrt(2) scores p1 and p2 equally (0.707); p3 scores 1.0.
    hits = rank_by_similarity(VECTORS, np.array([S, S], dtype=np.float32), IDS, k=3)
    assert [h.page_id for h in hits] == ["p3", "p1", "p2"]


class FakeEncoder:
    """Maps each text to a fixed vector and records the arguments it was called with."""

    def __init__(self, table: dict[str, list[float]]):
        self.table = table
        self.calls: list[dict] = []

    def _encode(self, sentences, **kwargs):
        self.calls.append(kwargs)
        return np.array([self.table[s] for s in sentences])

    encode_document = _encode
    encode_query = _encode


def test_retriever_end_to_end_with_fake_encoder():
    encoder = FakeEncoder({"text1": [1, 0], "text2": [0, 1], "query": [0, 1]})
    retriever = TextDenseRetriever(encoder)
    retriever.index([{"page_id": "a", "text": "text1"}, {"page_id": "b", "text": "text2"}])
    hits = retriever.search("query", k=2)
    assert [h.page_id for h in hits] == ["b", "a"]
    check_hits(hits, k=2, page_ids={"a", "b"})
    # Both sides must ask for unit vectors, or the dot product is not the cosine.
    assert all(call["normalize_embeddings"] is True for call in encoder.calls)


def test_search_before_index_fails():
    with pytest.raises(RuntimeError):
        TextDenseRetriever(FakeEncoder({})).search("q", k=1)


def test_load_index_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        TextDenseRetriever(FakeEncoder({})).load_index(["a", "b"], VECTORS)
