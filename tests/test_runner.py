import json

import pytest

from multimodal_rag.eval.runner import check_hits, file_sha256, run_retrieval, save_run, summarize, summarize_by
from multimodal_rag.retrieval.base import PageHit

PAGES = {"a", "b", "c", "d", "e"}


class FakeRetriever:
    """Returns a fixed ranking per question, so expected scores can be worked out by hand."""

    def __init__(self, rankings: dict[str, list[str]]):
        self.rankings = rankings

    def index(self, pages: list[dict]) -> None:
        pass

    def search(self, query: str, k: int) -> list[PageHit]:
        ranked = self.rankings[query][:k]
        # Scores fall with rank, as a real retriever's would.
        return [PageHit(p, 1.0 - i / 10) for i, p in enumerate(ranked)]


QUESTIONS = [
    {"qid": "q001", "question": "Q1", "gold_pages": "a", "author": "human", "evidence": "chart"},
    {"qid": "q002", "question": "Q2", "gold_pages": "a|b;c", "author": "claude", "evidence": "table"},
    {"qid": "q003", "question": "Q3", "gold_pages": "e", "author": "claude", "evidence": "chart"},
]
RANKINGS = {
    "Q1": ["a", "b", "c", "d", "e"],  # gold "a" at rank 1                      -> rank 1
    "Q2": ["d", "b", "e", "c", "a"],  # group [a|b] covered at 2, [c] at 4      -> rank 4
    "Q3": ["a", "b", "c", "d"],       # "e" never returned                      -> None
}


def run():
    return run_retrieval(FakeRetriever(RANKINGS), QUESTIONS, PAGES, k=5)


def test_run_records_rank_per_question():
    results = run()
    assert [r.qid for r in results] == ["q001", "q002", "q003"]
    assert [r.rank for r in results] == [1, 4, None]
    assert results[1].gold_groups == [["a", "b"], ["c"]]
    assert results[1].retrieved == ["d", "b", "e", "c", "a"]


def test_summarize_by_hand():
    s = summarize(run())
    # answerable within top k:   q001  q002  q003
    #   k=1                       yes   no    no   -> 1/3
    #   k=3                       yes   no    no   -> 1/3  (q002 needs rank 4)
    #   k=5                       yes   yes   no   -> 2/3
    # MRR = (1/1 + 1/4 + 0) / 3 = 1.25 / 3
    assert s["n"] == 3
    assert s["recall@1"] == pytest.approx(1 / 3)
    assert s["recall@3"] == pytest.approx(1 / 3)
    assert s["recall@5"] == pytest.approx(2 / 3)
    assert s["mrr"] == pytest.approx(1.25 / 3)


def test_summarize_by_author():
    by = summarize_by(run(), QUESTIONS, "author")
    assert list(by) == ["claude", "human"]
    # human: q001 only, rank 1. claude: q002 (rank 4) and q003 (never).
    assert by["human"]["n"] == 1 and by["human"]["mrr"] == pytest.approx(1.0)
    assert by["claude"]["n"] == 2
    assert by["claude"]["recall@5"] == pytest.approx(1 / 2)
    assert by["claude"]["mrr"] == pytest.approx((1 / 4 + 0) / 2)


def test_k_limits_search_depth():
    # With k=3, q002's "c" (rank 4) is cut off, so q002 is never answerable.
    results = run_retrieval(FakeRetriever(RANKINGS), QUESTIONS, PAGES, k=3)
    assert [r.rank for r in results] == [1, None, None]


@pytest.mark.parametrize(
    "hits, message",
    [
        ([PageHit("a", 1.0), PageHit("zzz", 0.5)], "not in data/pages.jsonl"),
        ([PageHit("a", 1.0), PageHit("a", 0.5)], "same page twice"),
        ([PageHit("a", 0.5), PageHit("b", 1.0)], "not sorted"),
        ([PageHit("a", 1.0), PageHit("b", 0.9), PageHit("c", 0.8)], "for k=2"),
    ],
)
def test_malformed_hits_are_rejected(hits, message):
    with pytest.raises(ValueError, match=message):
        check_hits(hits, k=2 if message == "for k=2" else 5, page_ids=PAGES)


def test_equal_scores_are_allowed():
    check_hits([PageHit("a", 0.5), PageHit("b", 0.5)], k=5, page_ids=PAGES)


def test_file_sha256_changes_with_content(tmp_path):
    f = tmp_path / "questions.csv"
    f.write_bytes(b"qid\nq001\n")
    before = file_sha256(f)
    f.write_bytes(b"qid\nq002\n")
    assert file_sha256(f) != before
    assert len(before) == 64


def test_save_run_writes_scores_and_refuses_overwrite(tmp_path):
    out = tmp_path / "run1"
    returned = save_run(out, {"method": "fake"}, run(), QUESTIONS)
    saved = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert saved == returned
    assert saved["method"] == "fake"
    assert saved["summary"]["mrr"] == pytest.approx(1.25 / 3)
    assert set(saved["by_author"]) == {"human", "claude"}
    lines = (out / "per_question.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "qid,rank,top10"
    assert lines[2] == "q002,4,d;b;e;c;a"
    assert lines[3] == "q003,,a;b;c;d"  # never answerable -> empty rank
    with pytest.raises(FileExistsError):
        save_run(out, {"method": "fake"}, run(), QUESTIONS)
