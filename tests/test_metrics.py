"""Tests for retrieval metrics over grouped gold pages.

Gold groups use the evalset.parse_gold shape: 'a|b;c' -> [['a', 'b'], ['c']]
(outer list = AND, inner list = OR). Expected values were worked out by hand;
the reasoning is in each comment.
"""

import pytest

from multimodal_rag.eval.metrics import (
    answerable_at_k,
    answerable_rank,
    mean_reciprocal_rank,
    recall_at_k,
)

# --- answerable_at_k -------------------------------------------------------


def test_single_gold_page_at_rank_1():
    # top-1 = [a]; the only group [a] is hit
    assert answerable_at_k(["a", "x", "y"], [["a"]], k=1) is True


def test_gold_page_outside_top_k_does_not_count():
    # top-2 = [x, y]; a is at rank 3, outside k
    assert answerable_at_k(["x", "y", "a"], [["a"]], k=2) is False


def test_or_group_needs_only_one_page():
    # group [a, b] is hit by b at rank 2
    assert answerable_at_k(["x", "b", "y"], [["a", "b"]], k=3) is True


def test_and_groups_all_need_a_hit():
    # group [a, b] is hit by b, but group [c] has no page in the top 3
    assert answerable_at_k(["x", "b", "y"], [["a", "b"], ["c"]], k=3) is False


def test_and_groups_all_hit():
    # [a, b] hit by b (rank 2), [c] hit by c (rank 4); both within k=4
    assert answerable_at_k(["x", "b", "y", "c"], [["a", "b"], ["c"]], k=4) is True


def test_fewer_results_than_k():
    # only one result, k=10: top-k is just [a], which hits [a]
    assert answerable_at_k(["a"], [["a"]], k=10) is True


def test_k_below_one_is_rejected():
    with pytest.raises(ValueError):
        answerable_at_k(["a"], [["a"]], k=0)


# --- answerable_rank -------------------------------------------------------


def test_rank_is_where_last_group_is_covered():
    # [a, b] covered at rank 2 (b), [c] covered at rank 4 -> all covered at 4
    assert answerable_rank(["x", "b", "y", "c"], [["a", "b"], ["c"]]) == 4


def test_rank_is_none_when_never_answerable():
    assert answerable_rank(["x", "y"], [["a"]]) is None


def test_rank_is_none_when_one_group_is_missing():
    # [a, b] is covered, [c] never appears
    assert answerable_rank(["a", "x", "b"], [["a", "b"], ["c"]]) is None


def test_rank_is_one_based():
    # first result is gold -> rank 1, not 0
    assert answerable_rank(["a", "x"], [["a"]]) == 1


def test_one_page_can_cover_two_groups():
    # page c appears in both groups, so retrieving c alone covers everything
    assert answerable_rank(["x", "c"], [["a", "c"], ["c", "d"]]) == 2


@pytest.mark.parametrize(
    ("retrieved", "groups"),
    [
        (["x", "b", "y", "c"], [["a", "b"], ["c"]]),
        (["a", "x", "b"], [["a", "b"], ["c"]]),
        (["x", "y", "a"], [["a"]]),
    ],
)
def test_rank_and_answerable_at_k_agree(retrieved, groups):
    # answerable within k  <=>  rank exists and rank <= k, for every k
    rank = answerable_rank(retrieved, groups)
    for k in range(1, len(retrieved) + 2):
        expected = rank is not None and rank <= k
        assert answerable_at_k(retrieved, groups, k) is expected


# --- input checks ----------------------------------------------------------


def test_duplicate_retrieved_pages_are_rejected():
    # duplicates would let one page take two slots in the top k;
    # retrievers must collapse chunks to unique pages before scoring
    with pytest.raises(ValueError):
        answerable_at_k(["a", "a", "b"], [["b"]], k=2)
    with pytest.raises(ValueError):
        answerable_rank(["a", "a", "b"], [["b"]])


@pytest.mark.parametrize(
    ("retrieved", "groups"),
    [
        (["toyota/p1"], ["toyota/p1"]),  # flat gold list: strings would be read char by char
        (["a"], "a|b"),  # raw gold string, not parsed with parse_gold
        ("abc", [["c"]]),  # retrieved as a string
        ([1, 2], [["1"]]),  # non-string page ids
        (["a"], [[1]]),
    ],
)
def test_wrong_types_are_rejected(retrieved, groups):
    # these used to give silently wrong scores instead of an error
    with pytest.raises(TypeError):
        answerable_at_k(retrieved, groups, k=1)
    with pytest.raises(TypeError):
        answerable_rank(retrieved, groups)


def test_empty_gold_is_rejected():
    with pytest.raises(ValueError):
        answerable_at_k(["a"], [], k=1)
    with pytest.raises(ValueError):
        answerable_rank(["a"], [["a"], []])


# --- averages over questions -----------------------------------------------


def test_recall_at_k_is_share_of_answerable_questions():
    results = [
        (["a", "x"], [["a"]]),  # answerable at k=1
        (["x", "a"], [["a"]]),  # rank 2 -> not answerable at k=1
        (["b", "c"], [["b"], ["c"]]),  # rank 2 -> not answerable at k=1
        (["c", "x"], [["c"]]),  # answerable at k=1
    ]
    # k=1: 2 of 4 answerable -> 0.5;  k=2: all 4 -> 1.0
    assert recall_at_k(results, k=1) == pytest.approx(0.5)
    assert recall_at_k(results, k=2) == pytest.approx(1.0)


def test_mrr_counts_unanswerable_as_zero():
    results = [
        (["a"], [["a"]]),  # rank 1 -> 1
        (["x", "a"], [["a"]]),  # rank 2 -> 1/2
        (["x", "y"], [["a"]]),  # never  -> 0
    ]
    # (1 + 0.5 + 0) / 3 = 0.5
    assert mean_reciprocal_rank(results) == pytest.approx(0.5)


def test_averages_reject_empty_results():
    with pytest.raises(ValueError):
        recall_at_k([], k=5)
    with pytest.raises(ValueError):
        mean_reciprocal_rank([])
