"""Retrieval metrics over grouped gold pages.

Gold pages come from evalset.parse_gold: 'a|b;c' -> [['a', 'b'], ['c']].
The outer list is AND (every group is needed to answer the question) and the
inner list is OR (any one page of a group is enough). A question is
"answerable within the top k" when every group has at least one page among
the top k retrieved pages.

This differs from the usual Recall@k (share of gold pages retrieved): a
question whose facts sit on two decks only counts once both are found.

Retrieved lists are page ids ranked best-first and must not contain
duplicates. Methods that retrieve chunks (A and B) collapse them to unique
pages before scoring, otherwise one page could fill several top-k slots.
"""

# (retrieved page ids best-first, gold groups) for one question
Result = tuple[list[str], list[list[str]]]


def _is_str_list(value: object) -> bool:
    # A str is iterable too, so check the container type explicitly: otherwise
    # "a|b" or ["p1"] would be compared character by character, silently.
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _check_inputs(retrieved: list[str], gold_groups: list[list[str]]) -> None:
    if not _is_str_list(retrieved):
        raise TypeError("retrieved must be a list of page id strings")
    if not isinstance(gold_groups, list) or not all(_is_str_list(group) for group in gold_groups):
        raise TypeError("gold_groups must be a list of lists of page id strings (use evalset.parse_gold)")
    if not gold_groups or any(not group for group in gold_groups):
        raise ValueError("gold_groups must be non-empty and contain no empty group")
    if len(set(retrieved)) != len(retrieved):
        raise ValueError("retrieved contains duplicate page ids; collapse chunks to pages first")


def answerable_at_k(retrieved: list[str], gold_groups: list[list[str]], k: int) -> bool:
    """True if the top-k results contain at least one page from every gold group.

    >>> answerable_at_k(["x", "b", "c"], [["a", "b"], ["c"]], k=3)
    True
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    _check_inputs(retrieved, gold_groups)
    top_k = set(retrieved[:k])
    return all(any(page in top_k for page in group) for group in gold_groups)


def answerable_rank(retrieved: list[str], gold_groups: list[list[str]]) -> int | None:
    """1-based rank at which every gold group is covered for the first time.

    Returns None if the retrieved list never covers all groups.

    >>> answerable_rank(["x", "b", "y", "c"], [["a", "b"], ["c"]])
    4
    """
    _check_inputs(retrieved, gold_groups)
    uncovered = [set(group) for group in gold_groups]
    for rank, page in enumerate(retrieved, start=1):
        # One page may belong to several groups, so drop every group it covers.
        uncovered = [group for group in uncovered if page not in group]
        if not uncovered:
            return rank
    return None


def recall_at_k(results: list[Result], k: int) -> float:
    """Share of questions that are answerable within the top k."""
    if not results:
        raise ValueError("results is empty")
    answerable = sum(answerable_at_k(retrieved, groups, k) for retrieved, groups in results)
    return answerable / len(results)


def mean_reciprocal_rank(results: list[Result]) -> float:
    """Mean of 1 / answerable_rank over questions; never-answerable questions count as 0."""
    if not results:
        raise ValueError("results is empty")
    total = 0.0
    for retrieved, groups in results:
        rank = answerable_rank(retrieved, groups)
        if rank is not None:
            total += 1 / rank
    return total / len(results)
