"""Training data for fine-tuning ColQwen2.5: company split and hard negatives.

Method C's misses on the evaluation set were almost all within the right
company: the wrong deck, or the wrong one of several similar pages
(results/failure_analysis.md). So each training query gets one hard negative:
the page of the same company that the base model ranks highest after the
positive page.

A page that scores almost as high as the positive may show the same fact (the
same figure is often repeated on a summary page). Training it as a negative
would teach the model the wrong thing, so pages scoring at least max_ratio of
the positive's score are skipped.
"""


def company_of(page_id: str) -> str:
    return page_id.split("/")[0]


def pick_hard_negative(ranked: list[tuple[str, float]], positive: str, max_ratio: float) -> str | None:
    """The highest-ranked page of the positive's company, other than the positive, scoring
    below max_ratio * the positive's score. `ranked` is (page_id, score), best first."""
    scores = dict(ranked)
    if positive not in scores:
        raise ValueError(f"{positive} is not in the ranking")
    limit = max_ratio * scores[positive]
    for page_id, score in ranked:
        if page_id != positive and company_of(page_id) == company_of(positive) and score < limit:
            return page_id
    return None


def drop_ambiguous_queries(rows: list[dict]) -> list[dict]:
    """Rows whose query text was also generated for another page are dropped.

    A summary page often repeats a figure from a detail page, so the generator
    can write the same question for both. Training would then call one of them
    the answer and the other a negative (in-batch or hard), which is wrong.
    """
    pages_by_query: dict[str, set[str]] = {}
    for r in rows:
        pages_by_query.setdefault(r["query"], set()).add(r["page_id"])
    return [r for r in rows if len(pages_by_query[r["query"]]) == 1]


def split_by_company(rows: list[dict], val_companies: set[str]) -> tuple[list[dict], list[dict]]:
    """Rows of validation companies go to the second list; no company is on both sides."""
    train = [r for r in rows if company_of(r["page_id"]) not in val_companies]
    val = [r for r in rows if company_of(r["page_id"]) in val_companies]
    return train, val


def overlapping_companies(train_sources: list[dict], eval_sources: list[dict]) -> set[str]:
    """Companies that appear in both source lists, matched by id or by name.

    Both are checked because ids are chosen by hand (mitsubishi_electric vs
    mitsubishielectric) and names are spelled differently (株式会社, spaces).
    Training on an evaluation company would leak its deck layouts into the model.
    """
    def name(row: dict) -> str:
        return row["company"].replace("株式会社", "").replace(" ", "").replace("　", "").strip()

    eval_ids = {r["company_id"] for r in eval_sources}
    eval_names = {name(r) for r in eval_sources}
    # "日立" and "日立製作所" are the same company: a name contained in the other counts as a match
    return {r["company"] for r in train_sources
            if r["company_id"] in eval_ids or any(name(r) in e or e in name(r) for e in eval_names)}
