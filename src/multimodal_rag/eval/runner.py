"""Run a retriever over the evaluation set and score it.

This module does not depend on any retrieval method: it calls
Retriever.search, checks that the output is well-formed, and hands the ranked
page ids to metrics.py. Malformed output (unknown page ids, duplicates, scores
out of order) raises instead of being scored, because a silently wrong ranking
would make the method comparison meaningless.
"""

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from multimodal_rag.eval.metrics import answerable_rank, mean_reciprocal_rank, recall_at_k
from multimodal_rag.evalset import parse_gold
from multimodal_rag.retrieval.base import PageHit, Retriever

REPORTED_KS = (1, 3, 5)
# How many retrieved pages per question to keep in per_question.csv.
SAVED_TOP = 10


@dataclass
class QuestionResult:
    qid: str
    retrieved: list[str]  # page ids, best first
    gold_groups: list[list[str]]
    rank: int | None  # from metrics.answerable_rank; None = never answerable


def file_sha256(path: Path) -> str:
    """Fingerprint of a file, saved with each run so results from different
    versions of the evaluation set are never compared by mistake."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hits(hits: list[PageHit], k: int, page_ids: set[str]) -> None:
    if len(hits) > k:
        raise ValueError(f"search returned {len(hits)} hits for k={k}")
    ids = [h.page_id for h in hits]
    unknown = [p for p in ids if p not in page_ids]
    if unknown:
        raise ValueError(f"search returned page ids not in data/pages.jsonl: {unknown[:5]}")
    if len(set(ids)) != len(ids):
        raise ValueError("search returned the same page twice")
    scores = [h.score for h in hits]
    if any(a < b for a, b in zip(scores, scores[1:])):
        raise ValueError("search results are not sorted best first")


def run_retrieval(retriever: Retriever, questions: list[dict], page_ids: set[str], k: int) -> list[QuestionResult]:
    """Search every question and record where its gold pages were found.

    `k` is how deep each search goes. This project ranks every page
    (k = number of pages), so MRR sees the full ranking.
    """
    results = []
    for row in questions:
        hits = retriever.search(row["question"], k)
        check_hits(hits, k, page_ids)
        retrieved = [h.page_id for h in hits]
        groups = parse_gold(row["gold_pages"])
        results.append(QuestionResult(row["qid"], retrieved, groups, answerable_rank(retrieved, groups)))
    return results


def summarize(results: list[QuestionResult], ks: tuple[int, ...] = REPORTED_KS) -> dict:
    pairs = [(r.retrieved, r.gold_groups) for r in results]
    summary: dict = {"n": len(results)}
    for k in ks:
        summary[f"recall@{k}"] = recall_at_k(pairs, k)
    summary["mrr"] = mean_reciprocal_rank(pairs)
    return summary


def summarize_by(results: list[QuestionResult], questions: list[dict], column: str,
                 ks: tuple[int, ...] = REPORTED_KS) -> dict[str, dict]:
    """summarize() per value of a question column, e.g. author or evidence."""
    value_of = {row["qid"]: row[column] for row in questions}
    groups: dict[str, list[QuestionResult]] = {}
    for r in results:
        groups.setdefault(value_of[r.qid], []).append(r)
    return {value: summarize(rs, ks) for value, rs in sorted(groups.items())}


def save_run(out_dir: Path, meta: dict, results: list[QuestionResult], questions: list[dict]) -> dict:
    """Write run.json (settings + scores) and per_question.csv to a new directory.

    `meta` holds everything needed to reproduce the run: config, model ids,
    hashes of the evaluation set and pages, git commit. Refuses to overwrite an
    existing run so earlier results are never lost.
    """
    out_dir.mkdir(parents=True, exist_ok=False)
    run = meta | {
        "summary": summarize(results),
        "by_author": summarize_by(results, questions, "author"),
        "by_evidence": summarize_by(results, questions, "evidence"),
    }
    with (out_dir / "run.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with (out_dir / "per_question.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["qid", "rank", f"top{SAVED_TOP}"])
        for r in results:
            writer.writerow([r.qid, "" if r.rank is None else r.rank, ";".join(r.retrieved[:SAVED_TOP])])
    return run
