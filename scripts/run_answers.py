"""Generate answers with Gemini from each retrieval run's top pages, then grade them.
Both models are called through OpenRouter (key: OPENROUTER_API_KEY in .env).

Run with:  uv run python scripts/run_answers.py --config configs/answer.toml --limit 3   (trial, not saved)
           uv run python scripts/run_answers.py --config configs/answer.toml              (all questions)

Conditions: one per retrieval run in [runs] (its top_k pages, in rank order),
plus "gold" (the gold pages). Every API reply is cached in outputs/cache/, so a
rerun costs nothing and gives the same answers. Numeric questions are graded
by rule; text questions by DeepSeek. A full run is saved to
results/answers/<timestamp>/.
"""

import argparse
import base64
import csv
import hashlib
import json
import platform
import tomllib
from collections import Counter
from datetime import datetime
from pathlib import Path

from run_eval import git_state  # scripts/ is on sys.path when this file is run

from multimodal_rag.eval.runner import file_sha256
from multimodal_rag.evalset import gold_page_list, load_questions, parse_gold
from multimodal_rag.generation.answer import answer_cache_path, answer_request_body, parse_answer
from multimodal_rag.generation.api import api_key, call_with_cache, chat
from multimodal_rag.generation.grade import judge_request_body, parse_judgement, units_match, values_match

ROOT = Path(__file__).resolve().parent.parent
PAGES = ROOT / "data" / "pages.jsonl"
QUESTIONS = ROOT / "eval" / "questions.csv"
CACHE = ROOT / "outputs" / "cache"


def retrieved_pages(run_dir: Path, eval_sha: str, k: int) -> dict[str, list[str]]:
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if meta["eval_set"]["sha256"] != eval_sha:
        raise SystemExit(f"{run_dir} was run on a different version of eval/questions.csv")
    if meta["git"]["dirty"]:
        raise SystemExit(f"{run_dir} was run from uncommitted code")
    with (run_dir / "per_question.csv").open(encoding="utf-8") as f:
        return {row["qid"]: row["top10"].split(";")[:k] for row in csv.DictReader(f)}


def generate(config: dict, question: str, page_ids: list[str], images: dict[str, Path], key: str) -> dict:
    path = answer_cache_path(CACHE / "answers", config, question, page_ids)

    def call() -> dict:
        b64 = [base64.b64encode(images[p].read_bytes()).decode() for p in page_ids]
        return chat(answer_request_body(config, question, b64), key)

    record = call_with_cache(path, {"question": question, "pages": page_ids}, call)
    reply = record["response"]
    try:
        parsed, error = parse_answer(reply), None
    except ValueError as e:
        parsed, error = {"answer": "", "value": "", "unit": ""}, str(e)
    served_by = f"{reply.get('provider')}:{reply.get('model')}"
    return parsed | {"error": error, "served_by": served_by, "usage": reply.get("usage", {})}


def judge(config: dict, question: str, reference: str, candidate: str, key: str) -> dict:
    request = judge_request_body(config, question, reference, candidate)
    name = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    record = call_with_cache(CACHE / "judgements" / f"{name}.json", request, lambda: chat(request, key))
    reply = record["response"]
    return parse_judgement(reply) | {"model": f"{reply.get('provider')}:{reply.get('model')}"}


def grade(q: dict, ans: dict, judge_fn) -> dict:
    if ans["error"]:
        return {"correct": False, "grader": "rule", "note": "no parseable answer"}
    if q["answer_value"]:
        value_ok = values_match(ans["value"], q["answer_value"])
        unit_ok = units_match(ans["unit"], q["answer_unit"])
        note = "unit differs" if value_ok and not unit_ok else ""
        return {"correct": value_ok and unit_ok, "grader": "rule", "note": note}
    j = judge_fn(q["question"], q["answer"], ans["answer"])
    return {"correct": j["correct"], "grader": j["model"], "note": j["reason"]}


def accuracy(rows: list[dict]) -> float:
    return sum(r["correct"] for r in rows) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="only the first N questions (trial run, not saved)")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))

    with PAGES.open(encoding="utf-8") as f:
        images = {p["page_id"]: ROOT / p["image"] for p in map(json.loads, f)}
    _, questions = load_questions(QUESTIONS)
    eval_sha = file_sha256(QUESTIONS)
    questions = questions[: args.limit] if args.limit else questions

    conditions = {m: retrieved_pages(ROOT / d, eval_sha, config["top_k"]) for m, d in config["runs"].items()}
    conditions["gold"] = {q["qid"]: gold_page_list(q["gold_pages"]) for q in questions}

    key = api_key()

    def judge_fn(question, reference, candidate):
        return judge(config["grader"], question, reference, candidate, key)

    rows = []
    for cond, pages_by_qid in conditions.items():
        for q in questions:
            pages = pages_by_qid[q["qid"]]
            ans = generate(config["generator"], q["question"], pages, images, key)
            g = grade(q, ans, judge_fn)
            rows.append({"condition": cond, "qid": q["qid"], "evidence": q["evidence"], "q_type": q["q_type"],
                         "author": q["author"], "answerable": all(any(p in pages for p in g) for g in parse_gold(q["gold_pages"])),
                         **{k: ans[k] for k in ("answer", "value", "unit", "error", "served_by")},
                         "usage": ans["usage"], **g})
            print(f"{cond:>4} {q['qid']} {'OK ' if g['correct'] else 'NG '} value={ans['value']!r} unit={ans['unit']!r}"
                  f" gold={q['answer_value']!r}{q['answer_unit']} {g['note'] or ''} {ans['error'] or ''}")

    summary = {c: {"accuracy": accuracy([r for r in rows if r["condition"] == c]),
                   "n": sum(r["condition"] == c for r in rows)} for c in conditions}
    print(json.dumps(summary, indent=2))
    if args.limit:
        print("trial run: not saved")
        return

    def by(column):
        """Accuracy and n per value of column, per condition; empty groups are left out
        (e.g. under "gold" every question is answerable)."""
        out = {}
        for c in conditions:
            groups = {v: [r for r in rows if r["condition"] == c and r[column] == v] for v in {r[column] for r in rows}}
            out[c] = {str(v): {"accuracy": accuracy(g), "n": len(g)} for v, g in sorted(groups.items()) if g}
        return out

    usage = Counter()
    for r in rows:
        usage.update({k: v for k, v in r["usage"].items() if isinstance(v, (int, float))})
    started = datetime.now()
    meta = {
        "config": config,
        "timestamp": started.isoformat(timespec="seconds"),
        "eval_set": {"path": "eval/questions.csv", "sha256": eval_sha, "n": len(questions)},
        "pages": {"path": "data/pages.jsonl", "sha256": file_sha256(PAGES)},
        # provider:model that actually answered, as reported in each reply
        "generator_served_by": sorted({r["served_by"] for r in rows}),
        "grader_models": sorted({r["grader"] for r in rows if r["grader"] != "rule"}),
        "generator_usage_total": dict(usage),
        "n_unparseable": sum(bool(r["error"]) for r in rows),
        "n_unit_differs": sum(r["note"] == "unit differs" for r in rows),
        "git": git_state(),
        "versions": {"python": platform.python_version()},
        "summary": summary,
        "by_evidence": by("evidence"),
        "by_q_type": by("q_type"),
        "by_author": by("author"),
        # accuracy when the given pages do / don't cover every gold group
        "by_answerable": by("answerable"),
    }
    out_dir = ROOT / "results" / "answers" / started.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    fields = ["condition", "qid", "correct", "grader", "note", "answerable", "value", "unit", "answer", "error"]
    with (out_dir / "per_question.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved to {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
