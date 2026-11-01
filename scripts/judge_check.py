"""Check DeepSeek's grading of text answers against the developer's own judgement.

Run with:  uv run python scripts/judge_check.py export   writes eval/judge_check_<run>.csv (never overwrites)
           uv run python scripts/judge_check.py agree    reports agreement once human_correct is filled in

eval/README.md asks for part of the LLM grading to be checked by eye. The export
lists every text question graded by DeepSeek in the latest clean answer run.
A row identical (question, reference, answer and DeepSeek's verdict) to one the
developer already labelled in an earlier eval/judge_check_*.csv keeps that
label, so only new judgements need checking. Fill in human_correct with true or
false (leave it empty to skip a row); the agreement is printed and written to
results/answers/<run>/judge_agreement.json.
"""

import argparse
import csv
import json
from pathlib import Path

from multimodal_rag.evalset import load_questions

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ["condition", "qid", "question", "reference", "answer", "deepseek_correct", "deepseek_reason", "human_correct"]


def latest_clean_answer_run() -> Path:
    runs = [p.parent for p in sorted((ROOT / "results" / "answers").glob("*/run.json"))
            if not json.loads(p.read_text(encoding="utf-8"))["git"]["dirty"]]
    if not runs:
        raise SystemExit("no clean answer run in results/answers/")
    return runs[-1]


def check_file(run_dir: Path) -> Path:
    return ROOT / "eval" / f"judge_check_{run_dir.name}.csv"


def same_item(r: dict) -> tuple:
    return (r["question"], r["reference"], r["answer"], r["deepseek_correct"])


def export() -> None:
    run_dir = latest_clean_answer_run()
    out = check_file(run_dir)
    if out.exists():
        raise SystemExit(f"{out.relative_to(ROOT)} already exists; it may hold your labels, so it is not overwritten")
    earlier = {}
    for path in sorted((ROOT / "eval").glob("judge_check_*.csv")):
        with path.open(encoding="utf-8") as f:
            earlier |= {same_item(r): r["human_correct"] for r in csv.DictReader(f) if r["human_correct"].strip()}
    _, questions = load_questions(ROOT / "eval" / "questions.csv")
    by_qid = {q["qid"]: q for q in questions}
    with (run_dir / "per_question.csv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["grader"] != "rule"]
    carried = 0
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            q = by_qid[r["qid"]]
            row = {"condition": r["condition"], "qid": r["qid"], "question": q["question"],
                   "reference": q["answer"], "answer": r["answer"],
                   "deepseek_correct": r["correct"].lower(), "deepseek_reason": r["note"], "human_correct": ""}
            row["human_correct"] = earlier.get(same_item(row), "")
            carried += bool(row["human_correct"])
            writer.writerow(row)
    print(f"wrote {len(rows)} rows from {run_dir.relative_to(ROOT)} to {out.relative_to(ROOT)}; "
          f"{carried} carried over from earlier checks, {len(rows) - carried} to label")


def agree() -> None:
    run_dir = latest_clean_answer_run()
    with check_file(run_dir).open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["human_correct"].strip()]
    bad = [r["human_correct"] for r in rows if r["human_correct"].strip().lower() not in ("true", "false")]
    if bad:
        raise SystemExit(f"human_correct must be true or false, got {bad[:3]}")
    if not rows:
        raise SystemExit("no rows have human_correct filled in yet")
    same = [r for r in rows if r["human_correct"].strip().lower() == r["deepseek_correct"]]
    differ = [(r["condition"], r["qid"]) for r in rows if r not in same]
    result = {"n_checked": len(rows), "n_agree": len(same), "agreement": len(same) / len(rows), "disagree": differ}
    out = run_dir / "judge_agreement.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["export", "agree"])
    {"export": export, "agree": agree}[parser.parse_args().command]()
