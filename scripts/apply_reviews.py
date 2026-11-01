"""Apply an independent AI review to Claude-written questions in eval/questions.csv.

Run with:  uv run python scripts/apply_reviews.py REVIEW.json [REVIEW.json ...] [--dry-run]

Each review file is a JSON array of {"qid", "verdict", "issues", "fix"} written
by a reviewer agent that did not write the questions (CLAUDE.md):
    pass    reviewed none -> ai
    fix     the fields in "fix" replace the row's fields; reviewed stays none,
            so the fixed row has to pass a second, fresh review
    reject  the row is removed (its qid is left unused, as eval/README.md says)
Only rows with author=claude and reviewed=none are touched. Every row is
validated before anything is written.
"""

import argparse
import json
from pathlib import Path

from multimodal_rag.evalset import load_questions, normalize_number, save_questions, validate

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "questions.csv"
PAGES = ROOT / "data" / "pages.jsonl"
FIXABLE = {"question", "gold_pages", "answer", "answer_value", "answer_unit", "q_type", "evidence", "notes"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reviews", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    header, rows = load_questions(QUESTIONS)
    by_qid = {r["qid"]: r for r in rows}
    removed = set()
    for review_file in args.reviews:
        for item in json.loads(review_file.read_text(encoding="utf-8")):
            row = by_qid[item["qid"]]
            if row["author"] != "claude" or row["reviewed"] != "none":
                raise SystemExit(f"{item['qid']} is not an unreviewed Claude question; refusing to change it")
            verdict = item["verdict"]
            if verdict == "pass":
                row["reviewed"] = "ai"
            elif verdict == "fix":
                unknown = set(item["fix"]) - FIXABLE
                if unknown:
                    raise SystemExit(f"{item['qid']}: fix touches fields that can't be fixed: {unknown}")
                for field, value in item["fix"].items():
                    row[field] = normalize_number(value) if field == "answer_value" else value
            elif verdict == "reject":
                removed.add(item["qid"])
            else:
                raise SystemExit(f"{item['qid']}: unknown verdict {verdict!r}")
            print(f"{item['qid']}: {verdict}" + (f" ({', '.join(item['fix'])})" if verdict == "fix" else ""))

    rows = [r for r in rows if r["qid"] not in removed]
    with PAGES.open(encoding="utf-8") as f:
        page_ids = {json.loads(line)["page_id"] for line in f}
    errors = validate(rows, header, page_ids)
    for e in errors:
        print(e)
    if errors:
        raise SystemExit(f"{len(errors)} problems; nothing written")
    if args.dry_run:
        print("--dry-run: nothing written")
        return
    save_questions(QUESTIONS, rows)
    print(f"saved {len(rows)} questions")


if __name__ == "__main__":
    main()
