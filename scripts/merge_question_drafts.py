"""Merge Claude-drafted questions (outputs/qgen/*.json) into eval/questions.csv.

Run with:  uv run python scripts/merge_question_drafts.py DRAFT.json [DRAFT.json ...] [--dry-run]

Draft files must be named explicitly: outputs/qgen/ also holds drafts that were
already merged and review files, and merging those again would duplicate questions.

For each draft:
- numbers are normalized (commas, full-width digits, △/▲ negatives)
- pages of the same company whose text contains the same value are added to
  the gold group as alternatives ("a|b"). Only values with 4+ digits are added
  automatically; shorter values (e.g. "100") match unrelated pages too often,
  so their candidates are only listed in the report for the reviewer.
- drafts still say gold_mode "all"/"any"; "all" with several pages becomes one
  group per page ("a;b") and is never expanded
- qids continue after the highest existing one; author=claude, reviewed=none

Every row is validated before anything is written. A report of the changes is
saved to outputs/qgen/merge_report.json for the review step.
"""

import argparse
import json
import unicodedata
from pathlib import Path

from multimodal_rag.evalset import format_gold, load_questions, next_qid, normalize_number, save_questions, validate

ROOT = Path(__file__).resolve().parent.parent
DRAFTS = ROOT / "outputs" / "qgen"
QUESTIONS = ROOT / "eval" / "questions.csv"
PAGES = ROOT / "data" / "pages.jsonl"
REPORT = DRAFTS / "merge_report.json"

MIN_DIGITS_FOR_AUTO_ADD = 4


def value_variants(value: str) -> set[str]:
    """How a plain number may be printed in a deck: 2934 -> {"2934", "2,934"}."""
    sign = "-" if value.startswith("-") else ""
    whole, _, frac = value.lstrip("-").partition(".")
    grouped = f"{int(whole):,}" + (f".{frac}" if frac else "")
    variants = {value.lstrip("-"), grouped}
    if sign:  # decks print negatives with △ / ▲
        variants = {p + v for v in variants for p in ("△", "▲", "-", "△ ", "▲ ")}
    return variants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("drafts", nargs="+", type=Path, help="draft JSON files to merge")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with PAGES.open(encoding="utf-8") as f:
        # NFKC: some decks' text layers use Kangxi radicals (⽀) instead of the usual kanji (支)
        pages = {r["page_id"]: unicodedata.normalize("NFKC", r["text"]) for r in map(json.loads, f)}

    header, rows = load_questions(QUESTIONS)
    report = []
    new_rows = []
    for draft_file in args.drafts:
        for d in json.loads(draft_file.read_text(encoding="utf-8")):
            value = normalize_number(str(d.get("answer_value") or ""))
            gold = list(dict.fromkeys(d["gold_pages"]))
            mode = d["gold_mode"]
            entry = {"source": draft_file.name, "question": d["question"], "drafted_gold": list(gold)}

            cross_page = mode == "all" and len(gold) > 1
            if value and not cross_page:
                company = gold[0].split("/")[0]
                variants = value_variants(value)
                found = [p for p, text in pages.items()
                         if p.startswith(company + "/") and p not in gold and any(v in text for v in variants)]
                digits = len(value.lstrip("-").replace(".", ""))
                if found and digits >= MIN_DIGITS_FOR_AUTO_ADD:
                    gold += found
                    entry["auto_added"] = found
                elif found:
                    entry["candidates_not_added"] = found

            row = {
                "qid": next_qid(rows + new_rows),
                "question": d["question"].strip(),
                # several pages that are all needed -> one group each; otherwise one group of alternatives
                "gold_pages": format_gold([[p] for p in gold] if cross_page else [gold]),
                "answer": d["answer"].strip(),
                "answer_value": value,
                "answer_unit": (d.get("answer_unit") or "").strip(),
                "q_type": d["q_type"],
                "evidence": d["evidence"],
                "author": "claude",
                "reviewed": "none",
                "notes": (d.get("notes") or "").strip(),
            }
            entry["qid"] = row["qid"]
            new_rows.append(row)
            report.append(entry)

    errors = validate(rows + new_rows, header, set(pages))
    for e in errors:
        print(e)
    if errors:
        raise SystemExit(f"{len(errors)} problems; nothing written")

    for e in report:
        extra = f" +{len(e['auto_added'])} pages" if "auto_added" in e else ""
        warn = f" (short value: {len(e['candidates_not_added'])} candidates for review)" if "candidates_not_added" in e else ""
        print(f"{e['qid']} [{e['source']}]{extra}{warn}")

    if args.dry_run:
        print(f"\n--dry-run: {len(new_rows)} questions would be added")
        return
    save_questions(QUESTIONS, rows + new_rows)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nadded {len(new_rows)} questions -> {QUESTIONS.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
