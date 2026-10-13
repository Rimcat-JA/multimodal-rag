"""Check the format of eval/questions.csv (see eval/README.md).

Run with:  uv run python scripts/validate_eval.py [path]

Only the format is checked: header, required fields, allowed values, unique
qids, page ids that exist in data/pages.jsonl, and number formatting.
The content of questions and answers is never judged or changed.
"""

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "eval" / "questions.csv"
PAGES = ROOT / "data" / "pages.jsonl"

HEADER = [
    "qid", "question", "gold_pages", "gold_mode", "answer",
    "answer_value", "answer_unit", "q_type", "evidence", "notes",
]
REQUIRED = ["qid", "question", "gold_pages", "gold_mode", "answer", "q_type", "evidence"]
ALLOWED = {
    "gold_mode": {"all", "any"},
    "q_type": {"lookup", "compare", "trend", "structure"},
    "evidence": {"chart", "table", "diagram", "text"},
}
QID_RE = re.compile(r"^q\d{3,}$")
# Plain number: optional minus, digits, optional decimals. No commas, units or "約".
NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?$")


def validate(rows: list[dict], header: list[str], page_ids: set[str]) -> list[str]:
    errors = []
    if header != HEADER:
        errors.append(f"header must be exactly: {','.join(HEADER)}")
        return errors

    seen = {}
    for line, row in enumerate(rows, start=2):  # line 1 is the header
        where = f"line {line} ({row.get('qid') or 'no qid'})"
        for col in REQUIRED:
            if not (row.get(col) or "").strip():
                errors.append(f"{where}: '{col}' is empty")

        qid = row["qid"].strip()
        if qid and not QID_RE.match(qid):
            errors.append(f"{where}: qid should look like q001")
        if qid in seen:
            errors.append(f"{where}: qid duplicates line {seen[qid]}")
        seen.setdefault(qid, line)

        for col, allowed in ALLOWED.items():
            value = row[col].strip()
            if value and value not in allowed:
                errors.append(f"{where}: {col}='{value}' is not one of {sorted(allowed)}")

        pages = [p.strip() for p in row["gold_pages"].split(";") if p.strip()]
        for p in pages:
            if p not in page_ids:
                errors.append(f"{where}: page id '{p}' not found in data/pages.jsonl")
        if len(pages) != len(set(pages)):
            errors.append(f"{where}: gold_pages lists the same page twice")

        value, unit = row["answer_value"].strip(), row["answer_unit"].strip()
        if value and not NUMBER_RE.match(value):
            errors.append(f"{where}: answer_value '{value}' must be a plain number (no commas, units or 約)")
        if value and not unit:
            errors.append(f"{where}: answer_value is set but answer_unit is empty")
        if unit and not value:
            errors.append(f"{where}: answer_unit is set but answer_value is empty")
    return errors


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    if not path.exists():
        sys.exit(f"{path} not found")

    with PAGES.open(encoding="utf-8") as f:
        page_ids = {json.loads(line)["page_id"] for line in f}

    # utf-8-sig also accepts a BOM, so a file saved by Excel won't fail on the header.
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)

    errors = validate(rows, header, page_ids)
    for e in errors:
        print(e)
    print(f"\n{len(rows)} questions, {len(errors)} problems")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
