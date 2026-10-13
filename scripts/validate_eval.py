"""Check the format of eval/questions.csv (see eval/README.md).

Run with:  uv run python scripts/validate_eval.py [path]

Only the format is checked: header, required fields, allowed values, unique
qids, page ids that exist in data/pages.jsonl, and number formatting.
The content of questions and answers is never judged or changed.
"""

import json
import sys
from pathlib import Path

from multimodal_rag.evalset import load_questions, validate

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "eval" / "questions.csv"
PAGES = ROOT / "data" / "pages.jsonl"


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    if not path.exists():
        sys.exit(f"{path} not found")

    with PAGES.open(encoding="utf-8") as f:
        page_ids = {json.loads(line)["page_id"] for line in f}

    header, rows = load_questions(path)
    errors = validate(rows, header, page_ids)
    for e in errors:
        print(e)
    print(f"\n{len(rows)} questions, {len(errors)} problems")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
