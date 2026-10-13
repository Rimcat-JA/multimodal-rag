"""Read, write, normalize and validate the evaluation set (eval/questions.csv).

The questions themselves are written by the developer (see CLAUDE.md). This
module only handles the file format: the editor app and the validator script
both go through it, so the CSV always has the same shape.
"""

import csv
import re
import unicodedata
from pathlib import Path

HEADER = [
    "qid", "question", "gold_pages", "gold_mode", "answer",
    "answer_value", "answer_unit", "q_type", "evidence", "notes",
]
REQUIRED = ["qid", "question", "gold_pages", "gold_mode", "answer", "q_type", "evidence"]

# Codes stored in the CSV -> labels shown to the person filling it in.
GOLD_MODE_LABELS = {
    "all": "すべて必要（根拠が1ページだけのときもこちら）",
    "any": "どれか1つで足りる（同じ値が複数ページにある）",
}
Q_TYPE_LABELS = {
    "lookup": "1つの値を読む",
    "compare": "2つ以上の値を比べる（差・比率を含む）",
    "trend": "推移・傾向を読む",
    "structure": "図の構造を読む",
}
EVIDENCE_LABELS = {
    "chart": "グラフ",
    "table": "表",
    "diagram": "図",
    "text": "本文の文字",
}
COMMON_UNITS = ["億円", "百万円", "兆円", "千台", "%", "pt", "円", "件", "人"]

QID_RE = re.compile(r"^q\d{3,}$")
# Plain number: optional minus, digits, optional decimals. No commas, units or "約".
NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?$")


def normalize_number(text: str) -> str:
    """Turn a number as typed (or as printed in an IR deck) into the CSV form.

    "２，９３４" -> "2934", "△1,925" -> "-1925", "▲3.1" -> "-3.1".
    Japanese IR decks print negatives with △ or ▲. Anything that still isn't a
    plain number afterwards (e.g. "約15") is left as is so validation flags it.
    """
    s = unicodedata.normalize("NFKC", text).strip().replace(",", "")
    if s[:1] in ("△", "▲"):
        s = "-" + s[1:].strip()
    return s


def qid_number(qid: str) -> int:
    return int(qid[1:])


def next_qid(rows: list[dict]) -> str:
    """One past the highest qid in the file. Deleted qids are never reused
    as long as the highest one stays in the file."""
    used = [qid_number(r["qid"]) for r in rows if QID_RE.match(r.get("qid", ""))]
    return f"q{(max(used) + 1 if used else 1):03d}"


def load_questions(path: Path) -> tuple[list[str], list[dict]]:
    if not path.exists():
        return list(HEADER), []
    # utf-8-sig also accepts a BOM, in case the file was saved by Excel.
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def save_questions(path: Path, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: qid_number(r["qid"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in HEADER} for r in rows)


def split_pages(gold_pages: str) -> list[str]:
    return [p.strip() for p in gold_pages.split(";") if p.strip()]


def validate_row(row: dict, page_ids: set[str]) -> list[str]:
    """Format problems in one row, as Japanese messages for the person editing."""
    errors = []
    labels = {
        "qid": "qid", "question": "質問", "gold_pages": "根拠ページ", "gold_mode": "根拠ページの扱い",
        "answer": "正解（文）", "q_type": "聞き方", "evidence": "根拠の種類",
    }
    for col in REQUIRED:
        if not (row.get(col) or "").strip():
            errors.append(f"「{labels[col]}」が空です")

    qid = (row.get("qid") or "").strip()
    if qid and not QID_RE.match(qid):
        errors.append(f"qid「{qid}」は q001 の形にしてください")

    for col, allowed in (("gold_mode", GOLD_MODE_LABELS), ("q_type", Q_TYPE_LABELS), ("evidence", EVIDENCE_LABELS)):
        value = (row.get(col) or "").strip()
        if value and value not in allowed:
            errors.append(f"{col}「{value}」は {sorted(allowed)} のどれかにしてください")

    pages = split_pages(row.get("gold_pages") or "")
    for p in pages:
        if p not in page_ids:
            errors.append(f"ページID「{p}」が data/pages.jsonl にありません")
    if len(pages) != len(set(pages)):
        errors.append("根拠ページに同じページが2回入っています")

    value = (row.get("answer_value") or "").strip()
    unit = (row.get("answer_unit") or "").strip()
    if value and not NUMBER_RE.match(value):
        errors.append(f"数値「{value}」は数字だけにしてください（カンマ・単位・「約」は不可）")
    if value and not unit:
        errors.append("数値があるのに単位が空です")
    if unit and not value:
        errors.append("単位があるのに数値が空です")
    return errors


def validate(rows: list[dict], header: list[str], page_ids: set[str]) -> list[str]:
    """Format problems in the whole file, each prefixed with where it is."""
    if header != HEADER:
        return [f"1行目のヘッダーは次のとおりにしてください: {','.join(HEADER)}"]

    errors = []
    seen: dict[str, int] = {}
    for line, row in enumerate(rows, start=2):  # line 1 is the header
        where = f"{line}行目 ({row.get('qid') or 'qidなし'})"
        errors += [f"{where}: {e}" for e in validate_row(row, page_ids)]
        qid = (row.get("qid") or "").strip()
        if qid in seen:
            errors.append(f"{where}: qid が {seen[qid]}行目と重複しています")
        seen.setdefault(qid, line)
    return errors
