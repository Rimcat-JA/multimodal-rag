"""Render every page in data/raw/ to PNG and extract its text.

Run with:  uv run python scripts/ingest_pages.py [--dpi 150] [--force]

Outputs
- data/pages/<company_id>/<period>/pNNN.png   page images
- data/pages/<company_id>/<period>/pages.jsonl per-document records (cache)
- data/pages.jsonl                             all page records, sorted by page id

Documents whose per-document records already exist are skipped unless --force.
Documents are processed in parallel, one process per document.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from multimodal_rag.ingest.pages import PageRecord, garbled_ratio, process_pdf

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PAGES_DIR = ROOT / "data" / "pages"
ALL_RECORDS = ROOT / "data" / "pages.jsonl"

# Thresholds for the quality report
NEAR_EMPTY_CHARS = 20
GARBLED_THRESHOLD = 0.05


def doc_id_for(pdf: Path) -> str:
    # data/raw/toyota/fy2025_full.pdf -> "toyota/fy2025_full"
    return f"{pdf.parent.name}/{pdf.stem}"


def ingest_one(pdf: Path, dpi: int, force: bool) -> tuple[str, int, bool]:
    doc_id = doc_id_for(pdf)
    out_dir = PAGES_DIR / doc_id
    cache = out_dir / "pages.jsonl"
    if cache.exists() and not force:
        n = sum(1 for _ in cache.open(encoding="utf-8"))
        return doc_id, n, False

    records = process_pdf(pdf, doc_id, out_dir, ROOT, dpi=dpi)
    with cache.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    return doc_id, len(records), True


def load_records(path: Path) -> list[PageRecord]:
    with path.open(encoding="utf-8") as f:
        return [PageRecord(**json.loads(line)) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--force", action="store_true", help="re-render documents that already have output")
    args = parser.parse_args()

    pdfs = sorted(RAW_DIR.glob("*/*.pdf"))
    with ProcessPoolExecutor() as pool:
        futures = [pool.submit(ingest_one, pdf, args.dpi, args.force) for pdf in pdfs]
        for fut in futures:
            doc_id, n, rendered = fut.result()
            print(f"{doc_id}: {n} pages {'rendered' if rendered else '(cached)'}")

    records: list[PageRecord] = []
    for pdf in pdfs:
        records += load_records(PAGES_DIR / doc_id_for(pdf) / "pages.jsonl")
    records.sort(key=lambda r: r.page_id)
    with ALL_RECORDS.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    near_empty = [r for r in records if len(r.text.strip()) < NEAR_EMPTY_CHARS]
    garbled = [(r, garbled_ratio(r.text)) for r in records]
    garbled = [(r, g) for r, g in garbled if g > GARBLED_THRESHOLD]

    print(f"\n{len(records)} pages from {len(pdfs)} documents -> {ALL_RECORDS.relative_to(ROOT).as_posix()}")
    print(f"near-empty text (< {NEAR_EMPTY_CHARS} chars): {len(near_empty)}")
    for r in near_empty[:10]:
        print(f"  {r.page_id}")
    print(f"garbled text (> {GARBLED_THRESHOLD:.0%} broken chars): {len(garbled)}")
    for r, g in sorted(garbled, key=lambda x: -x[1])[:10]:
        print(f"  {r.page_id}: {g:.0%}")


if __name__ == "__main__":
    main()
