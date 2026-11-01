"""Render every page in data/raw/ to PNG and extract its text.

Run with:  uv run python scripts/ingest_pages.py [--dpi 150] [--force] [--split train]

Outputs
- data/pages/<company_id>/<period>/pNNN.png   page images
- data/pages/<company_id>/<period>/pages.jsonl per-document records (cache)
- data/pages.jsonl                             all page records, sorted by page id

With --split train the same happens under data/train/ (raw/, pages/, pages.jsonl):
decks used only for fine-tuning are kept apart from the evaluation pages, so
data/pages.jsonl and its hash never change.

Documents whose per-document records already exist are skipped unless --force.
Documents are processed in parallel, one process per document. A document whose
result does not arrive within --timeout seconds is reported as failed (PyMuPDF
can hang on a page without raising) and data/pages.jsonl is not written: remove
the document from the sources list and its PDF, then run again.
"""

import argparse
import json
import sys
from multiprocessing import Pool, TimeoutError
from pathlib import Path

from multimodal_rag.ingest.pages import PageRecord, garbled_ratio, process_pdf

ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIRS = {"eval": ROOT / "data", "train": ROOT / "data" / "train"}
RAW_DIR = ROOT / "data" / "raw"
PAGES_DIR = ROOT / "data" / "pages"
ALL_RECORDS = ROOT / "data" / "pages.jsonl"

# Thresholds for the quality report
NEAR_EMPTY_CHARS = 20
GARBLED_THRESHOLD = 0.05


def doc_id_for(pdf: Path) -> str:
    # data/raw/toyota/fy2025_full.pdf -> "toyota/fy2025_full"
    return f"{pdf.parent.name}/{pdf.stem}"


def ingest_one(pdf: Path, dpi: int, force: bool, pages_dir: Path) -> tuple[str, int, bool]:
    doc_id = doc_id_for(pdf)
    out_dir = pages_dir / doc_id
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
    parser.add_argument("--split", choices=list(SPLIT_DIRS), default="eval")
    parser.add_argument("--timeout", type=float, default=600, help="seconds to wait for each document's result")
    args = parser.parse_args()
    global RAW_DIR, PAGES_DIR, ALL_RECORDS
    base = SPLIT_DIRS[args.split]
    RAW_DIR, PAGES_DIR, ALL_RECORDS = base / "raw", base / "pages", base / "pages.jsonl"

    pdfs = sorted(RAW_DIR.glob("*/*.pdf"))
    failed = []
    # multiprocessing.Pool rather than ProcessPoolExecutor: leaving the `with` block
    # terminates the workers, so a worker stuck on a page does not keep running.
    with Pool() as pool:
        results = [pool.apply_async(ingest_one, (pdf, args.dpi, args.force, PAGES_DIR)) for pdf in pdfs]
        for pdf, res in zip(pdfs, results):
            try:
                doc_id, n, rendered = res.get(timeout=args.timeout)
            except TimeoutError:
                failed.append(doc_id_for(pdf))
                print(f"{doc_id_for(pdf)}: FAILED (no result within {args.timeout:.0f} s)", flush=True)
                continue
            print(f"{doc_id}: {n} pages {'rendered' if rendered else '(cached)'}", flush=True)
    if failed:
        sys.exit(f"{len(failed)} documents failed; {ALL_RECORDS.relative_to(ROOT).as_posix()} not written")

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
