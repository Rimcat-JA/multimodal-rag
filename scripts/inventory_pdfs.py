"""Summarize the downloaded PDFs: pages, layout, and how chart-heavy they are.

Run with:  uv run python scripts/inventory_pdfs.py

Per document it reports:
- pages and the share of landscape pages (slide decks are landscape)
- vector drawing paths per page: charts and diagrams in IR decks are usually
  vector graphics, so this is a rough "how many charts" signal. Note that
  PyMuPDF's get_images() would miss these entirely.
- raster images per page, and extractable text characters per page
The table is also written to outputs/inventory.csv.
"""

import csv
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT = ROOT / "outputs" / "inventory.csv"


def summarize(path: Path) -> dict:
    doc = pymupdf.open(path)
    pages = len(doc)
    landscape = drawings = images = chars = 0
    for page in doc:
        if page.rect.width > page.rect.height:
            landscape += 1
        drawings += len(page.get_drawings())
        images += len(page.get_images(full=True))
        chars += len(page.get_text())
    doc.close()
    return {
        "doc": path.relative_to(RAW_DIR).as_posix(),
        "pages": pages,
        "landscape_pct": round(100 * landscape / pages),
        "drawings_per_page": round(drawings / pages),
        "images_per_page": round(images / pages, 1),
        "chars_per_page": round(chars / pages),
    }


def main() -> None:
    rows = [summarize(p) for p in sorted(RAW_DIR.glob("*/*.pdf"))]

    header = list(rows[0])
    widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in header}
    print("  ".join(k.ljust(widths[k]) for k in header))
    for r in rows:
        print("  ".join(str(r[k]).ljust(widths[k]) for k in header))
    print(f"\n{len(rows)} documents, {sum(r['pages'] for r in rows)} pages")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
