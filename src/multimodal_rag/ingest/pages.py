"""Render PDF pages to PNG images and extract their text.

Every retrieval method in this project starts from the same per-page records,
so page ids defined here (e.g. "toyota/fy2025_full/p001") are also what the
evaluation set uses to mark the correct page.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

# Characters that signal a broken text layer: the Unicode replacement character
# and the Private Use Area, which PDFs emit when a font has no Unicode mapping.
_GARBLED_RANGES = [(0xFFFD, 0xFFFD), (0xE000, 0xF8FF)]


@dataclass
class PageRecord:
    page_id: str  # "<doc_id>/p<NNN>", page numbers are 1-based
    doc_id: str  # "<company_id>/<period>", e.g. "toyota/fy2025_full"
    page: int
    image: str  # PNG path relative to the repo root, forward slashes
    width: int  # image size in pixels
    height: int
    text: str  # raw text layer, unnormalized

    def to_dict(self) -> dict:
        return asdict(self)


def page_id(doc_id: str, page: int) -> str:
    return f"{doc_id}/p{page:03d}"


def garbled_ratio(text: str) -> float:
    """Share of non-whitespace characters that look like a broken text layer."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    bad = sum(any(lo <= ord(c) <= hi for lo, hi in _GARBLED_RANGES) for c in chars)
    return bad / len(chars)


def process_pdf(pdf_path: Path, doc_id: str, image_dir: Path, root: Path, dpi: int = 150) -> list[PageRecord]:
    """Render each page of `pdf_path` into `image_dir` and return one record per page.

    `root` is the directory image paths are made relative to (the repo root),
    so records stay valid if the repo is moved.
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with pymupdf.open(pdf_path) as doc:
        for index, pg in enumerate(doc):
            number = index + 1
            pix = pg.get_pixmap(dpi=dpi, alpha=False)
            out = image_dir / f"p{number:03d}.png"
            pix.save(out)
            records.append(
                PageRecord(
                    page_id=page_id(doc_id, number),
                    doc_id=doc_id,
                    page=number,
                    image=out.relative_to(root).as_posix(),
                    width=pix.width,
                    height=pix.height,
                    # sort=True orders text blocks top-to-bottom, left-to-right,
                    # which reads more naturally than the PDF's internal order.
                    text=pg.get_text("text", sort=True),
                )
            )
    return records
