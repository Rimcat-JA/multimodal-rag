from pathlib import Path

import pymupdf
from PIL import Image

from multimodal_rag.ingest.pages import garbled_ratio, page_id, process_pdf


def make_pdf(path: Path) -> None:
    """Two landscape pages (960x540 pt, like a 16:9 slide) with known text."""
    doc = pymupdf.open()
    for text in ["Revenue 12.5", "営業利益 4.3"]:
        page = doc.new_page(width=960, height=540)
        # "japan" is a built-in CJK font, so no font file is needed.
        page.insert_text((72, 100), text, fontname="japan", fontsize=24)
    doc.save(path)
    doc.close()


def test_page_id_is_one_based_and_zero_padded():
    assert page_id("toyota/fy2025_full", 1) == "toyota/fy2025_full/p001"
    assert page_id("toyota/fy2025_full", 123) == "toyota/fy2025_full/p123"


def test_process_pdf_renders_every_page(tmp_path):
    pdf = tmp_path / "doc.pdf"
    make_pdf(pdf)
    image_dir = tmp_path / "pages" / "acme" / "fy2025_full"

    records = process_pdf(pdf, "acme/fy2025_full", image_dir, tmp_path, dpi=144)

    assert [r.page for r in records] == [1, 2]
    assert [r.page_id for r in records] == ["acme/fy2025_full/p001", "acme/fy2025_full/p002"]

    # 960 pt * 144/72 = 1920 px, 540 pt * 2 = 1080 px
    for r in records:
        assert (r.width, r.height) == (1920, 1080)
        assert r.image == f"pages/acme/fy2025_full/p{r.page:03d}.png"
        with Image.open(tmp_path / r.image) as img:
            assert img.size == (1920, 1080)


def test_process_pdf_extracts_text_per_page(tmp_path):
    pdf = tmp_path / "doc.pdf"
    make_pdf(pdf)

    records = process_pdf(pdf, "acme/fy2025_full", tmp_path / "out", tmp_path, dpi=72)

    assert "Revenue 12.5" in records[0].text
    assert "営業利益 4.3" in records[1].text
    assert "営業利益" not in records[0].text


def test_garbled_ratio():
    assert garbled_ratio("") == 0.0
    assert garbled_ratio("売上高 100") == 0.0
    assert garbled_ratio("��売上") == 0.5
    assert garbled_ratio("abc") == 0.25
