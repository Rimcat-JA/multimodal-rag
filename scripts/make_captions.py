"""Caption every page image with a local vision LLM for method B2.

Run with:  uv run python scripts/make_captions.py --config configs/method_b2.toml [--pages ID ...]

Each page's reply is saved to outputs/cache/captions_b2/<settings hash>/<page_id>.json
as soon as it arrives, and pages already saved are skipped, so the run can be
stopped and restarted. --pages captions only the given pages (for trying the
prompt on a few pages that are not gold pages).
"""

import argparse
import json
import time
import tomllib
from pathlib import Path

from multimodal_rag.ingest.captions import cache_path, caption_page, check_model_digest, settings_key

ROOT = Path(__file__).resolve().parent.parent
PAGES = ROOT / "data" / "pages.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pages", nargs="*", help="only these page ids")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))["caption"]

    check_model_digest(config)
    with PAGES.open(encoding="utf-8") as f:
        pages = [json.loads(line) for line in f]
    if args.pages:
        pages = [p for p in pages if p["page_id"] in set(args.pages)]

    cache_dir = ROOT / "outputs" / "cache" / "captions_b2" / settings_key(config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Keep the settings next to the captions, so the cache explains itself.
    (cache_dir / "settings.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
    todo = [p for p in pages if not cache_path(cache_dir, p["page_id"]).exists()]
    print(f"{len(pages) - len(todo)} pages already captioned, {len(todo)} to go -> {cache_dir.relative_to(ROOT)}")

    started = time.time()
    errors = 0
    for i, page in enumerate(todo, start=1):
        record = caption_page(config, ROOT / page["image"])
        out = cache_path(cache_dir, page["page_id"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        errors += record["error"] is not None
        left = (time.time() - started) / i * (len(todo) - i)
        print(f"[{i}/{len(todo)}] {page['page_id']}: {len(record['figures'])} figures, "
              f"{record['total_duration_s']}s{' ERROR ' + record['error'] if record['error'] else ''}"
              f"  (about {left / 60:.0f} min left)", flush=True)
    print(f"done, {errors} pages with unparseable replies")


if __name__ == "__main__":
    main()
