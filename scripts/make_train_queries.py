"""Write synthetic training queries for every page of the training decks (Gemini via OpenRouter).

Run with:  uv run python scripts/make_train_queries.py --config configs/finetune_c.toml --limit 5   (trial)
           uv run python scripts/make_train_queries.py --config configs/finetune_c.toml              (all pages)

Reads data/train/pages.jsonl and docs/train_sources.csv (company name and
period label for the prompt). Every reply is cached under outputs/cache/train_queries/,
so a stopped run resumes and a rerun costs nothing. Writes data/train/queries.jsonl:
one line per (query, page). A reply that can't be parsed is reported and gives no queries.
"""

import argparse
import base64
import csv
import json
import tomllib
from pathlib import Path

from multimodal_rag.finetune.queries import parse_queries, query_cache_path, query_request_body
from multimodal_rag.generation.api import api_key, call_with_cache, chat

ROOT = Path(__file__).resolve().parent.parent
PAGES = ROOT / "data" / "train" / "pages.jsonl"
SOURCES = ROOT / "docs" / "train_sources.csv"
CACHE = ROOT / "outputs" / "cache" / "train_queries"
OUT = ROOT / "data" / "train" / "queries.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="only the first N pages (trial; queries.jsonl is not written)")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))["queries"]

    with SOURCES.open(encoding="utf-8") as f:
        decks = {f"{r['company_id']}/{r['period']}": r for r in csv.DictReader(f)}
    with PAGES.open(encoding="utf-8") as f:
        pages = [json.loads(line) for line in f]
    if args.limit:
        pages = pages[:: max(1, len(pages) // args.limit)][: args.limit]  # spread the trial over decks

    key = api_key()
    rows, n_failed, cost = [], 0, 0.0
    for i, page in enumerate(pages, 1):
        deck = decks[page["doc_id"]]

        def call() -> dict:
            image = base64.b64encode((ROOT / page["image"]).read_bytes()).decode()
            return chat(query_request_body(config, deck["company"], deck["period_label"], image), key)

        record = call_with_cache(query_cache_path(CACHE, config, page["page_id"]), {"page_id": page["page_id"]}, call)
        cost += record["response"].get("usage", {}).get("cost", 0.0)
        try:
            queries = parse_queries(record["response"], config["queries_per_page"])
        except ValueError as e:
            n_failed += 1
            print(f"{page['page_id']}: {e}")
            continue
        rows += [{"query": q, "page_id": page["page_id"]} for q in queries]
        if args.limit or i % 50 == 0:
            print(f"[{i}/{len(pages)}] {page['page_id']}: {queries}")

    n_with = len({r["page_id"] for r in rows})
    print(f"{len(rows)} queries for {n_with}/{len(pages)} pages; {n_failed} unparseable; cost of these replies ${cost:.3f}")
    if args.limit:
        print("trial run: queries.jsonl not written")
        return
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"saved to {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
