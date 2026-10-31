"""Evaluate one retrieval method on the evaluation set.

Run with:  uv run python scripts/run_eval.py --config configs/method_a.toml
           uv run python scripts/run_eval.py --config configs/method_c.toml

Every page is ranked for every question (search depth = number of pages), so
MRR sees the full ranking. The page index is cached in outputs/cache/<method>/,
keyed by the settings that change it and the hash of data/pages.jsonl, and
reused when nothing changed. Results go to results/method_<x>/<timestamp>/.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import tomllib
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from multimodal_rag.eval.runner import file_sha256, run_retrieval, save_run
from multimodal_rag.evalset import load_questions

ROOT = Path(__file__).resolve().parent.parent
PAGES = ROOT / "data" / "pages.jsonl"
QUESTIONS = ROOT / "eval" / "questions.csv"

# Settings that change the index (and so belong in the cache key) per method.
# Batch sizes and chunking only change speed and memory, not the vectors.
INDEX_SETTINGS = {
    "A": ("model", "revision", "torch_dtype", "max_seq_length"),
    "C": ("model", "revision", "torch_dtype"),
}
INDEX_SUFFIX = {"A": ".npz", "C": ".pt"}


def git_state() -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    # Uncommitted results from earlier runs are not code changes, so leave results/ out.
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", ".", ":!results"], cwd=ROOT, capture_output=True, text=True,
    ).stdout
    return {"commit": commit, "dirty": bool(dirty.strip())}


def cache_key(config: dict, pages_sha: str) -> str:
    fields = {k: config[k] for k in INDEX_SETTINGS[config["method"]]}
    return hashlib.sha256(json.dumps([fields, pages_sha], sort_keys=True).encode()).hexdigest()[:16]


def build_retriever(config: dict, pages: list[dict]) -> tuple[object, dict]:
    """Load the model for config["method"] and return (retriever, method-specific notes for run.json)."""
    method = config["method"]
    if method == "A":
        from multimodal_rag.retrieval.text_dense import TextDenseRetriever, load_encoder

        encoder = load_encoder(config)
        token_counts = [len(encoder.tokenizer(p["text"])["input_ids"]) for p in pages]
        truncated = [p["page_id"] for p, n in zip(pages, token_counts) if n > config["max_seq_length"]]
        print(f"{len(pages)} pages, max {max(token_counts)} tokens, {len(truncated)} truncated")
        notes = {"truncation": {"max_tokens": max(token_counts), "n_truncated": len(truncated), "truncated": truncated}}
        return TextDenseRetriever(encoder, batch_size=config["batch_size"]), notes
    if method == "C":
        from multimodal_rag.retrieval.page_image import PageImageRetriever, load_encoder

        encoder = load_encoder(config)
        retriever = PageImageRetriever(encoder, ROOT, batch_size=config["batch_size"], chunk_pages=config["chunk_pages"])
        return retriever, {}
    raise ValueError(f"unknown method {method!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    method = config["method"]

    with PAGES.open(encoding="utf-8") as f:
        pages = [json.loads(line) for line in f]
    page_ids = [p["page_id"] for p in pages]
    _, questions = load_questions(QUESTIONS)
    pages_sha = file_sha256(PAGES)

    retriever, notes = build_retriever(config, pages)

    cache = ROOT / "outputs" / "cache" / f"method_{method.lower()}" / f"{cache_key(config, pages_sha)}{INDEX_SUFFIX[method]}"
    if cache.exists():
        retriever.load_index(cache)
        print(f"loaded index from {cache.relative_to(ROOT)}")
    else:
        retriever.index(pages)
        cache.parent.mkdir(parents=True, exist_ok=True)
        retriever.save_index(cache)
        print(f"saved index to {cache.relative_to(ROOT)}")
    if retriever.page_ids != page_ids:
        raise RuntimeError(f"{cache} does not match data/pages.jsonl; delete it and rerun")

    results = run_retrieval(retriever, questions, set(page_ids), k=len(pages))

    started = datetime.now()
    meta = {
        "method": method,
        "config": config,
        "search_depth": len(pages),
        "timestamp": started.isoformat(timespec="seconds"),
        "eval_set": {"path": "eval/questions.csv", "sha256": file_sha256(QUESTIONS), "n": len(questions)},
        "pages": {"path": "data/pages.jsonl", "sha256": pages_sha, "n": len(pages)},
        **notes,
        "git": git_state(),
        "versions": {pkg: version(pkg) for pkg in ("torch", "sentence-transformers", "transformers", "peft")}
        | {"python": platform.python_version()},
    }
    out_dir = ROOT / "results" / f"method_{method.lower()}" / started.strftime("%Y%m%d-%H%M%S")
    run = save_run(out_dir, meta, results, questions)

    print(f"\nsaved to {out_dir.relative_to(ROOT)}")
    print(json.dumps(run["summary"], indent=2))


if __name__ == "__main__":
    main()
