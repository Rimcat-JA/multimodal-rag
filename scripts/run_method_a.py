"""Evaluate method A (page text + BGE-M3) on the evaluation set.

Run with:  uv run python scripts/run_method_a.py [--config configs/method_a.toml]

Every page is ranked for every question (search depth = number of pages), so
MRR sees the full ranking. Page vectors are cached in outputs/cache/method_a/,
keyed by the model settings and the hash of data/pages.jsonl, and reused when
nothing changed. Results go to results/method_a/<timestamp>/.
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

import numpy as np

from multimodal_rag.eval.runner import file_sha256, run_retrieval, save_run
from multimodal_rag.evalset import load_questions
from multimodal_rag.retrieval.text_dense import TextDenseRetriever, load_encoder

ROOT = Path(__file__).resolve().parent.parent
PAGES = ROOT / "data" / "pages.jsonl"
QUESTIONS = ROOT / "eval" / "questions.csv"
CACHE_DIR = ROOT / "outputs" / "cache" / "method_a"
RESULTS_DIR = ROOT / "results" / "method_a"


def git_state() -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    # Uncommitted results from earlier runs are not code changes, so leave results/ out.
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", ".", ":!results"], cwd=ROOT, capture_output=True, text=True,
    ).stdout
    return {"commit": commit, "dirty": bool(dirty.strip())}


def cache_key(config: dict, pages_sha: str) -> str:
    # Only the settings that change the vectors; batch_size does not.
    fields = {k: config[k] for k in ("model", "revision", "torch_dtype", "max_seq_length")}
    return hashlib.sha256(json.dumps([fields, pages_sha], sort_keys=True).encode()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "method_a.toml")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))

    with PAGES.open(encoding="utf-8") as f:
        pages = [json.loads(line) for line in f]
    page_ids = [p["page_id"] for p in pages]
    _, questions = load_questions(QUESTIONS)
    pages_sha = file_sha256(PAGES)

    encoder = load_encoder(config)
    retriever = TextDenseRetriever(encoder, batch_size=config["batch_size"])

    token_counts = [len(encoder.tokenizer(p["text"])["input_ids"]) for p in pages]
    truncated = [pid for pid, n in zip(page_ids, token_counts) if n > config["max_seq_length"]]
    print(f"{len(pages)} pages, max {max(token_counts)} tokens, {len(truncated)} truncated")

    cache = CACHE_DIR / f"{cache_key(config, pages_sha)}.npz"
    if cache.exists():
        saved = np.load(cache)
        retriever.load_index(list(saved["page_ids"]), saved["vectors"])
        print(f"loaded page vectors from {cache.relative_to(ROOT)}")
    else:
        retriever.index(pages)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, page_ids=np.array(page_ids), vectors=retriever.page_vectors)
        print(f"saved page vectors to {cache.relative_to(ROOT)}")

    results = run_retrieval(retriever, questions, set(page_ids), k=len(pages))

    started = datetime.now()
    meta = {
        "method": config["method"],
        "config": config,
        "search_depth": len(pages),
        "timestamp": started.isoformat(timespec="seconds"),
        "eval_set": {"path": "eval/questions.csv", "sha256": file_sha256(QUESTIONS), "n": len(questions)},
        "pages": {"path": "data/pages.jsonl", "sha256": pages_sha, "n": len(pages)},
        "truncation": {"max_tokens": max(token_counts), "n_truncated": len(truncated), "truncated": truncated},
        "git": git_state(),
        "versions": {pkg: version(pkg) for pkg in ("torch", "sentence-transformers", "transformers")}
        | {"python": platform.python_version()},
    }
    out_dir = RESULTS_DIR / started.strftime("%Y%m%d-%H%M%S")
    run = save_run(out_dir, meta, results, questions)

    print(f"\nsaved to {out_dir.relative_to(ROOT)}")
    print(json.dumps(run["summary"], indent=2))


if __name__ == "__main__":
    main()
