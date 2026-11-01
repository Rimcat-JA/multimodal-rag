"""Fine-tune ColQwen2.5 (method C) on training decks, choosing the checkpoint on held-out companies.

Run with:  uv run python scripts/finetune_colqwen.py mine  --config configs/finetune_c.toml
           uv run python scripts/finetune_colqwen.py train --config configs/finetune_c.toml

mine   indexes data/train/pages.jsonl with the base model (cached in outputs/cache/finetune/)
       and writes data/train/triplets.jsonl: each training query with its positive page and
       one hard negative (src/multimodal_rag/finetune/data.py).
train  trains the LoRA adapters with in-batch + hard negatives (CachedMultiVectorMultipleNegatives-
       RankingLoss), evaluates MRR@10 on the validation companies' queries against all their pages,
       keeps the best checkpoint, and saves it to outputs/models/colqwen_ft/<timestamp>/.
       Metadata (config, data hashes, validation scores, git state) goes to results/finetune/<timestamp>/.

The evaluation set (data/pages.jsonl, eval/questions.csv) is never read here.
"""

import argparse
import csv
import hashlib
import json
import platform
import random
import tomllib
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from run_eval import git_state  # scripts/ is on sys.path when this file is run

from multimodal_rag.eval.runner import file_sha256
from multimodal_rag.finetune.data import company_of, overlapping_companies, pick_hard_negative, split_by_company

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "train"
PAGES = TRAIN / "pages.jsonl"
QUERIES = TRAIN / "queries.jsonl"
TRIPLETS = TRAIN / "triplets.jsonl"
SOURCES = ROOT / "docs" / "train_sources.csv"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def check_no_eval_companies() -> None:
    with SOURCES.open(encoding="utf-8") as f, (ROOT / "docs" / "sources.csv").open(encoding="utf-8") as g:
        overlap = overlapping_companies(list(csv.DictReader(f)), list(csv.DictReader(g)))
    if overlap:
        raise SystemExit(f"training decks include evaluation companies {sorted(overlap)}")


def val_companies(config: dict, pages: list[dict]) -> set[str]:
    companies = sorted({company_of(p["page_id"]) for p in pages})
    return set(random.Random(config["split"]["seed"]).sample(companies, config["split"]["n_val_companies"]))


def load_base(config: dict):
    from multimodal_rag.retrieval.page_image import load_encoder

    base = tomllib.loads((ROOT / config["base"]).read_text(encoding="utf-8"))
    return base, load_encoder(base)


def mine(config: dict) -> None:
    from multimodal_rag.retrieval.page_image import PageImageRetriever

    pages, queries = load_jsonl(PAGES), load_jsonl(QUERIES)
    base, encoder = load_base(config)
    retriever = PageImageRetriever(encoder, ROOT, batch_size=base["batch_size"], chunk_pages=base["chunk_pages"])
    cache = ROOT / "outputs" / "cache" / "finetune" / f"base_index_{file_sha256(PAGES)[:16]}.pt"
    if cache.exists():
        retriever.load_index(cache)
    else:
        retriever.index(pages)
        cache.parent.mkdir(parents=True, exist_ok=True)
        retriever.save_index(cache)
    if retriever.page_ids != [p["page_id"] for p in pages]:
        raise SystemExit(f"{cache} does not match {PAGES}; delete it")

    rows, n_without = [], 0
    for i, q in enumerate(queries, 1):
        ranked = [(h.page_id, h.score) for h in retriever.search(q["query"], k=len(pages))]
        negative = pick_hard_negative(ranked, q["page_id"], config["negatives"]["max_ratio"])
        n_without += negative is None
        rank = next(r for r, (p, _) in enumerate(ranked, 1) if p == q["page_id"])
        rows.append(q | {"negative": negative, "base_rank": rank})
        if i % 200 == 0:
            print(f"[{i}/{len(queries)}]")
    with TRIPLETS.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ranks = [r["base_rank"] for r in rows]
    print(f"{len(rows)} queries; {n_without} without a hard negative (dropped in training); "
          f"base model ranks the positive 1st for {sum(r == 1 for r in ranks)}, "
          f"MRR {sum(1 / r for r in ranks) / len(ranks):.3f} over all training pages")


def train(config: dict) -> None:
    import torch
    from datasets import Dataset, Image
    from sentence_transformers.base.sampler import BatchSamplers
    from sentence_transformers.multi_vector_encoder import (
        MultiVectorEncoderTrainer,
        MultiVectorEncoderTrainingArguments,
    )
    from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorInformationRetrievalEvaluator
    from sentence_transformers.multi_vector_encoder.losses import CachedMultiVectorMultipleNegativesRankingLoss

    from multimodal_rag.retrieval.page_image import open_page_image

    cfg = config["train"]
    pages = {p["page_id"]: p for p in load_jsonl(PAGES)}
    triplets = [t for t in load_jsonl(TRIPLETS) if t["negative"] is not None]
    val = val_companies(config, list(pages.values()))
    train_rows, val_rows = split_by_company(triplets, val)
    print(f"validation companies {sorted(val)}: {len(train_rows)} training triplets, {len(val_rows)} validation queries")

    def image_path(page_id: str) -> str:
        return str(ROOT / pages[page_id]["image"])

    train_ds = Dataset.from_dict({
        "query": [t["query"] for t in train_rows],
        "positive": [image_path(t["page_id"]) for t in train_rows],
        "negative": [image_path(t["negative"]) for t in train_rows],
    }).cast_column("positive", Image()).cast_column("negative", Image())  # decoded by PIL, not torchvision

    val_pages = [p for p in pages if company_of(p) in val]
    evaluator = MultiVectorInformationRetrievalEvaluator(
        queries={str(i): t["query"] for i, t in enumerate(val_rows)},
        corpus={p: open_page_image(ROOT / pages[p]["image"]) for p in val_pages},
        relevant_docs={str(i): {t["page_id"]} for i, t in enumerate(val_rows)},
        name="val", batch_size=2,
    )

    base, model = load_base(config)
    for name, param in model.named_parameters():
        param.requires_grad = cfg["trainable"] == "lora" and "lora_" in name
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_trainable:,}")

    started = datetime.now()
    stamp = started.strftime("%Y%m%d-%H%M%S")
    model_dir = ROOT / "outputs" / "models" / "colqwen_ft" / stamp
    args = MultiVectorEncoderTrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        num_train_epochs=cfg["epochs"], per_device_train_batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"], warmup_ratio=cfg["warmup_ratio"], seed=cfg["seed"],
        bf16=True, gradient_checkpointing=True, batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps", eval_steps=cfg["eval_steps"], save_strategy="steps", save_steps=cfg["eval_steps"],
        save_total_limit=2, save_only_model=True, load_best_model_at_end=True,
        metric_for_best_model=cfg["metric_for_best_model"], logging_steps=5, report_to="none",
    )
    loss = CachedMultiVectorMultipleNegativesRankingLoss(model, mini_batch_size=cfg["mini_batch_size"])
    before = evaluator(model)
    print("validation before training:", {k: round(v, 4) for k, v in before.items() if "mrr" in k or "ndcg@10" in k})
    trainer = MultiVectorEncoderTrainer(model=model, args=args, train_dataset=train_ds, loss=loss, evaluator=evaluator)
    trainer.train()
    after = evaluator(model)
    print("validation after training (best checkpoint):", {k: round(v, 4) for k, v in after.items() if "mrr" in k or "ndcg@10" in k})

    final = model_dir / "final"
    model.save(str(final))
    files = sorted(p for p in final.rglob("*") if p.is_file())
    digest = hashlib.sha256(b"".join(p.relative_to(final).as_posix().encode() + bytes.fromhex(file_sha256(p))
                                     for p in files)).hexdigest()
    out = ROOT / "results" / "finetune" / stamp
    out.mkdir(parents=True, exist_ok=False)
    meta = {
        "config": config, "base_config": base, "timestamp": started.isoformat(timespec="seconds"),
        "data": {name: {"path": path.relative_to(ROOT).as_posix(), "sha256": file_sha256(path)}
                 for name, path in (("train_sources", SOURCES), ("pages", PAGES), ("queries", QUERIES), ("triplets", TRIPLETS))},
        "val_companies": sorted(val), "n_train_triplets": len(train_rows), "n_val_queries": len(val_rows),
        "n_val_pages": len(val_pages), "trainable_parameters": n_trainable,
        "validation_before": before, "validation_after": after, "log_history": trainer.state.log_history,
        "model_dir": final.relative_to(ROOT).as_posix(), "model_sha256": digest,
        "git": git_state(),
        "versions": {pkg: version(pkg) for pkg in ("torch", "sentence-transformers", "transformers", "peft")}
        | {"python": platform.python_version()},
    }
    (out / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str) + "\n",
                                  encoding="utf-8", newline="\n")
    print(f"model saved to {final.relative_to(ROOT)} (sha256 {digest[:16]}); metadata in {out.relative_to(ROOT)}")
    del model
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["mine", "train"])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    check_no_eval_companies()
    {"mine": mine, "train": train}[args.step](config)


if __name__ == "__main__":
    main()
