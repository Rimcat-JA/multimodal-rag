"""Build the method comparison (tables + charts) from saved runs.

Run with:  uv run python scripts/compare_methods.py

For each method it takes the latest run in results/method_<x>/ that was made
from committed code (git.dirty is false), refuses to compare runs made on
different versions of the evaluation set or page list, and writes:
    results/comparison.md                 tables (also the text version of every chart)
    results/figures/<chart>_<theme>.png   charts in light and dark versions
Everything is computed from run.json / per_question.csv; nothing is re-run.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt

from multimodal_rag.eval.compare import answerable_within, paired_bootstrap, reciprocal_rank
from multimodal_rag.evalset import load_questions

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

# Fixed order and colour per method: a method keeps its colour even when another
# method has no run yet. Colours are slots 1-3 of the dataviz reference palette,
# validated for colour-vision deficiency in both themes.
METHODS = ["A", "B2", "C"]
LABELS = {
    "A": "A: page text (BGE-M3)",
    "B2": "B2: VLM figure captions (Qwen2.5-VL)",
    "C": "C: page images (ColQwen2.5)",
}
THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
              "series": {"A": "#2a78d6", "B2": "#eb6834", "C": "#1baf7a"}},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "muted": "#c3c2b7", "grid": "#3a3936",
             "series": {"A": "#3987e5", "B2": "#d95926", "C": "#199e70"}},
}
KS = (1, 3, 5)
METRICS = [f"recall@{k}" for k in KS] + ["mrr"]
METRIC_LABELS = {"recall@1": "Recall@1", "recall@3": "Recall@3", "recall@5": "Recall@5", "mrr": "MRR"}


def latest_clean_run(method: str) -> Path | None:
    runs = sorted((RESULTS / f"method_{method.lower()}").glob("*/run.json"))
    clean = [r.parent for r in runs if not json.loads(r.read_text(encoding="utf-8"))["git"]["dirty"]]
    return clean[-1] if clean else None


def load_ranks(run_dir: Path) -> dict[str, int | None]:
    with (run_dir / "per_question.csv").open(encoding="utf-8") as f:
        return {row["qid"]: int(row["rank"]) if row["rank"] else None for row in csv.DictReader(f)}


def scores(ranks: list[int | None]) -> dict[str, float]:
    out = {f"recall@{k}": sum(answerable_within(r, k) for r in ranks) / len(ranks) for k in KS}
    out["mrr"] = sum(reciprocal_rank(r) for r in ranks) / len(ranks)
    return out


def style_axes(ax, theme: dict) -> None:
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["muted"])
    ax.tick_params(colors=theme["muted"], length=0)
    ax.yaxis.grid(True, color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def grouped_bars(ax, groups: list[str], values: dict[str, list[float]], theme: dict) -> None:
    methods = list(values)
    width = 0.8 / len(methods)
    for i, m in enumerate(methods):
        xs = [g + (i - (len(methods) - 1) / 2) * width for g in range(len(groups))]
        # White edge = a small gap between neighbouring bars.
        ax.bar(xs, values[m], width, color=theme["series"][m], edgecolor=theme["surface"], linewidth=2, label=LABELS[m])
    ax.set_xticks(range(len(groups)), groups, color=theme["text"])
    ax.set_ylim(0, 1)


def save(fig, name: str, theme_name: str, theme: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(theme["surface"])
    fig.savefig(FIGURES / f"{name}_{theme_name}.png", dpi=150, bbox_inches="tight", facecolor=theme["surface"])
    plt.close(fig)


def legend(ax, theme: dict, **kwargs) -> None:
    leg = ax.legend(frameon=False, labelcolor=theme["text"], **kwargs)
    for text in leg.get_texts():
        text.set_color(theme["text"])


def main() -> None:
    runs = {m: d for m in METHODS if (d := latest_clean_run(m))}
    if "A" not in runs:
        raise SystemExit("no clean run of method A (the baseline) in results/")
    metas = {m: json.loads((d / "run.json").read_text(encoding="utf-8")) for m, d in runs.items()}
    for key in ("eval_set", "pages"):
        hashes = {m: meta[key]["sha256"] for m, meta in metas.items()}
        if len(set(hashes.values())) != 1:
            raise SystemExit(f"runs use different {key} versions, so they can't be compared: {hashes}")

    _, questions = load_questions(ROOT / "eval" / "questions.csv")
    qids = [q["qid"] for q in questions]
    ranks = {m: load_ranks(d) for m, d in runs.items()}
    for m, r in ranks.items():
        if sorted(r) != sorted(qids):
            raise SystemExit(f"method {m}'s run doesn't cover exactly the questions in eval/questions.csv")
    overall = {m: scores([ranks[m][q] for q in qids]) for m in runs}
    for m in runs:  # the recomputed numbers must agree with what the run itself reported
        for metric in METRICS:
            if abs(overall[m][metric] - metas[m]["summary"][metric]) > 1e-12:
                raise SystemExit(f"{m} {metric}: recomputed {overall[m][metric]} != run.json {metas[m]['summary'][metric]}")

    def subset(column: str) -> dict[str, dict[str, dict[str, float]]]:
        values = sorted({q[column] for q in questions})
        return {v: {m: scores([ranks[m][q["qid"]] for q in questions if q[column] == v]) for m in runs} | {
            "n": sum(q[column] == v for q in questions)} for v in values}

    by_evidence, by_author = subset("evidence"), subset("author")
    diffs = {}
    for m in runs:
        if m == "A":
            continue
        diffs[m] = {
            "mrr": paired_bootstrap([reciprocal_rank(ranks["A"][q]) for q in qids], [reciprocal_rank(ranks[m][q]) for q in qids]),
            "recall@5": paired_bootstrap([answerable_within(ranks["A"][q], 5) for q in qids],
                                         [answerable_within(ranks[m][q], 5) for q in qids]),
        }

    # ---- tables ----
    lines = [
        "# Method comparison",
        "",
        "Generated by `scripts/compare_methods.py` from the runs listed below; do not edit by hand.",
        f"All runs use the same evaluation set (`eval/questions.csv`, sha256 `{metas['A']['eval_set']['sha256'][:12]}`, "
        f"{len(qids)} questions) and page list ({metas['A']['pages']['n']} pages). Every page is ranked for every question.",
        "",
        "| Method | Run | Commit |",
        "|---|---|---|",
    ]
    lines += [f"| {LABELS[m]} | `{runs[m].relative_to(ROOT).as_posix()}` | `{metas[m]['git']['commit'][:7]}` |" for m in runs]
    lines += ["", "## Overall", "", "| Method | " + " | ".join(METRIC_LABELS[x] for x in METRICS) + " |",
              "|---|" + "---|" * len(METRICS)]
    lines += [f"| {LABELS[m]} | " + " | ".join(f"{overall[m][x]:.3f}" for x in METRICS) + " |" for m in runs]
    if diffs:
        lines += ["", "## Difference from A (paired bootstrap, 10,000 resamples of questions, seed 0)", "",
                  "| Method | MRR − A [95% interval] | Recall@5 − A [95% interval] |", "|---|---|---|"]
        lines += [f"| {LABELS[m]} | " + " | ".join(f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]" for d in (diffs[m]["mrr"], diffs[m]["recall@5"]))
                  + " |" for m in diffs]
    for title, table in (("By evidence type", by_evidence), ("By question author", by_author)):
        lines += ["", f"## {title}", "",
                  "Few questions per group: treat differences here as observations, not conclusions.", "",
                  "| Group | n | " + " | ".join(f"{m} R@5" for m in runs) + " | " + " | ".join(f"{m} MRR" for m in runs) + " |",
                  "|---|---|" + "---|" * (2 * len(runs))]
        lines += [f"| {g} | {row['n']} | " + " | ".join(f"{row[m]['recall@5']:.2f}" for m in runs) + " | "
                  + " | ".join(f"{row[m]['mrr']:.3f}" for m in runs) + " |" for g, row in table.items()]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    # ---- charts ----
    for theme_name, theme in THEMES.items():
        fig, ax = plt.subplots(figsize=(8, 4))
        style_axes(ax, theme)
        grouped_bars(ax, [METRIC_LABELS[x] for x in METRICS], {m: [overall[m][x] for x in METRICS] for m in runs}, theme)
        ax.set_title(f"Retrieval quality on {len(qids)} questions (higher is better)", color=theme["text"], loc="left")
        legend(ax, theme, loc="upper left", bbox_to_anchor=(0, -0.08), ncol=len(runs))
        save(fig, "overall", theme_name, theme)

        fig, ax = plt.subplots(figsize=(8, 4))
        style_axes(ax, theme)
        groups = list(by_evidence)
        grouped_bars(ax, [f"{g} (n={by_evidence[g]['n']})" for g in groups],
                     {m: [by_evidence[g][m]["mrr"] for g in groups] for m in runs}, theme)
        ax.set_title("MRR by where the answer is printed", color=theme["text"], loc="left")
        legend(ax, theme, loc="upper left", bbox_to_anchor=(0, -0.08), ncol=len(runs))
        save(fig, "mrr_by_evidence", theme_name, theme)

        if diffs:
            fig, ax = plt.subplots(figsize=(8, 0.8 + 0.7 * len(diffs)))
            style_axes(ax, theme)
            ax.yaxis.grid(False)
            ax.xaxis.grid(True, color=theme["grid"], linewidth=0.8)
            ax.axvline(0, color=theme["muted"], linewidth=1)
            for i, m in enumerate(diffs):
                mean, low, high = diffs[m]["mrr"]
                ax.plot([low, high], [i, i], color=theme["series"][m], linewidth=2)
                ax.plot([mean], [i], "o", markersize=9, color=theme["series"][m],
                        markeredgecolor=theme["surface"], markeredgewidth=2)
                ax.annotate(f"{mean:+.3f}  [{low:+.3f}, {high:+.3f}]", (high, i), xytext=(8, 0),
                            textcoords="offset points", va="center", color=theme["text"])
            ax.set_yticks(range(len(diffs)), [LABELS[m] for m in diffs], color=theme["text"])
            ax.set_ylim(-0.6, len(diffs) - 0.4)
            ax.invert_yaxis()
            ax.set_title("MRR difference from A, with 95% paired bootstrap interval", color=theme["text"], loc="left")
            save(fig, "mrr_diff_vs_a", theme_name, theme)

    print((RESULTS / "comparison.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
