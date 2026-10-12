# multimodal-rag

Multimodal RAG over financial-report charts and tables. The goal is to compare three retrieval approaches on a hand-built evaluation set:

- **A. Text only**: PyMuPDF text extraction + text embeddings (baseline)
- **B. Figure captioning**: a vision LLM reads each whole page image and returns one structured description per chart or table (type, title, unit, Markdown table, summary), which is then embedded (B1: Claude API, B2: local Qwen2.5-VL)
- **C. Page-image retrieval**: ColQwen2.5 / ColPali multi-vector embeddings with late interaction (MaxSim)

> Status: data collected and pages rendered (week 1). 15 companies across 5 industries, 30 decks, 1,429 pages. Sources are listed in [docs/sources.csv](docs/sources.csv). The PDFs themselves are not redistributed.

## Design decisions

**Caption whole pages instead of cropping figures (method B).** I first tried locating figures from the PDF structure: clustered vector drawings, embedded images and detected tables, merged into regions. On a sample of six slides from four industries, the crops regularly left out what a captioner needs most: units, legends, axis labels and the slide title. Some figures that fill the whole slide were split into fragments, and some tables lost their middle rows. IR slides are designed so that one page carries one message, so cropping removes context instead of focusing it. Method B therefore sends the full page image and asks for one structured description per figure. That keeps figure-level retrieval units without the cropping step.

## Environment

| | |
|---|---|
| OS | Windows 11 |
| GPU | AMD Radeon RX 9070 XT (gfx1201, 16 GB) |
| Python | 3.12 (system install via winget) |
| PyTorch | 2.14.0 + ROCm 10.1.0 (AMD's Windows wheels) |
| Local VLM | Ollama + `qwen2.5vl:7b` |

## Setup

```powershell
winget install --id Python.Python.3.12 -e
winget install --id astral-sh.uv -e
uv sync
uv run python scripts/check_gpu.py   # expects "GPU available : True" and "OK"
```

```powershell
winget install --id Ollama.Ollama -e
ollama pull qwen2.5vl:7b
```

## Setup notes

Problems hit while setting up ROCm PyTorch on Windows, and how they were solved:

1. **ROCm build of torch from AMD's index.** `pyproject.toml` lists AMD's index before PyPI, so `torch` and its `rocm-sdk-*` / `amd-torch-device-*` dependencies always come from AMD. This stops libraries added later from pulling in the CUDA build.
2. **`triton` breaks `uv lock`.** AMD's Windows wheel doesn't depend on `triton`, but the resolver reads the Linux wheel's metadata, which pins a Linux-only `triton`. `exclude-dependencies = ["triton"]` drops it.
3. **`uv.toml` overrides `[tool.uv]`.** When both exist, uv ignores the settings in `pyproject.toml`, so all uv settings live in `pyproject.toml`.
4. **AppData redirection.** Processes launched from a packaged (MSIX) app have their AppData writes redirected to a private location. A uv-managed Python installed that way can't be seen from a normal terminal. The fix is to use the winget Python (`python-preference = "only-system"`) and to keep the uv cache outside AppData.
