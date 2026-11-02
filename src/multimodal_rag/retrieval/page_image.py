"""Method C: embed each page image with ColQwen2.5 and rank pages by late interaction.

Each page becomes several hundred 128-dim unit vectors (about one per image
patch, plus a few prompt tokens); a query becomes one vector per token. A
page's score is MaxSim: for every query token take its best match among the
page's vectors, then sum over query tokens. No text extraction is involved,
so charts and tables are seen the way they are drawn.

As in method A, the encoder is passed in so tests can use a small fake.
"""

from pathlib import Path
from typing import Protocol

import hashlib

import torch
from PIL import Image
from sentence_transformers.util import maxsim

from multimodal_rag.retrieval.base import PageHit, top_k_hits


class MultiVectorEncoderLike(Protocol):
    """The parts of sentence_transformers.MultiVectorEncoder this module uses."""

    device: torch.device

    def encode_document(self, inputs: list, **kwargs) -> list[torch.Tensor]: ...

    def encode_query(self, inputs: list[str], **kwargs) -> list[torch.Tensor]: ...


def model_dir_sha256(model_dir: Path) -> str:
    """Fingerprint of a saved model directory: every file's relative path and contents."""
    files = sorted(p for p in model_dir.rglob("*") if p.is_file())
    return hashlib.sha256(b"".join(p.relative_to(model_dir).as_posix().encode()
                                   + hashlib.sha256(p.read_bytes()).digest() for p in files)).hexdigest()


def load_encoder(config: dict, root: Path | None = None):
    """Load the MultiVectorEncoder described by configs/method_c.toml onto the GPU.

    A Hugging Face model is pinned by `revision`. A fine-tuned model saved on disk
    (configs/method_c_ft.toml) is given by `model_path` under `root` and checked
    against `model_sha256`, so a retrained model is never evaluated by mistake.
    """
    from sentence_transformers import MultiVectorEncoder

    if "model_path" in config:
        model_dir = root / config["model_path"]
        digest = model_dir_sha256(model_dir)
        if digest != config["model_sha256"]:
            raise ValueError(f"{model_dir} has sha256 {digest}, not {config['model_sha256']}")
        source, kwargs = str(model_dir), {}
    else:
        source, kwargs = config["model"], {"revision": config["revision"]}
    return MultiVectorEncoder(
        source,
        device="cuda",  # ROCm builds of torch also use the "cuda" device name
        model_kwargs={"torch_dtype": getattr(torch, config["torch_dtype"])},
        **kwargs,
    )


def open_page_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")  # convert() loads the pixels, so the file can be closed


class PageImageRetriever:
    def __init__(self, encoder: MultiVectorEncoderLike, root: Path, batch_size: int = 2, chunk_pages: int = 16):
        self.encoder = encoder
        self.root = root  # page["image"] paths are relative to the repo root
        self.batch_size = batch_size
        self.chunk_pages = chunk_pages
        self.page_ids: list[str] = []
        self.page_embeddings: list[torch.Tensor] | None = None

    def index(self, pages: list[dict]) -> None:
        self.page_ids = [p["page_id"] for p in pages]
        embeddings: list[torch.Tensor] = []
        # Open only a few images at a time: all 1,429 decoded pages at
        # 2000x1125 RGB would take about 10 GB of RAM.
        for start in range(0, len(pages), self.chunk_pages):
            images = [open_page_image(self.root / p["image"]) for p in pages[start:start + self.chunk_pages]]
            embeddings += self.encoder.encode_document(images, batch_size=self.batch_size)
            print(f"  encoded {min(start + self.chunk_pages, len(pages))}/{len(pages)} pages", flush=True)
        self.page_embeddings = embeddings

    def save_index(self, path: Path) -> None:
        torch.save({"page_ids": self.page_ids, "embeddings": [e.cpu() for e in self.page_embeddings]}, path)

    def load_index(self, path: Path) -> None:
        """Reuse embeddings saved by save_index instead of re-encoding every page image."""
        saved = torch.load(path, weights_only=True)
        if len(saved["page_ids"]) != len(saved["embeddings"]):
            raise ValueError("page_ids and embeddings differ in length")
        self.page_ids = saved["page_ids"]
        self.page_embeddings = [e.to(self.encoder.device) for e in saved["embeddings"]]

    def search(self, query: str, k: int) -> list[PageHit]:
        if self.page_embeddings is None:
            raise RuntimeError("call index() or load_index() before search()")
        query_embedding = self.encoder.encode_query([query])
        # maxsim returns float32 scores even for bf16 inputs, so nearby pages don't tie by rounding.
        scores = maxsim(query_embedding, self.page_embeddings)[0]
        return top_k_hits(scores.cpu().numpy(), self.page_ids, k)
