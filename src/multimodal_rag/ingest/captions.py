"""Ask a local vision LLM (Ollama) to describe every figure on a page image (method B2).

The whole page image goes in; a JSON list of figures comes out, one entry per
chart, table or diagram (see README, Design decisions). Ollama's structured
output ("format": JSON schema) constrains the reply to that shape.

Each page's reply is cached as its own file under a directory named after a
hash of every setting that affects the reply (model digest, prompt, schema,
sampling options). A run can be stopped and resumed, and a changed prompt
never reuses old captions.
"""

import base64
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

FIGURE_TYPES = ["chart", "table", "diagram"]
FIGURE_FIELDS = ["type", "title", "unit", "table_markdown", "summary"]
FIGURE_SCHEMA = {
    "type": "object",
    "properties": {
        "figures": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": FIGURE_TYPES},
                    "title": {"type": "string"},
                    "unit": {"type": "string"},
                    "table_markdown": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": FIGURE_FIELDS,
            },
        }
    },
    "required": ["figures"],
}
OPTION_KEYS = ("temperature", "seed", "num_ctx", "num_predict")


def build_prompt(prompt: str) -> str:
    # Ollama's docs suggest also putting the schema in the prompt to ground the reply.
    return prompt.strip() + "\n\nJSON スキーマ：" + json.dumps(FIGURE_SCHEMA, ensure_ascii=False)


def settings_key(config: dict) -> str:
    """Hash of everything that changes the model's reply; names the cache directory."""
    fields = {k: config[k] for k in ("model", "model_digest", "prompt", *OPTION_KEYS)}
    fields["schema"] = FIGURE_SCHEMA
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def request_body(config: dict, image_b64: str) -> dict:
    return {
        "model": config["model"],
        "stream": False,
        "format": FIGURE_SCHEMA,
        "keep_alive": "10m",
        "options": {k: config[k] for k in OPTION_KEYS},
        "messages": [{"role": "user", "content": build_prompt(config["prompt"]), "images": [image_b64]}],
    }


def parse_figures(content: str) -> list[dict]:
    """The figures list from the model's reply. Raises ValueError if it isn't the expected shape."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"reply is not valid JSON: {e}") from e
    figures = data.get("figures") if isinstance(data, dict) else None
    if not isinstance(figures, list):
        raise ValueError("reply has no 'figures' list")
    for fig in figures:
        if not isinstance(fig, dict) or any(not isinstance(fig.get(k), str) for k in FIGURE_FIELDS):
            raise ValueError(f"figure is missing a string field: {fig!r}")
        if fig["type"] not in FIGURE_TYPES:
            raise ValueError(f"unknown figure type {fig['type']!r}")
    return [{k: fig[k] for k in FIGURE_FIELDS} for fig in figures]


def post_json(url: str, body: dict, timeout: float = 900) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def check_model_digest(config: dict) -> None:
    """Stop if the local model isn't the exact build named in the config."""
    with urllib.request.urlopen(f"{config['ollama_url']}/api/tags", timeout=30) as resp:
        models = {m["name"]: m["digest"] for m in json.load(resp)["models"]}
    digest = models.get(config["model"])
    if digest is None or not digest.startswith(config["model_digest"]):
        raise RuntimeError(f"{config['model']} digest is {digest}, config expects {config['model_digest']}")


def caption_page(config: dict, image_path: Path, post: Callable[[str, dict], dict] = post_json) -> dict:
    """Describe one page. A reply that can't be parsed, or that the server aborts (e.g.
    "token repeat limit reached" when the model loops), is kept with an error and no
    figures, not retried: with temperature 0 and a fixed seed, asking again would fail
    the same way. Connection errors still raise, so a stopped run can be resumed."""
    image_b64 = base64.b64encode(image_path.read_bytes()).decode()
    try:
        reply = post(f"{config['ollama_url']}/api/chat", request_body(config, image_b64))
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        return {"raw": "", "done_reason": None, "prompt_eval_count": None, "eval_count": None,
                "total_duration_s": 0, "figures": [], "error": f"HTTP {e.code}: {body}"}
    record = {
        "raw": reply["message"]["content"],
        "done_reason": reply.get("done_reason"),
        "prompt_eval_count": reply.get("prompt_eval_count"),
        "eval_count": reply.get("eval_count"),
        "total_duration_s": round(reply.get("total_duration", 0) / 1e9, 2),
    }
    try:
        record["figures"], record["error"] = parse_figures(record["raw"]), None
    except ValueError as e:
        record["figures"], record["error"] = [], str(e)
    return record


def cache_path(cache_dir: Path, page_id: str) -> Path:
    return cache_dir / f"{page_id}.json"  # page ids contain "/", so this nests by company and deck


def load_captions(cache_dir: Path, page_ids: list[str]) -> dict[str, list[dict]]:
    """page_id -> figures, for every page. Fails if any page hasn't been captioned yet."""
    missing = [p for p in page_ids if not cache_path(cache_dir, p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} pages have no captions in {cache_dir}, e.g. {missing[:3]}")
    return {p: json.loads(cache_path(cache_dir, p).read_text(encoding="utf-8"))["figures"] for p in page_ids}


def figure_text(fig: dict) -> str:
    """The text that gets embedded for one figure."""
    lines = [fig["title"]]
    if fig["unit"]:
        lines.append(f"単位：{fig['unit']}")
    lines.append(fig["summary"])
    if fig["table_markdown"]:
        lines.append(fig["table_markdown"])
    return "\n".join(lines)
