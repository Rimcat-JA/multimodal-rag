"""Write synthetic search queries for training pages with Gemini (through OpenRouter).

Fine-tuning ColQwen2.5 needs (query, page image) pairs. For every page of the
training decks (companies not in the evaluation set), Gemini sees the page
image, the company name and the period, and writes up to N queries that this
page answers. Pages with nothing specific to ask (covers, agendas, disclaimers,
section dividers) get no queries.

Claude wrote the evaluation questions, so a different model writes the training
queries: otherwise the training queries could copy the evaluation set's style.
Replies are cached like the answer replies (generation/api.py).
"""

import hashlib
import json
from pathlib import Path

from multimodal_rag.generation.api import message_content, provider_preferences

QUERY_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
    "additionalProperties": False,
}
SETTING_KEYS = ("model", "provider", "reasoning_effort", "max_tokens", "prompt", "queries_per_page")


def settings_key(config: dict) -> str:
    fields = {k: config[k] for k in SETTING_KEYS} | {"schema": QUERY_SCHEMA}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def query_cache_path(cache_root: Path, config: dict, page_id: str) -> Path:
    return cache_root / settings_key(config) / f"{page_id}.json"  # page ids contain "/", so this nests


def query_request_body(config: dict, company: str, period: str, image_b64: str) -> dict:
    prompt = config["prompt"].strip().format(company=company, period=period, n=config["queries_per_page"])
    return {
        "model": config["model"],
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ]}],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "queries", "strict": True, "schema": QUERY_SCHEMA}},
        "reasoning": {"effort": config["reasoning_effort"]},
        "max_tokens": config["max_tokens"],
        "provider": provider_preferences(config["provider"]),
    }


def parse_queries(reply: dict, limit: int) -> list[str]:
    """The non-empty, distinct queries in a reply, at most `limit`. Raises ValueError otherwise."""
    try:
        data = json.loads(message_content(reply))
    except json.JSONDecodeError as e:
        raise ValueError(f"reply is not valid JSON: {e}") from e
    queries = data.get("queries") if isinstance(data, dict) else None
    if not isinstance(queries, list) or any(not isinstance(q, str) for q in queries):
        raise ValueError(f"reply has no list of strings: {data!r}")
    return list(dict.fromkeys(q.strip() for q in queries if q.strip()))[:limit]
