"""Answer a question from page images with Gemini (through OpenRouter).

The generator sees only the question and the page images, labelled ページ1..N
in the order given (retrieval rank). Page ids are not shown: they name the
company and deck, which would be a hint the retrieval didn't earn.

Gemini 3.x ignores temperature, so a reply can't be reproduced by calling
again; each reply is cached under a hash of the settings, question and pages.
"""

import hashlib
import json
from pathlib import Path

from multimodal_rag.generation.api import message_content, provider_preferences

ANSWER_FIELDS = ["answer", "value", "unit"]
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "value": {"type": "string"},
        "unit": {"type": "string"},
    },
    "required": ANSWER_FIELDS,
    "additionalProperties": False,  # required by strict structured outputs
}
SETTING_KEYS = ("model", "provider", "reasoning_effort", "max_tokens", "prompt")


def settings_key(config: dict) -> str:
    fields = {k: config[k] for k in SETTING_KEYS} | {"schema": ANSWER_SCHEMA}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def answer_cache_path(cache_root: Path, config: dict, question: str, page_ids: list[str]) -> Path:
    """One file per (settings, question, pages in order)."""
    key = hashlib.sha256(json.dumps([question, page_ids], ensure_ascii=False).encode()).hexdigest()[:16]
    return cache_root / settings_key(config) / f"{key}.json"


def answer_request_body(config: dict, question: str, images_b64: list[str]) -> dict:
    content = [{"type": "text", "text": f"{config['prompt'].strip()}\n\n質問：{question}"}]
    for i, image in enumerate(images_b64, start=1):
        content += [{"type": "text", "text": f"ページ{i}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}]
    return {
        "model": config["model"],
        "messages": [{"role": "user", "content": content}],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "answer", "strict": True, "schema": ANSWER_SCHEMA}},
        "reasoning": {"effort": config["reasoning_effort"]},  # mapped to Gemini's thinkingLevel
        "max_tokens": config["max_tokens"],
        "provider": provider_preferences(config["provider"]),
    }


def parse_answer(reply: dict) -> dict:
    """{"answer", "value", "unit"} from a chat completion. Raises ValueError otherwise."""
    text = message_content(reply)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"reply is not valid JSON: {e}") from e
    if not isinstance(data, dict) or any(not isinstance(data.get(k), str) for k in ANSWER_FIELDS):
        raise ValueError(f"reply is missing a string field: {data!r}")
    return {k: data[k] for k in ANSWER_FIELDS}
