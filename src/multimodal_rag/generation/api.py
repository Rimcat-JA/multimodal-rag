"""Shared plumbing for hosted LLMs via OpenRouter: key, provider pinning, retries, cache.

Both the generator (Gemini) and the grader (DeepSeek) are called through
OpenRouter's OpenAI-compatible chat completions API. OpenRouter normally picks
a provider per request and falls back to others; here every request is pinned
to the model maker's own provider, fallbacks are off, and only providers that
support every parameter sent may serve it, so a setting can't be silently
dropped. The provider that answered is recorded from the reply.

Every reply is written to outputs/cache/ together with the input that produced
it, and a cached reply is never requested again (CLAUDE.md). For models that
ignore temperature, this cache is what makes a run reproducible.
"""

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRY_CODES = {429, 500, 502, 503, 504}  # rate limits and temporary server errors


def api_key(name: str = "OPENROUTER_API_KEY") -> str:
    """Read an API key from .env (or the environment). The value is never printed."""
    load_dotenv(ROOT / ".env")
    key = os.environ.get(name)
    if not key:
        raise SystemExit(f"{name} is not set: add it to .env in the project root")
    return key


def provider_preferences(provider: str) -> dict:
    """Only this provider, no fallback, and it must support every parameter in the request."""
    return {"order": [provider], "allow_fallbacks": False, "require_parameters": True}


def post_json(url: str, body: dict, headers: dict, timeout: float = 300) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def post_json_with_retry(url: str, body: dict, headers: dict, post: Callable = post_json,
                         sleep: Callable[[float], None] = time.sleep, attempts: int = 5) -> dict:
    """POST, retrying rate limits and temporary server errors with exponential backoff.
    Other HTTP errors (e.g. 400 for a malformed request) raise at once."""
    for i in range(attempts):
        try:
            return post(url, body, headers)
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_CODES or i == attempts - 1:
                raise
            sleep(2 ** i * 5)
    raise AssertionError("unreachable")


def chat(body: dict, key: str) -> dict:
    return post_json_with_retry(OPENROUTER_URL, body, {"Authorization": f"Bearer {key}"})


def message_content(reply: dict) -> str:
    """The final text of a chat completion (reasoning is a separate field). Raises ValueError
    if the reply was cut off or carries an error."""
    if "error" in reply:
        raise ValueError(f"error in reply: {reply['error']}")
    choices = reply.get("choices") or []
    if not choices:
        raise ValueError("reply has no choices")
    finish = choices[0].get("finish_reason")
    if finish != "stop":
        raise ValueError(f"generation stopped with {finish} ({choices[0].get('native_finish_reason')})")
    return choices[0]["message"].get("content") or ""


def call_with_cache(path: Path, request_input: dict, call: Callable[[], dict]) -> dict:
    """Return {"input", "response"} from the cache file, calling the API only if it doesn't exist.
    Failed calls raise and are not cached, so the next run tries again."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    record = {"input": request_input, "response": call()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    return record
