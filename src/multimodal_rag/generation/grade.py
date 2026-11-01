"""Grade answers against eval/questions.csv.

Numeric questions (answer_value set) are graded by rule, as eval/README.md
defines: the number must equal the printed value (15 is wrong when 15.2 is
printed) and the unit must be the same. Other spellings of the same unit
count as the same (UNIT_ALIASES, e.g. ポイント for pt), but there is no unit
conversion: "0.4兆円" for a gold "4000億円" counts as wrong, and such cases are
reported separately.

Text questions are graded by DeepSeek (through OpenRouter), which
compares the answer with the reference answer. Neither Claude (who wrote the
questions) nor Gemini (who answered) grades.
"""

import json
import unicodedata
from decimal import Decimal, InvalidOperation

from multimodal_rag.evalset import NUMBER_RE, normalize_number
from multimodal_rag.generation.api import message_content, provider_preferences


def values_match(predicted: str, gold: str) -> bool:
    """Same number as printed. Formatting (commas, full-width, △) is ignored; 4 == 4.0."""
    p = normalize_number(predicted)
    if not NUMBER_RE.match(p):
        return False
    try:
        return Decimal(p) == Decimal(gold)
    except InvalidOperation:
        return False


# Other spellings of the units that appear in eval/questions.csv, written before
# the full answer run (2026-11-01) and decided by the developer. Only the same unit
# spelled differently; never a different scale (百万円 is not 億円).
UNIT_ALIASES = {
    "pt": ["ポイント", "%ポイント", "point", "points", "pp"],
    "%": ["パーセント", "percent"],
    "百万ドル": ["百万米ドル", "百万us$", "百万usドル", "millionsofdollars", "milliondollars", "millionusd",
               "usdmillion", "us$million", "$million"],
    "us$/bbl": ["ドル/バレル", "us$/バレル", "usd/bbl", "$/bbl", "usドル/バレル", "米ドル/バレル"],
    "倍": ["x", "times"],
    "円": ["yen"],
}


def normalize_unit(unit: str) -> str:
    """Width, case and spaces removed, then mapped to the canonical spelling."""
    u = unicodedata.normalize("NFKC", unit).strip().replace(" ", "").lower()
    for canonical, aliases in UNIT_ALIASES.items():
        if u in aliases:
            return canonical
    return u


def units_match(predicted: str, gold: str) -> bool:
    return bool(predicted.strip()) and normalize_unit(predicted) == normalize_unit(gold)


def judge_request_body(config: dict, question: str, reference: str, candidate: str) -> dict:
    user = f"質問：{question}\n\n正解：{reference}\n\n回答：{candidate}"
    return {
        "model": config["model"],
        "messages": [{"role": "system", "content": config["prompt"].strip()}, {"role": "user", "content": user}],
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
        # Thinking off: DeepSeek ignores temperature in thinking mode.
        "reasoning": {"effort": config["reasoning_effort"]},
        "response_format": {"type": "json_object"},
        "provider": provider_preferences(config["provider"]),
    }


def parse_judgement(reply: dict) -> dict:
    """{"correct": bool, "reason": str} from a chat completion. Raises ValueError otherwise."""
    try:
        data = json.loads(message_content(reply))
    except json.JSONDecodeError as e:
        raise ValueError(f"judge reply is not valid JSON: {e}") from e
    if not isinstance(data.get("correct"), bool):
        raise ValueError(f"judge reply has no boolean 'correct': {data!r}")
    return {"correct": data["correct"], "reason": str(data.get("reason", ""))}
