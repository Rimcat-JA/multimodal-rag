import io
import json
import urllib.error

import pytest

from multimodal_rag.generation.answer import (
    ANSWER_FIELDS,
    answer_cache_path,
    answer_request_body,
    parse_answer,
    settings_key,
)
from multimodal_rag.generation.api import call_with_cache, post_json_with_retry
from multimodal_rag.generation.grade import judge_request_body, parse_judgement, units_match, values_match

GEN = {"model": "google/gemini-x", "provider": "google-ai-studio", "reasoning_effort": "medium",
       "max_tokens": 8192, "prompt": "answer from the pages"}
JUDGE = {"model": "deepseek/x", "provider": "deepseek", "temperature": 0, "reasoning_effort": "none",
         "max_tokens": 512, "prompt": "judge"}
PINNED = {"order": ["google-ai-studio"], "allow_fallbacks": False, "require_parameters": True}


# ---- numeric grading: same number as printed, same unit ----

@pytest.mark.parametrize("pred, gold", [
    ("37662", "37662"),
    ("37,662", "37662"),      # commas are formatting
    ("４.０", "4.0"),          # full-width digits
    ("4", "4.0"),             # same number: 4 == 4.0
    ("△2.6", "-2.6"),         # IR decks print negatives with △
    ("-2.60", "-2.6"),
])
def test_values_match(pred, gold):
    assert values_match(pred, gold)


@pytest.mark.parametrize("pred, gold", [
    ("15", "15.2"),           # rounded: not the printed value (eval/README)
    ("2.6", "-2.6"),          # wrong sign
    ("", "4.0"),              # no number given
    ("約15", "15"),           # not a plain number
    ("3.7662", "37662"),      # a different number, e.g. another unit
])
def test_values_do_not_match(pred, gold):
    assert not values_match(pred, gold)


def test_units_match_accepts_other_spellings_of_the_same_unit():
    assert units_match("ポイント", "pt")
    assert units_match("millions of dollars", "百万ドル")
    assert units_match("ドル／バレル", "US$/BBL")  # full-width slash is normalized first
    assert not units_match("百万円", "百万ドル")
    assert not units_match("億円", "兆円")       # a different scale is never the same unit


def test_units_match_ignores_width_case_and_spaces():
    assert units_match("億円", "億円")
    assert units_match("US$/bbl", "US$/BBL")
    assert units_match("％", "%")
    assert units_match(" pt ", "pt")
    assert not units_match("百万円", "億円")
    assert not units_match("", "億円")


# ---- generation request and reply ----

def test_answer_request_has_question_and_every_image_in_order():
    body = answer_request_body(GEN, "Q?", ["aW1nMQ==", "aW1nMg=="])
    content = body["messages"][0]["content"]
    assert content[0]["text"].startswith("answer from the pages") and "Q?" in content[0]["text"]
    urls = [c["image_url"]["url"] for c in content if c["type"] == "image_url"]
    assert urls == ["data:image/png;base64,aW1nMQ==", "data:image/png;base64,aW1nMg=="]
    # each image is preceded by a label so the answer can refer to it
    assert [c["text"] for c in content[1:] if c["type"] == "text"] == ["ページ1", "ページ2"]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["reasoning"] == {"effort": "medium"}
    assert body["provider"] == PINNED
    assert "temperature" not in body  # Gemini 3.x ignores it; sending it would only mislead


def chat_reply(text, finish="stop"):
    return {"choices": [{"message": {"content": text, "reasoning": "thinking..."}, "finish_reason": finish}],
            "provider": "Google AI Studio", "model": "google/gemini-x", "usage": {"prompt_tokens": 10}}


def test_parse_answer_reads_content_not_reasoning():
    reply = chat_reply(json.dumps({"answer": "37,662億円です", "value": "37662", "unit": "億円"}))
    assert parse_answer(reply) == {"answer": "37,662億円です", "value": "37662", "unit": "億円"}


@pytest.mark.parametrize("reply", [
    chat_reply("not json"),
    chat_reply(json.dumps({"answer": "x"})),                         # missing fields
    chat_reply(json.dumps({"answer": "x", "value": "1", "unit": "%"}), finish="length"),
    {"choices": []},                                                 # empty
    {"error": {"message": "provider error"}},
])
def test_parse_answer_rejects_bad_replies(reply):
    with pytest.raises(ValueError):
        parse_answer(reply)


def test_settings_key_tracks_prompt_model_and_options():
    base = settings_key(GEN)
    for change in ({"prompt": "other"}, {"model": "y"}, {"reasoning_effort": "low"}, {"provider": "google-vertex"}):
        assert settings_key(GEN | change) != base


def test_answer_cache_path_depends_on_question_and_pages(tmp_path):
    a = answer_cache_path(tmp_path, GEN, "Q?", ["p1", "p2"])
    assert a == answer_cache_path(tmp_path, GEN, "Q?", ["p1", "p2"])
    assert a != answer_cache_path(tmp_path, GEN, "Q?", ["p2", "p1"])  # order is part of the input
    assert a != answer_cache_path(tmp_path, GEN, "Q2?", ["p1", "p2"])
    assert ANSWER_FIELDS == ["answer", "value", "unit"]


# ---- text grading request and reply ----

def test_judge_request_is_deterministic_json_without_thinking():
    body = judge_request_body(JUDGE, question="Q", reference="R", candidate="C")
    assert body["temperature"] == 0 and body["reasoning"] == {"effort": "none"}
    assert body["response_format"] == {"type": "json_object"}
    assert body["provider"] == PINNED | {"order": ["deepseek"]}
    user = body["messages"][-1]["content"]
    assert "Q" in user and "R" in user and "C" in user


def test_parse_judgement():
    reply = {"choices": [{"message": {"content": '{"correct": true, "reason": "same"}'}, "finish_reason": "stop"}]}
    assert parse_judgement(reply) == {"correct": True, "reason": "same"}
    bad = {"choices": [{"message": {"content": '{"correct": "yes"}'}, "finish_reason": "stop"}]}
    with pytest.raises(ValueError):
        parse_judgement(bad)


# ---- caching and retries ----

def test_call_with_cache_calls_once(tmp_path):
    calls = []

    def call():
        calls.append(1)
        return {"x": 1}

    path = tmp_path / "a" / "b.json"
    assert call_with_cache(path, {"q": "in"}, call)["response"] == {"x": 1}
    assert call_with_cache(path, {"q": "in"}, call)["response"] == {"x": 1}
    assert len(calls) == 1
    assert json.loads(path.read_text(encoding="utf-8"))["input"] == {"q": "in"}


def http_error(code):
    return urllib.error.HTTPError("u", code, "err", {}, io.BytesIO(b"{}"))


def test_retry_on_rate_limit_then_succeed():
    attempts = []

    def flaky(url, body, headers):
        attempts.append(1)
        if len(attempts) < 3:
            raise http_error(429)
        return {"ok": True}

    assert post_json_with_retry("u", {}, {}, post=flaky, sleep=lambda s: None) == {"ok": True}
    assert len(attempts) == 3


def test_no_retry_on_bad_request():
    def bad(url, body, headers):
        raise http_error(400)

    with pytest.raises(urllib.error.HTTPError):
        post_json_with_retry("u", {}, {}, post=bad, sleep=lambda s: None)
