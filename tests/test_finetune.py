import json

import pytest

from multimodal_rag.finetune.data import pick_hard_negative, split_by_company
from multimodal_rag.finetune.queries import parse_queries, query_cache_path, query_request_body, settings_key

QCFG = {"model": "google/gemini-x", "provider": "google-ai-studio", "reasoning_effort": "low",
        "max_tokens": 2048, "prompt": "{company} の {period} の資料。{n} 個まで質問を書く", "queries_per_page": 2}


def test_hard_negative_is_the_best_other_page_of_the_same_company():
    # positive scored 10.0; acme/p2 (9.0) is the best other page of acme and below 0.95 * 10
    ranked = [("acme/d/p1", 10.0), ("other/d/p9", 9.8), ("acme/d/p2", 9.0), ("acme/d/p3", 8.0)]
    assert pick_hard_negative(ranked, positive="acme/d/p1", max_ratio=0.95) == "acme/d/p2"


def test_pages_scoring_almost_as_high_as_the_positive_are_skipped():
    # acme/p2 scores 9.7 >= 0.95 * 10: it may show the same fact (a false negative), so skip it
    ranked = [("acme/d/p1", 10.0), ("acme/d/p2", 9.7), ("acme/e/p5", 9.2)]
    assert pick_hard_negative(ranked, positive="acme/d/p1", max_ratio=0.95) == "acme/e/p5"


def test_no_hard_negative_when_no_page_of_the_company_qualifies():
    ranked = [("acme/d/p1", 10.0), ("other/d/p2", 5.0)]
    assert pick_hard_negative(ranked, positive="acme/d/p1", max_ratio=0.95) is None


def test_positive_missing_from_ranking_raises():
    with pytest.raises(ValueError):
        pick_hard_negative([("acme/d/p2", 1.0)], positive="acme/d/p1", max_ratio=0.95)


def test_split_by_company_keeps_each_company_on_one_side():
    rows = [{"page_id": "a/x/p1"}, {"page_id": "b/x/p1"}, {"page_id": "a/y/p2"}, {"page_id": "c/x/p3"}]
    train, val = split_by_company(rows, val_companies={"b", "c"})
    assert [r["page_id"] for r in train] == ["a/x/p1", "a/y/p2"]
    assert [r["page_id"] for r in val] == ["b/x/p1", "c/x/p3"]


def test_query_request_fills_company_period_and_count():
    body = query_request_body(QCFG, "三井物産", "2026年3月期 通期", "aW1n")
    text = body["messages"][0]["content"][0]["text"]
    assert text == "三井物産 の 2026年3月期 通期 の資料。2 個まで質問を書く"
    assert body["messages"][0]["content"][1]["image_url"]["url"] == "data:image/png;base64,aW1n"
    assert body["provider"]["allow_fallbacks"] is False


def test_parse_queries_dedupes_strips_and_limits():
    reply = {"choices": [{"message": {"content": json.dumps({"queries": [" q1 ", "q1", "", "q2", "q3"]})},
                          "finish_reason": "stop"}]}
    assert parse_queries(reply, limit=2) == ["q1", "q2"]
    empty = {"choices": [{"message": {"content": '{"queries": []}'}, "finish_reason": "stop"}]}
    assert parse_queries(empty, limit=2) == []


def test_query_cache_path_changes_with_settings(tmp_path):
    a = query_cache_path(tmp_path, QCFG, "acme/d/p001")
    assert a.name == "p001.json"
    assert settings_key(QCFG | {"queries_per_page": 3}) != settings_key(QCFG)


def test_overlapping_companies_matches_by_id_or_name():
    from multimodal_rag.finetune.data import overlapping_companies

    eval_sources = [{"company_id": "mitsubishielectric", "company": "三菱電機"}, {"company_id": "hitachi", "company": "日立製作所"}]
    train = [{"company_id": "mitsubishi_electric", "company": "三菱電機株式会社"},  # same company, other id
             {"company_id": "hitachi", "company": "日立"},                          # same id, other name
             {"company_id": "honda", "company": "本田技研工業"}]
    assert overlapping_companies(train, eval_sources) == {"三菱電機株式会社", "日立"}


def test_overlapping_companies_matches_shortened_names():
    from multimodal_rag.finetune.data import overlapping_companies

    eval_sources = [{"company_id": "hitachi", "company": "日立製作所"}]
    assert overlapping_companies([{"company_id": "hitachi_ltd", "company": "日立"}], eval_sources) == {"日立"}
