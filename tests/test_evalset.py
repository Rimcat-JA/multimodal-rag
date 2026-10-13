from multimodal_rag.evalset import (
    HEADER,
    load_questions,
    next_qid,
    normalize_number,
    save_questions,
    validate,
)

PAGES = {"acme/fy2025_full/p001", "acme/fy2025_full/p002", "acme/fy2026_q1/p001"}


def row(**overrides):
    base = {
        "qid": "q001",
        "question": "架空の質問",
        "gold_pages": "acme/fy2025_full/p001",
        "gold_mode": "all",
        "answer": "架空の答え",
        "answer_value": "15.2",
        "answer_unit": "%",
        "q_type": "lookup",
        "evidence": "chart",
        "author": "human",
        "reviewed": "none",
        "notes": "",
    }
    return base | overrides


def test_valid_rows_have_no_errors():
    rows = [row(), row(qid="q002", gold_pages="acme/fy2025_full/p002;acme/fy2026_q1/p001",
                       q_type="compare", answer_value="", answer_unit="")]
    assert validate(rows, HEADER, PAGES) == []


def test_wrong_header_is_reported_alone():
    errors = validate([row()], HEADER[:-1], PAGES)
    assert len(errors) == 1 and "ヘッダー" in errors[0]


def test_unknown_page_id_is_caught():
    # A typo like p01 instead of p001 would silently score as a miss.
    errors = validate([row(gold_pages="acme/fy2025_full/p01")], HEADER, PAGES)
    assert any("ありません" in e for e in errors)


def test_duplicate_qid_and_bad_code():
    errors = validate([row(), row(evidence="graph")], HEADER, PAGES)
    assert any("重複" in e for e in errors)
    assert any("graph" in e for e in errors)


def test_author_and_reviewed_must_use_known_codes():
    assert validate([row(author="claude", reviewed="ai")], HEADER, PAGES) == []
    assert any("gpt" in e for e in validate([row(author="gpt")], HEADER, PAGES))
    assert any("yes" in e for e in validate([row(reviewed="yes")], HEADER, PAGES))
    assert any("作成者" in e for e in validate([row(author="")], HEADER, PAGES))


def test_number_formatting_rules():
    for bad in ["1,234", "約15", "15%", "15."]:
        errors = validate([row(answer_value=bad)], HEADER, PAGES)
        assert any("数字だけ" in e for e in errors), bad
    assert validate([row(answer_value="-3.1")], HEADER, PAGES) == []


def test_value_and_unit_go_together():
    assert validate([row(answer_unit="")], HEADER, PAGES)
    assert validate([row(answer_value="")], HEADER, PAGES)


def test_normalize_number_handles_ir_notation():
    assert normalize_number("2,934") == "2934"
    assert normalize_number("２，９３４") == "2934"  # full-width
    assert normalize_number("△1,925") == "-1925"  # IR decks print negatives with △
    assert normalize_number("▲ 3.1") == "-3.1"
    assert normalize_number(" 15.2 ") == "15.2"
    assert normalize_number("約15") == "約15"  # left for validation to flag


def test_next_qid_continues_after_highest():
    assert next_qid([]) == "q001"
    assert next_qid([row(qid="q001"), row(qid="q007")]) == "q008"


def test_save_then_load_round_trip(tmp_path):
    path = tmp_path / "questions.csv"
    rows = [row(qid="q010"), row(qid="q002", notes='カンマ, と "引用符" を含むメモ')]
    save_questions(path, rows)
    header, loaded = load_questions(path)
    assert header == HEADER
    assert [r["qid"] for r in loaded] == ["q002", "q010"]  # sorted by number
    assert loaded[0]["notes"] == 'カンマ, と "引用符" を含むメモ'
