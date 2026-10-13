import importlib.util
from pathlib import Path

# scripts/ is not a package, so load the module from its file path.
_spec = importlib.util.spec_from_file_location(
    "validate_eval", Path(__file__).resolve().parent.parent / "scripts" / "validate_eval.py"
)
validate_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validate_eval)

HEADER = validate_eval.HEADER
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
        "notes": "",
    }
    return base | overrides


def test_valid_rows_have_no_errors():
    rows = [row(), row(qid="q002", gold_pages="acme/fy2025_full/p002;acme/fy2026_q1/p001",
                       q_type="compare", answer_value="", answer_unit="")]
    assert validate_eval.validate(rows, HEADER, PAGES) == []


def test_wrong_header_is_reported_alone():
    errors = validate_eval.validate([row()], HEADER[:-1], PAGES)
    assert len(errors) == 1 and "header" in errors[0]


def test_unknown_page_id_is_caught():
    # A typo like p01 instead of p001 would silently score as a miss.
    errors = validate_eval.validate([row(gold_pages="acme/fy2025_full/p01")], HEADER, PAGES)
    assert any("not found" in e for e in errors)


def test_duplicate_qid_and_bad_enum():
    errors = validate_eval.validate([row(), row(evidence="graph")], HEADER, PAGES)
    assert any("duplicates" in e for e in errors)
    assert any("evidence='graph'" in e for e in errors)


def test_number_formatting_rules():
    for bad in ["1,234", "約15", "15%", "15."]:
        errors = validate_eval.validate([row(answer_value=bad)], HEADER, PAGES)
        assert any("plain number" in e for e in errors), bad
    errors = validate_eval.validate([row(answer_value="-3.1")], HEADER, PAGES)
    assert errors == []


def test_value_and_unit_go_together():
    assert validate_eval.validate([row(answer_unit="")], HEADER, PAGES)
    assert validate_eval.validate([row(answer_value="")], HEADER, PAGES)
