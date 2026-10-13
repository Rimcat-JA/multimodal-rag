"""Editor for the evaluation set: look at page images, write questions, save CSV.

Run with:  uv run streamlit run scripts/eval_editor.py

The developer writes every question and answer here. The app only builds page
ids from the page being viewed, normalizes number formatting, validates the
row, and writes eval/questions.csv in the agreed format (eval/README.md).
"""

import csv
import json
from collections import Counter
from pathlib import Path

import streamlit as st

from multimodal_rag.evalset import (
    COMMON_UNITS,
    EVIDENCE_LABELS,
    GOLD_MODE_LABELS,
    Q_TYPE_LABELS,
    load_questions,
    next_qid,
    normalize_number,
    save_questions,
    split_pages,
    validate_row,
)

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "questions.csv"
PAGES = ROOT / "data" / "pages.jsonl"
SOURCES = ROOT / "docs" / "sources.csv"

PERIOD_LABELS = {"fy2025_full": "2026年3月期 通期", "fy2026_q1": "2027年3月期 第1四半期"}
INDUSTRY_LABELS = {
    "manufacturing": "製造", "finance": "金融", "pharma": "医薬品",
    "it_telecom": "IT・通信", "trading_realestate": "商社・不動産",
}
# Targets for the first 40 questions (eval/README.md)
TARGET_EVIDENCE = {"chart": 16, "table": 12, "diagram": 6, "text": 6}
TARGET_Q_TYPE = {"lookup": 16, "compare": 12, "trend": 8, "structure": 4}
TARGET_PER_INDUSTRY = 8

NO_UNIT, OTHER_UNIT = "（なし）", "その他（下に入力）"
NEW = "（新しい質問）"


@st.cache_data
def load_pages() -> dict[str, dict]:
    with PAGES.open(encoding="utf-8") as f:
        return {r["page_id"]: r for r in map(json.loads, f)}


@st.cache_data
def load_companies() -> dict[str, dict]:
    with SOURCES.open(encoding="utf-8") as f:
        companies = {}
        for r in csv.DictReader(f):
            companies.setdefault(r["company_id"], {"name": r["company"], "industry": r["industry"]})
        return companies


pages = load_pages()
companies = load_companies()
page_counts = Counter(p["doc_id"] for p in pages.values())


# ---------- state ----------

def reset_form() -> None:
    s = st.session_state
    s.editing = None
    s.gold = []
    s.f_question = s.f_answer = s.f_value = s.f_unit_other = s.f_notes = ""
    s.f_gold_mode = "all"
    s.f_q_type = None
    s.f_evidence = None
    s.f_unit = NO_UNIT


def init_state() -> None:
    s = st.session_state
    if "initialized" in s:
        return
    s.initialized = True
    s.msg = None
    s.co = next(iter(companies))
    s.period = "fy2025_full"
    s.pg = 1
    s.edit_select = NEW
    reset_form()


def doc_id() -> str:
    return f"{st.session_state.co}/{st.session_state.period}"


def clamp_page() -> None:
    st.session_state.pg = min(st.session_state.pg, page_counts[doc_id()])


def show_page(pid: str) -> None:
    company, period, p = pid.split("/")
    st.session_state.co, st.session_state.period, st.session_state.pg = company, period, int(p[1:])


def step_page(delta: int) -> None:
    st.session_state.pg = max(1, min(page_counts[doc_id()], st.session_state.pg + delta))


def add_gold(pid: str) -> None:
    if pid not in st.session_state.gold:
        st.session_state.gold.append(pid)


def remove_gold(pid: str) -> None:
    st.session_state.gold.remove(pid)


def load_into_form() -> None:
    s = st.session_state
    if s.edit_select == NEW:
        reset_form()
        return
    _, rows = load_questions(QUESTIONS)
    r = next(r for r in rows if r["qid"] == s.edit_select)
    s.editing = r["qid"]
    s.gold = split_pages(r["gold_pages"])
    s.f_question, s.f_answer, s.f_notes = r["question"], r["answer"], r["notes"]
    s.f_value = r["answer_value"]
    s.f_gold_mode = r["gold_mode"] or "all"
    s.f_q_type = r["q_type"] or None
    s.f_evidence = r["evidence"] or None
    unit = r["answer_unit"]
    if not unit:
        s.f_unit, s.f_unit_other = NO_UNIT, ""
    elif unit in COMMON_UNITS:
        s.f_unit, s.f_unit_other = unit, ""
    else:
        s.f_unit, s.f_unit_other = OTHER_UNIT, unit
    if s.gold:
        show_page(s.gold[0])


def save() -> None:
    s = st.session_state
    _, rows = load_questions(QUESTIONS)
    unit = {NO_UNIT: "", OTHER_UNIT: s.f_unit_other.strip()}.get(s.f_unit, s.f_unit)
    row = {
        "qid": s.editing or next_qid(rows),
        "question": s.f_question.strip(),
        "gold_pages": ";".join(s.gold),
        "gold_mode": s.f_gold_mode,
        "answer": s.f_answer.strip(),
        "answer_value": normalize_number(s.f_value),
        "answer_unit": unit,
        "q_type": s.f_q_type or "",
        "evidence": s.f_evidence or "",
        "notes": s.f_notes.strip(),
    }
    errors = validate_row(row, set(pages))
    if errors:
        s.msg = ("error", errors)
        return
    rows = [r for r in rows if r["qid"] != row["qid"]] + [row]
    save_questions(QUESTIONS, rows)
    s.msg = ("ok", [f"{row['qid']} を保存しました（数値は「{row['answer_value'] or 'なし'}」として記録）"])
    s.edit_select = NEW
    reset_form()


# ---------- layout ----------

st.set_page_config(page_title="評価セット入力", layout="wide")
init_state()
s = st.session_state
_, rows = load_questions(QUESTIONS)

st.title("評価セット入力")
if s.msg:
    kind, lines = s.msg
    (st.success if kind == "ok" else st.error)("\n".join(f"- {line}" for line in lines))

left, right = st.columns([3, 2], gap="large")

with left:
    c1, c2, c3 = st.columns([2, 2, 1])
    c1.selectbox("企業", list(companies), key="co", on_change=clamp_page,
                 format_func=lambda c: f"{companies[c]['name']}（{INDUSTRY_LABELS[companies[c]['industry']]}）")
    c2.radio("資料", list(PERIOD_LABELS), key="period", on_change=clamp_page,
             format_func=PERIOD_LABELS.get, horizontal=True)
    c3.number_input(f"ページ（全{page_counts[doc_id()]}）", min_value=1,
                    max_value=page_counts[doc_id()], key="pg")

    pid = f"{doc_id()}/p{s.pg:03d}"
    b1, b2, b3, b4 = st.columns([1, 1, 3, 3])
    b1.button("◀ 前", on_click=step_page, args=(-1,), use_container_width=True)
    b2.button("次 ▶", on_click=step_page, args=(1,), use_container_width=True)
    b3.button("このページを根拠に追加", on_click=add_gold, args=(pid,), type="primary",
              use_container_width=True, disabled=pid in s.gold)
    b4.code(pid, language=None)

    st.image(str(ROOT / pages[pid]["image"]), use_container_width=True)
    with st.expander("このページから取り出せたテキスト（方式Aが見る情報）"):
        st.text(pages[pid]["text"] or "（テキストなし）")

    st.subheader("同じ値が載っているページを探す")
    query = st.text_input("値（例：2,934 や △1,925）", key="search")
    if query.strip():
        num = normalize_number(query)
        variants = {query.strip(), num}
        if num.lstrip("-").replace(".", "").isdigit():
            whole, _, frac = num.lstrip("-").partition(".")
            variants.add(f"{int(whole):,}" + (f".{frac}" if frac else ""))
        hits = [p for p, r in pages.items() if any(v in r["text"] for v in variants)]
        st.caption(f"{len(hits)}ページ（テキスト抽出で見つかったものだけ。画像の中の文字は対象外）")
        for hit in hits[:30]:
            h1, h2, h3 = st.columns([4, 1, 1])
            h1.write(f"`{hit}`")
            h2.button("表示", key=f"show_{hit}", on_click=show_page, args=(hit,))
            h3.button("追加", key=f"add_{hit}", on_click=add_gold, args=(hit,), disabled=hit in s.gold)

with right:
    st.selectbox("編集する質問", [NEW] + [r["qid"] for r in rows], key="edit_select", on_change=load_into_form)
    st.markdown(f"**qid：** {s.editing or next_qid(rows) + '（保存時に付きます）'}")

    st.text_area("質問（スライドの見出しをそのまま写さない）", key="f_question", height=100)

    st.markdown("**根拠ページ**（左の「このページを根拠に追加」で入ります）")
    if not s.gold:
        st.caption("まだありません")
    for g in s.gold:
        g1, g2, g3 = st.columns([4, 1, 1])
        g1.write(f"`{g}`")
        g2.button("表示", key=f"gshow_{g}", on_click=show_page, args=(g,))
        g3.button("外す", key=f"gdel_{g}", on_click=remove_gold, args=(g,))
    st.radio("根拠ページの扱い", list(GOLD_MODE_LABELS), key="f_gold_mode", format_func=GOLD_MODE_LABELS.get)

    st.text_input("正解（文）", key="f_answer")
    v1, v2 = st.columns(2)
    v1.text_input("数値（数値で答える質問のみ。資料の表示どおり）", key="f_value",
                  help="カンマ付き・全角・△や▲のマイナスでも、保存時に整えます")
    v2.selectbox("単位", [NO_UNIT] + COMMON_UNITS + [OTHER_UNIT], key="f_unit")
    if s.f_unit == OTHER_UNIT:
        st.text_input("単位（その他）", key="f_unit_other")

    st.radio("聞き方", list(Q_TYPE_LABELS), key="f_q_type", format_func=Q_TYPE_LABELS.get, index=None)
    st.radio("根拠の種類", list(EVIDENCE_LABELS), key="f_evidence", format_func=EVIDENCE_LABELS.get,
             index=None, horizontal=True)
    st.text_area("メモ（迷った点など）", key="f_notes", height=70)

    a1, a2 = st.columns(2)
    a1.button("保存", on_click=save, type="primary", use_container_width=True)
    a2.button("入力をクリア", on_click=reset_form, use_container_width=True)

# ---------- progress ----------

st.divider()
st.subheader(f"進み具合：{len(rows)}問（最初の目標は40問）")
by_evidence = Counter(r["evidence"] for r in rows)
by_q_type = Counter(r["q_type"] for r in rows)
by_industry = Counter(
    companies[split_pages(r["gold_pages"])[0].split("/")[0]]["industry"]
    for r in rows if split_pages(r["gold_pages"])
)
p1, p2, p3 = st.columns(3)
with p1:
    st.markdown("**根拠の種類**")
    for code, target in TARGET_EVIDENCE.items():
        st.write(f"{EVIDENCE_LABELS[code]}：{by_evidence[code]} / {target}")
with p2:
    st.markdown("**聞き方**")
    for code, target in TARGET_Q_TYPE.items():
        st.write(f"{Q_TYPE_LABELS[code]}：{by_q_type[code]} / {target}")
with p3:
    st.markdown("**業界**")
    for code, label in INDUSTRY_LABELS.items():
        st.write(f"{label}：{by_industry[code]} / {TARGET_PER_INDUSTRY}")

st.dataframe(rows, use_container_width=True, hide_index=True)
