import io
import json
import urllib.error

import numpy as np
import pytest

from multimodal_rag.eval.runner import check_hits
from multimodal_rag.ingest.captions import (
    cache_path,
    caption_page,
    figure_text,
    load_captions,
    parse_figures,
    request_body,
    settings_key,
)
from multimodal_rag.retrieval.caption_dense import CaptionDenseRetriever, best_unit_per_page

CONFIG = {
    "ollama_url": "http://localhost:11434", "model": "m", "model_digest": "abc",
    "temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 4096, "prompt": "describe",
}
FIG = {"type": "table", "title": "売上", "unit": "億円", "table_markdown": "| a | 1 |", "summary": "増えた"}


def test_parse_figures_accepts_valid_reply():
    assert parse_figures(json.dumps({"figures": [FIG]})) == [FIG]
    assert parse_figures('{"figures": []}') == []


@pytest.mark.parametrize("content", [
    "not json",
    '{"pages": []}',
    json.dumps({"figures": [FIG | {"type": "photo"}]}),
    json.dumps({"figures": [{"type": "chart", "title": "t"}]}),
])
def test_parse_figures_rejects_bad_replies(content):
    with pytest.raises(ValueError):
        parse_figures(content)


def test_request_is_deterministic_and_constrained():
    body = request_body(CONFIG, "aW1n")
    assert body["options"] == {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 4096}
    assert body["format"]["required"] == ["figures"]
    assert body["messages"][0]["images"] == ["aW1n"]
    assert body["messages"][0]["content"].startswith("describe")


def test_settings_key_tracks_everything_that_changes_the_reply():
    base = settings_key(CONFIG)
    assert settings_key(dict(CONFIG)) == base
    for change in ({"prompt": "other"}, {"seed": 1}, {"model_digest": "def"}, {"num_predict": 10}):
        assert settings_key(CONFIG | change) != base
    # The server address doesn't change the reply.
    assert settings_key(CONFIG | {"ollama_url": "http://other:1"}) == base


def test_caption_page_keeps_unparseable_reply_with_error(tmp_path):
    image = tmp_path / "p.png"
    image.write_bytes(b"png")
    reply = {"message": {"content": '{"figures": [{"type": "chart"'}, "done_reason": "length", "eval_count": 4096,
             "total_duration": 2_500_000_000}
    record = caption_page(CONFIG, image, post=lambda url, body: reply)
    assert record["figures"] == [] and "not valid JSON" in record["error"]
    assert record["done_reason"] == "length" and record["total_duration_s"] == 2.5


def test_load_captions_requires_every_page(tmp_path):
    path = cache_path(tmp_path, "acme/fy/p001")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"figures": [FIG]}), encoding="utf-8")
    assert load_captions(tmp_path, ["acme/fy/p001"]) == {"acme/fy/p001": [FIG]}
    with pytest.raises(FileNotFoundError):
        load_captions(tmp_path, ["acme/fy/p001", "acme/fy/p002"])


def test_figure_text_skips_empty_fields():
    assert figure_text(FIG) == "売上\n単位：億円\n増えた\n| a | 1 |"
    assert figure_text(FIG | {"unit": "", "table_markdown": ""}) == "売上\n増えた"


def test_best_unit_per_page_by_hand():
    # units: (page 0, 0.2), (page 0, 0.9), (page 2, 0.5); page 1 has no units
    scores = best_unit_per_page(np.array([0.2, 0.9, 0.5]), np.array([0, 0, 2]), n_pages=3)
    assert scores[0] == pytest.approx(0.9)
    assert scores[1] == -np.inf
    assert scores[2] == pytest.approx(0.5)


class FakeEncoder:
    def __init__(self, table):
        self.table = table

    def _encode(self, sentences, **kwargs):
        return np.array([self.table[s] for s in sentences], dtype=np.float32)

    encode_document = _encode
    encode_query = _encode


def test_retriever_ranks_pages_by_best_figure_and_skips_empty_pages(tmp_path):
    encoder = FakeEncoder({"f1": [1, 0], "f2": [0, 1], "f3": [0.6, 0.8], "q": [0, 1]})
    pages = [
        {"page_id": "a", "captions": ["f1", "f2"]},  # best figure f2: 1.0
        {"page_id": "b", "captions": []},            # no figures -> not returned
        {"page_id": "c", "captions": ["f3"]},        # 0.8
    ]
    retriever = CaptionDenseRetriever(encoder)
    retriever.index(pages)
    hits = retriever.search("q", k=3)
    assert [h.page_id for h in hits] == ["a", "c"]
    assert [h.score for h in hits] == pytest.approx([1.0, 0.8])
    check_hits(hits, k=3, page_ids={"a", "b", "c"})

    retriever.save_index(tmp_path / "index.npz")
    reloaded = CaptionDenseRetriever(encoder)
    reloaded.load_index(tmp_path / "index.npz")
    assert reloaded.page_ids == ["a", "b", "c"]
    assert reloaded.search("q", k=3) == hits


def test_caption_page_records_server_abort_as_error(tmp_path):
    image = tmp_path / "p.png"
    image.write_bytes(b"png")

    def aborting_post(url, body):
        raise urllib.error.HTTPError(url, 500, "Internal Server Error", {},
                                     io.BytesIO(b'{"error":"prediction aborted, token repeat limit reached"}'))

    record = caption_page(CONFIG, image, post=aborting_post)
    assert record["figures"] == []
    assert record["error"].startswith("HTTP 500") and "repeat limit" in record["error"]


def test_caption_page_lets_connection_errors_raise(tmp_path):
    image = tmp_path / "p.png"
    image.write_bytes(b"png")

    def refused(url, body):
        raise urllib.error.URLError("connection refused")

    with pytest.raises(urllib.error.URLError):
        caption_page(CONFIG, image, post=refused)
