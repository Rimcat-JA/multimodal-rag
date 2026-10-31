import pytest
import torch
from PIL import Image
from sentence_transformers.util import maxsim

from multimodal_rag.eval.runner import check_hits
from multimodal_rag.retrieval.page_image import PageImageRetriever


def test_maxsim_by_hand():
    # query tokens q1=(1,0), q2=(0,1); page tokens d1=(1,0), d2=(0.6,0.8)
    # q1: max(q1.d1, q1.d2) = max(1.0, 0.6) = 1.0
    # q2: max(q2.d1, q2.d2) = max(0.0, 0.8) = 0.8
    # MaxSim = 1.0 + 0.8 = 1.8
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    page = torch.tensor([[1.0, 0.0], [0.6, 0.8]])
    assert maxsim([query], [page]).item() == pytest.approx(1.8)


def test_maxsim_ignores_page_length():
    # Adding a weak extra token to a page doesn't raise its score: only the
    # best match per query token counts, so long pages aren't favoured.
    query = torch.tensor([[1.0, 0.0]])
    short = torch.tensor([[0.8, 0.6]])
    long = torch.tensor([[0.8, 0.6], [0.0, 1.0], [0.0, -1.0]])
    assert maxsim([query], [short, long])[0].tolist() == pytest.approx([0.8, 0.8])


# Pages are tiny solid-colour images; the fake encoder turns each colour into
# fixed token vectors, so scores can be worked out by hand.
PAGE_TOKENS = {
    (255, 0, 0): [[1.0, 0.0], [0.0, 1.0]],    # red page
    (0, 0, 255): [[0.6, 0.8]],                # blue page
}
QUERY_TOKENS = {"q": [[1.0, 0.0], [0.0, 1.0]]}


class FakeEncoder:
    device = torch.device("cpu")

    def __init__(self):
        self.document_batches: list[int] = []

    def encode_document(self, inputs, **kwargs):
        self.document_batches.append(len(inputs))
        return [torch.tensor(PAGE_TOKENS[img.getpixel((0, 0))]) for img in inputs]

    def encode_query(self, inputs, **kwargs):
        return [torch.tensor(QUERY_TOKENS[q]) for q in inputs]


@pytest.fixture
def pages(tmp_path):
    records = []
    for i, colour in enumerate([(255, 0, 0), (0, 0, 255), (255, 0, 0)], start=1):
        rel = f"img/p{i:03d}.png"
        (tmp_path / "img").mkdir(exist_ok=True)
        Image.new("RGB", (4, 4), colour).save(tmp_path / rel)
        records.append({"page_id": f"doc/p{i:03d}", "image": rel})
    return records


def test_search_ranks_by_maxsim(tmp_path, pages):
    retriever = PageImageRetriever(FakeEncoder(), root=tmp_path, chunk_pages=2)
    retriever.index(pages)
    hits = retriever.search("q", k=3)
    # red pages: 1.0 + 1.0 = 2.0; blue page: 0.6 + 0.8 = 1.4
    # p001 and p003 tie at 2.0 and keep page order.
    assert [h.page_id for h in hits] == ["doc/p001", "doc/p003", "doc/p002"]
    assert [h.score for h in hits] == pytest.approx([2.0, 2.0, 1.4])
    check_hits(hits, k=3, page_ids={p["page_id"] for p in pages})


def test_index_reads_images_in_chunks(tmp_path, pages):
    encoder = FakeEncoder()
    PageImageRetriever(encoder, root=tmp_path, chunk_pages=2).index(pages)
    assert encoder.document_batches == [2, 1]


def test_save_and_load_index_round_trip(tmp_path, pages):
    first = PageImageRetriever(FakeEncoder(), root=tmp_path)
    first.index(pages)
    first.save_index(tmp_path / "index.pt")
    second = PageImageRetriever(FakeEncoder(), root=tmp_path)
    second.load_index(tmp_path / "index.pt")
    assert second.page_ids == first.page_ids
    assert second.search("q", k=3) == first.search("q", k=3)


def test_search_before_index_fails(tmp_path):
    with pytest.raises(RuntimeError):
        PageImageRetriever(FakeEncoder(), root=tmp_path).search("q", k=1)
