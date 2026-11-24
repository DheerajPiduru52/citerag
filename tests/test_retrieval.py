"""Phase 2 tests: BM25, hybrid retrieval, reranking."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rerank import Reranker  # noqa: E402
from app.config import Settings  # noqa: E402
from app.retrieval import HybridRetriever, tokenize  # noqa: E402
from app.store import DocumentStore  # noqa: E402
from helpers import PAGES, FakeEmbeddings, add_pages  # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample.pdf"


@pytest.fixture
def retriever(tmp_path: Path) -> HybridRetriever:
    store = DocumentStore(FakeEmbeddings(), tmp_path / "idx")
    add_pages(store, "d1", PAGES)
    return HybridRetriever(store)


def test_tokenizer_keeps_ids_and_parts() -> None:
    assert tokenize("Error E-4012?") == ["error", "e-4012", "e", "4012"]
    assert "pn-4471-b" in tokenize("part PN-4471-B.") and "4471" in tokenize("PN-4471-B")


def test_bm25_exact_id(retriever: HybridRetriever) -> None:
    top = retriever.bm25_search("What does E-4012 mean?")[0]
    assert retriever.store.chunks[top[0]]["page"] == 1 and top[1] > 0
    top = retriever.bm25_search("PN-4471-B")[0]
    assert retriever.store.chunks[top[0]]["page"] == 2


def test_bm25_tiny_corpora_return_exact_id_hit(tmp_path: Path) -> None:
    one = HybridRetriever(DocumentStore(FakeEmbeddings(), tmp_path / "one"))
    add_pages(one.store, "a", [(1, PAGES[0][1])])
    hits = one.bm25_search("E-4012")
    assert len(hits) == 1 and one.store.chunks[hits[0][0]]["page"] == 1

    two = HybridRetriever(DocumentStore(FakeEmbeddings(), tmp_path / "two"))
    add_pages(two.store, "a", PAGES[:2])
    hits = two.bm25_search("E-4012")
    assert [two.store.chunks[c]["page"] for c, _ in hits] == [1]
    assert [c.page for c in two.retrieve_bm25("PN-4471-B")] == [2]


def test_bm25_no_token_overlap_not_returned(retriever: HybridRetriever) -> None:
    assert retriever.bm25_search("zorblax quux") == []
    assert retriever.retrieve_bm25("zorblax") == []


def test_retrieve_bm25_candidates(retriever: HybridRetriever) -> None:
    cands = retriever.retrieve_bm25("E-4012")
    assert cands and cands[0].page == 1 and cands[0].bm25_rank == 1
    assert cands[0].vector_score is None and cands[0].fused_score is None


def test_rrf_k_zero_honoured(retriever: HybridRetriever) -> None:
    cands = retriever.retrieve_hybrid("E-4012", rrf_k=0)
    top = cands[0]
    ranks = [r for r in (top.vector_rank, top.bm25_rank) if r is not None]
    assert top.fused_score == pytest.approx(sum(1.0 / r for r in ranks))


def test_default_settings_counts() -> None:
    s = Settings()
    assert (s.vector_k, s.bm25_k, s.rrf_k) == (20, 20, 60)


def test_hybrid_leaves_rerank_score_none(retriever: HybridRetriever) -> None:
    assert all(c.rerank_score is None for c in retriever.retrieve_hybrid("tank E-4012"))


def test_bm25_empty_corpus_and_query(tmp_path: Path) -> None:
    r = HybridRetriever(DocumentStore(FakeEmbeddings(), tmp_path / "e"))
    assert r.bm25_search("anything") == []
    assert r.retrieve_hybrid("anything") == []
    assert r.retrieve_vector("anything") == []


def test_hybrid_merges_both_lists(retriever: HybridRetriever) -> None:
    q = "E-4012 tank nearly empty"
    vec, kw = dict(retriever.vector_search(q)), dict(retriever.bm25_search(q))
    cands = retriever.retrieve_hybrid(q)
    assert {c.chunk_id for c in cands} == set(vec) | set(kw)
    for c in cands:
        assert c.vector_score == vec.get(c.chunk_id)
        assert c.bm25_score == kw.get(c.chunk_id)
        assert c.fused_score is not None and c.text
        assert (c.vector_rank is None) == (c.chunk_id not in vec)
        assert (c.bm25_rank is None) == (c.chunk_id not in kw)
    fused = [c.fused_score for c in cands]
    assert fused == sorted(fused, reverse=True)
    # chunks with no term overlap are in the vector list only
    assert any(c.bm25_score is None for c in cands)
    # a bounded vector list leaves BM25-only candidates with vector_score None
    narrow = retriever.retrieve_hybrid(q, vector_k=1, bm25_k=4)
    assert any(c.vector_score is None and c.bm25_score is not None for c in narrow)


def test_vector_only_mode(retriever: HybridRetriever) -> None:
    cands = retriever.retrieve_vector("tank nearly empty", k=2)
    assert len(cands) == 2
    assert all(c.bm25_score is None and c.fused_score is None for c in cands)
    assert cands[0].vector_score >= cands[1].vector_score


def test_candidate_counts_configurable(retriever: HybridRetriever) -> None:
    assert len(retriever.retrieve_vector("pump tank error", k=1)) == 1
    assert len(retriever.bm25_search("pump tank error", k=1)) == 1


def test_bm25_rebuild_after_add_and_delete(retriever: HybridRetriever) -> None:
    store = retriever.store
    assert retriever.bm25_search("zorblax") == []
    add_pages(store, "d2", [(1, "The zorblax valve must be calibrated.")])
    hits = retriever.bm25_search("zorblax")
    assert [store.chunks[c]["doc_id"] for c, _ in hits] == ["d2"]
    for cid in [c for c, v in store.chunks.items() if v["doc_id"] == "d2"]:
        del store.chunks[cid]
    assert retriever.bm25_search("zorblax") == []
    retriever.bm25.rebuild()  # explicit rebuild is idempotent
    assert retriever.bm25_search("E-4012")


def test_real_store_delete_invalidates_bm25(tmp_path: Path) -> None:
    store = DocumentStore(FakeEmbeddings(), tmp_path / "idx")
    r = HybridRetriever(store)
    assert r.bm25_search("E-4012") == []
    res = store.add_document(SAMPLE)
    assert r.bm25_search("E-4012")
    store.delete_document(res.doc_id)
    assert r.bm25_search("E-4012") == []


class FakeCrossEncoder:
    """Scores by number of query words in the text; counts predict calls."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls += 1
        return [float(sum(w in t.lower() for w in q.lower().split())) for q, t in pairs]


def test_reranker_orders_and_top_k(retriever: HybridRetriever) -> None:
    ce = FakeCrossEncoder()
    rr = Reranker(model=ce)
    cands = retriever.retrieve_hybrid("tank nearly empty")
    out = rr.rerank("tank nearly empty", cands, top_k=2)
    assert len(out) == 2 and out[0].rerank_score >= out[1].rerank_score
    assert out[0].page == 4
    assert out[0].vector_score == next(
        c.vector_score for c in cands if c.chunk_id == out[0].chunk_id
    )
    assert len(rr.rerank("tank", cands)) == min(5, len(cands))  # default top_k
    assert rr.rerank("x", []) == [] and ce.calls == 2


def test_rerank_top_k_validation_and_no_mutation(retriever: HybridRetriever) -> None:
    rr = Reranker(model=FakeCrossEncoder())
    cands = retriever.retrieve_hybrid("tank nearly empty")
    before = [(c.chunk_id, c.rerank_score) for c in cands]
    assert rr.rerank("tank", cands, top_k=0) == []
    with pytest.raises(ValueError):
        rr.rerank("tank", cands, top_k=-1)
    out = rr.rerank("tank nearly empty", cands, top_k=3)
    assert all(c.rerank_score is not None for c in out)
    assert [(c.chunk_id, c.rerank_score) for c in cands] == before
    assert all(c.rerank_score is None for c in cands)


def test_reranker_loads_cross_encoder_once(
    monkeypatch: pytest.MonkeyPatch, retriever: HybridRetriever
) -> None:
    import sentence_transformers

    loads: list[str] = []

    class Counting(FakeCrossEncoder):
        def __init__(self, name: str) -> None:
            super().__init__()
            loads.append(name)

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", Counting)
    rr = Reranker()
    cands = retriever.retrieve_hybrid("tank")
    rr.rerank("tank", cands)
    rr.rerank("pump", cands)
    assert len(loads) == 1 and rr.model.calls == 2


@pytest.mark.slow
def test_real_models_on_sample_pdf(tmp_path: Path) -> None:
    from app.store import make_embeddings

    store = DocumentStore(make_embeddings(), tmp_path / "idx")
    store.add_document(SAMPLE)
    retriever, reranker = HybridRetriever(store), Reranker()

    def pages(q: str) -> list[int]:
        return [c.page for c in reranker.rerank(q, retriever.retrieve_hybrid(q), 5)]

    assert 3 in pages("What does error E-4012 mean?")
    assert 8 in pages(
        "why does the pump keep switching off by itself when the tank is nearly empty"
    )


def test_vector_hits_without_metadata_are_skipped(retriever: HybridRetriever) -> None:
    full = retriever.vector_search("error E-4012", 10)
    assert full
    dropped = full[0][0]
    del retriever.store.chunks[dropped]  # vector remains, metadata gone
    assert dropped not in [c for c, _ in retriever.vector_search("error E-4012", 10)]
    assert dropped not in [c.chunk_id for c in retriever.retrieve_vector("error E-4012", 10)]
    assert dropped not in [c.chunk_id for c in retriever.retrieve_hybrid("error E-4012")]
