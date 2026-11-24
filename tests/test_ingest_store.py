"""Phase 1 tests: chunking, hashing/dedupe, store round-trip and persistence."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.ingest import chunk_pages, doc_id_from_hash, load_pdf, sha256_file  # noqa: E402
from app.store import DocumentStore  # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample.pdf"
DIM = 64


class FakeEmbeddings(Embeddings):
    """Deterministic hashed bag-of-words embeddings, L2-normalized."""

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(DIM, dtype=np.float32)
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % DIM] += 1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture
def store(tmp_path: Path) -> DocumentStore:
    return DocumentStore(FakeEmbeddings(), tmp_path / "idx")


def test_chunking_sizes_and_metadata() -> None:
    settings = Settings(chunk_size=200, chunk_overlap=20)
    pages = [(1, "word " * 200), (3, "other " * 100)]
    chunks = chunk_pages(pages, "abc", "f.pdf", settings)
    assert len(chunks) > 2
    assert all(len(c.page_content) <= 200 for c in chunks)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(set(ids)) == len(ids)
    assert ids[0] == "abc:1:0"
    for c in chunks:
        assert set(c.metadata) == {"doc_id", "filename", "page", "chunk_id"}
        assert c.metadata["doc_id"] == "abc" and c.metadata["filename"] == "f.pdf"
    assert {c.metadata["page"] for c in chunks} == {1, 3}


def test_sample_pdf_pages_and_hash() -> None:
    doc_id, digest, chunks, pages = load_pdf(SAMPLE)
    assert pages == 22
    assert len(chunks) >= 40
    assert digest == sha256_file(SAMPLE) == hashlib.sha256(SAMPLE.read_bytes()).hexdigest()
    assert doc_id == doc_id_from_hash(digest)
    assert {c.metadata["page"] for c in chunks} == set(range(1, pages + 1))
    assert any("E-4012" in c.page_content for c in chunks)
    assert {c.metadata["page"] for c in chunks if "E-4012" in c.page_content} == {3}


def test_add_dedupe_list_delete(store: DocumentStore) -> None:
    first = store.add_document(SAMPLE)
    assert not first.skipped and first.chunks > 0
    second = store.add_document(SAMPLE)
    assert second.skipped and second.doc_id == first.doc_id
    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0]["chunks"] == first.chunks and docs[0]["filename"] == "sample.pdf"
    assert len(store.all_chunks()) == first.chunks
    assert store.similarity_search("error E-4012 suction pressure", 3)

    assert store.delete_document(first.doc_id)
    assert not store.delete_document(first.doc_id)
    assert store.list_documents() == [] and store.all_chunks() == []
    assert store.similarity_search("anything", 3) == []


def test_persistence_reload(store: DocumentStore, tmp_path: Path) -> None:
    res = store.add_document(SAMPLE)
    reloaded = DocumentStore(FakeEmbeddings(), tmp_path / "idx")
    assert reloaded.list_documents() == store.list_documents()
    assert len(reloaded.all_chunks()) == res.chunks
    assert reloaded.add_document(SAMPLE).skipped
    hits = reloaded.similarity_search("error E-4012 suction pressure", 1)
    assert hits and hits[0][0].metadata["doc_id"] == res.doc_id
    reloaded.delete_document(res.doc_id)
    again = DocumentStore(FakeEmbeddings(), tmp_path / "idx")
    assert again.list_documents() == []


def make_pdf(path: Path, pages: list[str]) -> Path:
    """Generate a small PDF with one text per page (needs fpdf2)."""
    from fpdf import FPDF

    pdf = FPDF()
    for text in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, text)
    pdf.output(str(path))
    return path


def test_two_documents_delete_one_and_reload(tmp_path: Path) -> None:
    other = make_pdf(
        tmp_path / "other.pdf",
        ["Zephyr gearbox lubrication uses grease GR-77 every season. " * 8,
         "Gearbox torque limit is 90 Nm for bolt BT-12. " * 8],
    )
    idx = tmp_path / "idx"
    store = DocumentStore(FakeEmbeddings(), idx)
    a = store.add_document(SAMPLE)
    b = store.add_document(other)
    assert a.doc_id != b.doc_id and not b.skipped and b.chunks > 0
    assert len(store.all_chunks()) == a.chunks + b.chunks
    assert store.vs is not None and store.vs.index.ntotal == a.chunks + b.chunks

    assert store.delete_document(a.doc_id)
    assert [d["doc_id"] for d in store.list_documents()] == [b.doc_id]
    assert len(store.all_chunks()) == b.chunks
    hits = store.similarity_search("error E-4012 suction pressure", 10)
    assert len(hits) == b.chunks
    assert {d.metadata["doc_id"] for d, _ in hits} == {b.doc_id}

    again = DocumentStore(FakeEmbeddings(), idx)
    assert [d["doc_id"] for d in again.list_documents()] == [b.doc_id]
    assert len(again.all_chunks()) == b.chunks
    assert again.vs is not None and again.vs.index.ntotal == b.chunks
    assert set(again.vs.index_to_docstore_id.values()) == set(again.chunks)
    hits = again.similarity_search("gearbox torque", 10)
    assert {d.metadata["doc_id"] for d, _ in hits} == {b.doc_id}
    assert again.add_document(other).skipped and not again.add_document(SAMPLE).skipped


def _state(store: DocumentStore) -> tuple:
    vs = store.vs
    return (
        store.documents,
        store.chunks,
        None if vs is None else (vs.index.ntotal, dict(vs.index_to_docstore_id),
                                 set(vs.docstore._dict)),
    )


def test_add_document_rolls_back_when_save_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = make_pdf(tmp_path / "o.pdf", ["Rollback test content about widgets. " * 10])
    store = DocumentStore(FakeEmbeddings(), tmp_path / "idx")
    store.add_document(SAMPLE)
    before = _state(store)
    before = (dict(before[0]), dict(before[1]), before[2])

    def boom() -> None:
        raise OSError("disk full")

    monkeypatch.setattr(store, "save", boom)
    with pytest.raises(OSError):
        store.add_document(other)
    assert _state(store) == before
    # same for a store that has no index yet
    empty = DocumentStore(FakeEmbeddings(), tmp_path / "idx2")
    monkeypatch.setattr(empty, "save", boom)
    with pytest.raises(OSError):
        empty.add_document(other)
    assert empty.vs is None and empty.documents == {} and empty.chunks == {}

    # delete_document rolls back too
    doc_id = store.list_documents()[0]["doc_id"]
    with pytest.raises(OSError):
        store.delete_document(doc_id)
    assert _state(store) == before
    assert store.similarity_search("error E-4012 suction pressure", 1)


def test_save_leaves_no_temp_dirs(store: DocumentStore, tmp_path: Path) -> None:
    store.add_document(SAMPLE)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["idx"]
    assert sorted(p.name for p in (tmp_path / "idx").iterdir()) == [
        "index.faiss", "index.pkl", "metadata.json",
    ]


@pytest.mark.slow
def test_real_bge_embeddings(tmp_path: Path) -> None:
    from app.store import make_embeddings

    s = DocumentStore(make_embeddings(), tmp_path / "real")
    s.add_document(SAMPLE)
    top, score = s.similarity_search("What does error E-4012 mean?", 1)[0]
    assert top.metadata["page"] == 3
    assert 0 < score <= 1.0001
    q = "why does the pump keep switching off by itself when the tank is nearly empty"
    best = s.similarity_search(q, 3)
    assert any(d.metadata["page"] == 8 for d, _ in best)
