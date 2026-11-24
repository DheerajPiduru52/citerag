"""Store tests: dedupe, round-trip, persistence, rollback (chunking: test_chunking.py)."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def _two_pdfs(tmp_path: Path) -> tuple[Path, Path]:
    a = make_pdf(tmp_path / "a.pdf", ["Alpha pump error E-4012 suction pressure. " * 8])
    b = make_pdf(tmp_path / "b.pdf", ["Zephyr gearbox grease GR-77 every season. " * 8])
    return a, b


def test_load_drops_orphan_vectors_after_partial_save(tmp_path: Path) -> None:
    """Index files newer than metadata.json: orphan vectors are dropped, re-upload works."""
    a, b = _two_pdfs(tmp_path)
    idx = tmp_path / "idx"
    store = DocumentStore(FakeEmbeddings(), idx)
    ra = store.add_document(a)
    old_meta = (idx / "metadata.json").read_bytes()
    rb = store.add_document(b)
    (idx / "metadata.json").write_bytes(old_meta)  # crash before metadata was replaced

    again = DocumentStore(FakeEmbeddings(), idx)
    assert [d["doc_id"] for d in again.list_documents()] == [ra.doc_id]
    assert again.vs is not None and again.vs.index.ntotal == ra.chunks
    assert set(again.vs.index_to_docstore_id.values()) == set(again.chunks)
    assert {d.metadata["doc_id"] for d, _ in again.similarity_search("gearbox grease", 10)} == {
        ra.doc_id
    }
    res = again.add_document(b)  # re-upload is not a duplicate and is searchable
    assert not res.skipped and res.doc_id == rb.doc_id
    assert again.vs.index.ntotal == ra.chunks + rb.chunks


def test_load_drops_documents_missing_vectors(tmp_path: Path) -> None:
    """metadata.json newer than the index files: chunks without vectors are dropped."""
    a, b = _two_pdfs(tmp_path)
    idx = tmp_path / "idx"
    store = DocumentStore(FakeEmbeddings(), idx)
    ra = store.add_document(a)
    old_faiss = (idx / "index.faiss").read_bytes()
    old_pkl = (idx / "index.pkl").read_bytes()
    rb = store.add_document(b)
    (idx / "index.faiss").write_bytes(old_faiss)
    (idx / "index.pkl").write_bytes(old_pkl)

    again = DocumentStore(FakeEmbeddings(), idx)
    assert [d["doc_id"] for d in again.list_documents()] == [ra.doc_id]
    assert len(again.all_chunks()) == ra.chunks
    assert not again.add_document(b).skipped
    assert again.find_by_hash(again.documents[rb.doc_id]["sha256"]) is not None
