"""Chunking tests: sizes, overlap, metadata, ids, hashing and settings overrides."""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.ingest import chunk_pages, doc_id_from_hash, load_pdf, sha256_file  # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample.pdf"


def _text(n_words: int) -> str:
    """Distinct words, so overlaps are unambiguous."""
    return " ".join(f"word{i:04d}" for i in range(n_words))


def _overlap(prev: str, nxt: str) -> int:
    """Length of the longest suffix of `prev` that is also a prefix of `nxt`."""
    for k in range(min(len(prev), len(nxt)), 0, -1):
        if prev.endswith(nxt[:k]):
            return k
    return 0


def test_default_settings_are_800_and_120() -> None:
    s = Settings(_env_file=None)
    assert (s.chunk_size, s.chunk_overlap) == (800, 120)


def test_chunk_size_never_exceeds_limit() -> None:
    chunks = chunk_pages([(1, _text(600))], "d", "f.pdf", Settings(_env_file=None))
    assert len(chunks) > 3
    assert all(len(c.page_content) <= 800 for c in chunks)


def test_consecutive_chunks_overlap_within_limit() -> None:
    chunks = chunk_pages([(1, _text(600))], "d", "f.pdf", Settings(_env_file=None))
    overlaps = [_overlap(a.page_content, b.page_content) for a, b in zip(chunks, chunks[1:])]
    assert overlaps and all(0 < o <= 120 for o in overlaps)
    assert max(overlaps) >= 100  # close to the configured 120


def test_no_overlap_across_pages() -> None:
    chunks = chunk_pages([(1, _text(300)), (2, _text(300))], "d", "f.pdf")
    first_p2 = next(c for c in chunks if c.metadata["page"] == 2)
    last_p1 = [c for c in chunks if c.metadata["page"] == 1][-1]
    assert first_p2.metadata["chunk_id"].endswith(":2:0")
    assert _overlap(last_p1.page_content, first_p2.page_content) in (0, len(first_p2.page_content))


def test_metadata_and_one_based_pages() -> None:
    settings = Settings(chunk_size=200, chunk_overlap=20, _env_file=None)
    chunks = chunk_pages([(1, "word " * 200), (3, "other " * 100)], "abc", "f.pdf", settings)
    assert len(chunks) > 2
    for c in chunks:
        assert set(c.metadata) == {"doc_id", "filename", "page", "chunk_id"}
        assert c.metadata["doc_id"] == "abc" and c.metadata["filename"] == "f.pdf"
        assert isinstance(c.metadata["page"], int) and c.metadata["page"] >= 1
    assert {c.metadata["page"] for c in chunks} == {1, 3}
    assert chunks[0].metadata["chunk_id"] == "abc:1:0"


def test_chunk_ids_unique_and_deterministic() -> None:
    pages = [(1, _text(400)), (2, _text(400))]
    a = chunk_pages(pages, "d", "f.pdf")
    b = chunk_pages(pages, "d", "f.pdf")
    ids = [c.metadata["chunk_id"] for c in a]
    assert len(set(ids)) == len(ids)
    assert ids == [c.metadata["chunk_id"] for c in b]
    assert [c.page_content for c in a] == [c.page_content for c in b]
    other = chunk_pages(pages, "e", "f.pdf")
    assert not {c.metadata["chunk_id"] for c in other} & set(ids)


def test_settings_override_changes_chunking() -> None:
    pages = [(1, _text(400))]
    big = chunk_pages(pages, "d", "f.pdf", Settings(chunk_size=800, chunk_overlap=120, _env_file=None))
    small = chunk_pages(pages, "d", "f.pdf", Settings(chunk_size=200, chunk_overlap=20, _env_file=None))
    assert len(small) > len(big)
    assert all(len(c.page_content) <= 200 for c in small)
    no_ov = chunk_pages(pages, "d", "f.pdf", Settings(chunk_size=200, chunk_overlap=0, _env_file=None))
    assert all(_overlap(a.page_content, b.page_content) == 0
               for a, b in zip(no_ov, no_ov[1:]))


def test_hash_is_stable_and_doc_id_derives_from_it(tmp_path: Path) -> None:
    digest = sha256_file(SAMPLE)
    assert digest == hashlib.sha256(SAMPLE.read_bytes()).hexdigest()
    assert sha256_file(SAMPLE) == digest
    copy = tmp_path / "renamed.pdf"
    copy.write_bytes(SAMPLE.read_bytes())
    assert sha256_file(copy) == digest
    assert doc_id_from_hash(digest) == digest[:16]


def test_sample_pdf_pages_and_chunks() -> None:
    doc_id, digest, chunks, pages = load_pdf(SAMPLE)
    assert pages == 22 and len(chunks) >= 40
    assert doc_id == doc_id_from_hash(digest)
    assert {c.metadata["page"] for c in chunks} == set(range(1, pages + 1))
    assert all(len(c.page_content) <= 800 for c in chunks)
    assert {c.metadata["page"] for c in chunks if "E-4012" in c.page_content} == {3}
    assert all(c.metadata["filename"] == "sample.pdf" for c in chunks)
