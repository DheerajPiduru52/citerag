"""PDF loading, hashing and chunking."""
import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import Settings, get_settings


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def doc_id_from_hash(sha256: str) -> str:
    """Derive a short, deterministic document id from a file hash."""
    return sha256[:16]


def extract_pages(path: Path) -> list[tuple[int, str]]:
    """Extract (1-based page number, text) pairs, skipping empty pages."""
    reader = PdfReader(str(path))
    pages = [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, 1)]
    return [(n, t) for n, t in pages if t.strip()]


def make_splitter(settings: Settings | None = None) -> RecursiveCharacterTextSplitter:
    """Build the text splitter from settings."""
    s = settings or get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap
    )


def chunk_pages(
    pages: list[tuple[int, str]],
    doc_id: str,
    filename: str,
    settings: Settings | None = None,
) -> list[Document]:
    """Split pages into chunks carrying doc_id, filename, page and chunk_id."""
    splitter = make_splitter(settings)
    chunks: list[Document] = []
    for page, text in pages:
        for n, piece in enumerate(splitter.split_text(text)):
            meta = {
                "doc_id": doc_id,
                "filename": filename,
                "page": page,
                "chunk_id": f"{doc_id}:{page}:{n}",
            }
            chunks.append(Document(page_content=piece, metadata=meta))
    return chunks


def load_pdf(
    path: Path,
    filename: str | None = None,
    settings: Settings | None = None,
    digest: str | None = None,
) -> tuple[str, str, list[Document], int]:
    """Hash (unless a precomputed digest is given), read and chunk a PDF.

    Returns (doc_id, sha256, chunks, page_count_with_text).
    """
    digest = digest or sha256_file(path)
    doc_id = doc_id_from_hash(digest)
    pages = extract_pages(path)
    name = filename or path.name
    return doc_id, digest, chunk_pages(pages, doc_id, name, settings), len(pages)
