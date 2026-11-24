"""Shared test helpers (imported by several test modules)."""
import hashlib
import sys
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingest import chunk_pages  # noqa: E402
from app.store import DocumentStore  # noqa: E402

DIM = 64


class FakeEmbeddings(Embeddings):
    """Deterministic hashed bag-of-words embeddings, L2-normalized."""

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(DIM, dtype=np.float32)
        for tok in text.lower().split():
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


PAGES = [
    (1, "The controller shows error E-4012 when suction pressure is too low."),
    (2, "Replace the board PN-4471-B only with an approved spare part."),
    (3, "Pumps move liquid. Regular maintenance keeps pumps running well."),
    (4, "The tank level sensor reports when the tank is nearly empty."),
]


def add_pages(store: DocumentStore, doc_id: str, pages: list[tuple[int, str]]) -> None:
    """Insert chunks directly (bypasses PDF parsing), in memory only."""
    chunks = chunk_pages(pages, doc_id, f"{doc_id}.pdf", store.settings)
    store._add_chunks(chunks)
    store.documents[doc_id] = {
        "doc_id": doc_id, "filename": f"{doc_id}.pdf", "sha256": doc_id,
        "chunks": len(chunks), "pages": len(pages),
    }
    for c in chunks:
        store.chunks[c.metadata["chunk_id"]] = {"text": c.page_content, **c.metadata}
