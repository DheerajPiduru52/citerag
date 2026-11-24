"""FAISS vector store plus a JSON metadata store, persisted to disk."""
import copy
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import Settings, get_settings
from app.ingest import load_pdf, sha256_file

logger = logging.getLogger("citerag")

INDEX_NAME = "index"
META_FILE = "metadata.json"


def make_embeddings(settings: Settings | None = None) -> Embeddings:
    """Construct the (normalized) HuggingFace embedding model. Call once."""
    from langchain_huggingface import HuggingFaceEmbeddings

    s = settings or get_settings()
    return HuggingFaceEmbeddings(
        model_name=s.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


@dataclass
class AddResult:
    """Outcome of add_document."""

    doc_id: str
    filename: str
    chunks: int
    skipped: bool


class DocumentStore:
    """Vector index (inner product) and metadata for ingested documents."""

    def __init__(
        self,
        embeddings: Embeddings,
        index_dir: str | Path | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Open (or create) the store in `index_dir`, loading any saved index."""
        self.settings = settings or get_settings()
        self.embeddings = embeddings
        self.dir = Path(index_dir or self.settings.index_path)
        self.documents: dict[str, dict[str, Any]] = {}
        self.chunks: dict[str, dict[str, Any]] = {}
        self.vs: FAISS | None = None
        self._load()

    # ---- persistence ----
    def _load(self) -> None:
        """Load metadata and FAISS index from disk if present."""
        meta_path = self.dir / META_FILE
        if meta_path.exists():
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            self.documents = data["documents"]
            self.chunks = data["chunks"]
        if (self.dir / f"{INDEX_NAME}.faiss").exists():
            self.vs = FAISS.load_local(
                str(self.dir),
                self.embeddings,
                index_name=INDEX_NAME,
                allow_dangerous_deserialization=True,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
                normalize_L2=False,
            )
        self._reconcile()

    def _reconcile(self) -> None:
        """Make vectors and metadata agree after a partial save (in memory only).

        save() writes the FAISS files before metadata.json, so a crash can leave them
        out of step. Vectors whose id is not in metadata (orphans) are deleted from
        the index so a re-upload works; documents with chunks missing from the index
        are dropped with all their chunks and vectors. The next save() persists this.
        """
        vector_ids: set[str] = set()
        if self.vs is not None:
            vector_ids = set(self.vs.index_to_docstore_id.values())
            in_docstore = set(self.vs.docstore._dict)  # type: ignore[attr-defined]
            if vector_ids != in_docstore:
                logger.warning(
                    "FAISS docstore and index ids disagree (%d vs %d); using the docstore ids.",
                    len(vector_ids), len(in_docstore),
                )
                vector_ids &= in_docstore
        bad_docs = {c["doc_id"] for cid, c in self.chunks.items() if cid not in vector_ids}
        bad_docs |= {
            d for d, info in self.documents.items()
            if info.get("chunks", 0) > 0
            and not any(c["doc_id"] == d for c in self.chunks.values())
        }
        if bad_docs:
            logger.warning(
                "Index and metadata disagree; dropping %d document(s) with missing vectors: %s",
                len(bad_docs), sorted(bad_docs),
            )
            self.chunks = {c: v for c, v in self.chunks.items() if v["doc_id"] not in bad_docs}
            for d in bad_docs:
                self.documents.pop(d, None)
        orphans = [i for i in vector_ids if i not in self.chunks]
        if orphans and self.vs is not None:
            logger.warning("Dropping %d orphan vector(s) with no metadata.", len(orphans))
            self.vs.delete(orphans)

    def save(self) -> None:
        """Persist the FAISS index and metadata to disk, as atomically as possible.

        Everything is first written into a temp directory next to the index
        directory (same filesystem), then moved into place with os.replace.
        os.replace is atomic per file only (a directory with content cannot be
        swapped atomically on Windows), so the order matters: the FAISS files
        (.faiss, then .pkl) are replaced first and metadata.json LAST. A crash
        mid-way never leaves a truncated file; at worst the vectors are newer
        than metadata.json, which is the source of truth for listing/BM25.
        """
        self.dir.parent.mkdir(parents=True, exist_ok=True)
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=f".{self.dir.name}.tmp-", dir=self.dir.parent))
        try:
            names: list[str] = []
            if self.vs is not None:
                self.vs.save_local(str(tmp), index_name=INDEX_NAME)
                names += [f"{INDEX_NAME}.faiss", f"{INDEX_NAME}.pkl"]
            payload = {"documents": self.documents, "chunks": self.chunks}
            (tmp / META_FILE).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            names.append(META_FILE)
            for name in names:
                os.replace(tmp / name, self.dir / name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # ---- in-memory snapshot / rollback ----
    def _snapshot(self) -> dict[str, Any]:
        """Capture in-memory state (FAISS index, docstore, mappings, metadata)."""
        vs = None
        if self.vs is not None:
            vs = (
                faiss.serialize_index(self.vs.index),
                dict(self.vs.docstore._dict),  # type: ignore[attr-defined]
                dict(self.vs.index_to_docstore_id),
            )
        return {
            "vs": vs,
            "documents": copy.deepcopy(self.documents),
            "chunks": copy.deepcopy(self.chunks),
        }

    def _restore(self, snap: dict[str, Any]) -> None:
        """Restore in-memory state captured by _snapshot."""
        self.documents = snap["documents"]
        self.chunks = snap["chunks"]
        if snap["vs"] is None:
            self.vs = None
            return
        index_bytes, docs, mapping = snap["vs"]
        self.vs = FAISS(
            self.embeddings,
            faiss.deserialize_index(index_bytes),
            InMemoryDocstore(docs),
            mapping,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            normalize_L2=False,
        )

    # ---- vector index ----
    def _new_vs(self) -> FAISS:
        """Create an empty inner-product FAISS store."""
        dim = len(self.embeddings.embed_query("dimension probe"))
        return FAISS(
            self.embeddings,
            faiss.IndexFlatIP(dim),
            InMemoryDocstore({}),
            {},
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            normalize_L2=False,
        )

    def _add_chunks(self, chunks: list[Document]) -> None:
        """Embed chunks and add them to FAISS under their chunk_ids."""
        if self.vs is None:
            self.vs = self._new_vs()
        self.vs.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])

    # ---- public API ----
    def find_by_hash(self, sha256: str) -> dict[str, Any] | None:
        """Return the document record with this hash, if any."""
        return next(
            (d for d in self.documents.values() if d["sha256"] == sha256), None
        )

    def add_document(self, path: Path, filename: str | None = None) -> AddResult:
        """Ingest a PDF; skip (skipped=True) if its SHA-256 is already stored.

        The file is hashed before any parsing/chunking. If persisting fails,
        in-memory state is rolled back and the exception is re-raised.
        """
        digest = sha256_file(path)
        existing = self.find_by_hash(digest)
        if existing:
            return AddResult(
                existing["doc_id"], existing["filename"], existing["chunks"], True
            )
        doc_id, digest, chunks, pages = load_pdf(path, filename, self.settings, digest)
        name = filename or path.name
        snap = self._snapshot()
        try:
            if chunks:
                self._add_chunks(chunks)
            self.documents[doc_id] = {
                "doc_id": doc_id,
                "filename": name,
                "sha256": digest,
                "chunks": len(chunks),
                "pages": pages,
            }
            for c in chunks:
                self.chunks[c.metadata["chunk_id"]] = {
                    "text": c.page_content,
                    **c.metadata,
                }
            self.save()
        except BaseException:
            self._restore(snap)
            raise
        return AddResult(doc_id, name, len(chunks), False)

    def list_documents(self) -> list[dict[str, Any]]:
        """List ingested documents."""
        return list(self.documents.values())

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document's vectors and metadata. Returns False if unknown.

        If persisting fails, in-memory state is rolled back and the error re-raised.
        """
        if doc_id not in self.documents:
            return False
        ids = [cid for cid, c in self.chunks.items() if c["doc_id"] == doc_id]
        snap = self._snapshot()
        try:
            if self.vs is not None and ids:
                self.vs.delete(ids)
            for cid in ids:
                del self.chunks[cid]
            del self.documents[doc_id]
            self.save()
        except BaseException:
            self._restore(snap)
            raise
        return True

    def all_chunks(self) -> list[dict[str, Any]]:
        """Return all chunk records (text + metadata), e.g. for BM25."""
        return list(self.chunks.values())

    def similarity_search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Inner-product search; returns (chunk, score) pairs."""
        if self.vs is None or not self.chunks:
            return []
        return self.vs.similarity_search_with_score(query, k=k)
