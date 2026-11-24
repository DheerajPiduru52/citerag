"""FastAPI app: routes, lifespan-loaded models and error mapping."""
import logging
import re
import tempfile
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path, PureWindowsPath
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from langchain_core.embeddings import Embeddings
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Settings, get_settings
from app.generate import answer_question
from app.ingest import EncryptedPDFError, PDFParseError
from app.llm import LLMClient, LLMError, LLMQuotaExceeded, LLMUpstreamError, OpenAICompatibleLLM
from app.rerank import Reranker
from app.retrieval import Candidate, HybridRetriever
from app.schemas import (
    Citation, DeleteResponse, DocumentInfo, DocumentListResponse, DocumentUploadResponse,
    DocumentUploadResult, ErrorResponse, HealthResponse, QueryRequest, QueryResponse,
    RetrievedChunk,
)
from app.store import DocumentStore, make_embeddings

logger = logging.getLogger("citerag")

TAGLINE = "Hybrid-Search Document Q&A with Verified Citations"
VERSION = "0.1.0"
PDF_MAGIC = b"%PDF-"
NO_TEXT_MSG = (
    "No extractable text found in '{name}' (it may be a scanned/image-only PDF); "
    "nothing was indexed."
)


class LockedRetriever:
    """HybridRetriever wrapper that holds the store lock while retrieving.

    Only retrieval takes the lock; reranking and the (slow) LLM call run outside it,
    so uploads are never blocked behind a model call.
    """

    def __init__(self, inner: HybridRetriever, lock: Any) -> None:
        """Wrap `inner`; `lock` is the store's (re-entrant) lock."""
        self.inner, self.lock = inner, lock

    def retrieve_hybrid(self, *args: Any, **kwargs: Any) -> list[Candidate]:
        """Hybrid retrieval under the store lock."""
        with self.lock:
            return self.inner.retrieve_hybrid(*args, **kwargs)


class ValidationErrorResponse(ErrorResponse):
    """ErrorResponse plus the per-field validation errors (HTTP 422)."""

    errors: list[dict[str, Any]]


def _err(status: int, detail: str, code: str) -> JSONResponse:
    """JSON error response in the standard ErrorResponse shape."""
    return JSONResponse(ErrorResponse(detail=detail, code=code).model_dump(), status_code=status)


def safe_basename(name: str | None) -> str:
    """Keep only the final path component of a client filename (metadata only)."""
    base = PureWindowsPath(name or "").name
    base = re.sub(r"[\x00-\x1f\x7f]", "", base).strip()
    return base[:255] or "upload.pdf"


def _to_chunk(c: Candidate) -> RetrievedChunk:
    """Convert a retrieval Candidate to its API model."""
    return RetrievedChunk(
        chunk_id=c.chunk_id, doc_id=c.doc_id, filename=c.filename, page=c.page, text=c.text,
        vector_score=c.vector_score, bm25_score=c.bm25_score, fused_score=c.fused_score,
        rerank_score=c.rerank_score,
    )


def _stage_upload(
    up: UploadFile, name: str, dest: Path, limit: int, budget: list[int] | None = None
) -> Path:
    """Validate and copy an upload to `dest` (a server-chosen path), enforcing size caps.

    `limit` is the per-file cap; `budget` is a one-element list holding the bytes still
    allowed for the whole request (decremented as files are staged).
    """
    if not name.lower().endswith(".pdf"):
        raise HTTPException(415, f"'{name}' is not a PDF (expected a .pdf file).")
    size = 0
    with open(dest, "wb") as out:
        while block := up.file.read(1 << 20):
            size += len(block)
            if size > limit:
                raise HTTPException(
                    413, f"'{name}' exceeds the maximum upload size of {limit / 1048576:g} MB."
                )
            if budget is not None:
                budget[0] -= len(block)
                if budget[0] < 0:
                    raise HTTPException(413, "Total upload size for this request is too large.")
            out.write(block)
    if size == 0:
        raise HTTPException(400, f"'{name}' is empty.")
    with open(dest, "rb") as f:
        if PDF_MAGIC not in f.read(1024):
            raise HTTPException(415, f"'{name}' is not a PDF (missing %PDF header).")
    return dest


def _ingest(
    store: DocumentStore, path: Path, name: str, added: list[str]
) -> DocumentUploadResult:
    """Ingest one staged PDF; map parse failures and text-less PDFs to HTTP errors.

    Only PDFParseError (raised by ingest while parsing) is a client error; embedding,
    FAISS and save failures propagate and become a 500.
    """
    try:
        res = store.add_document(path, name)
    except EncryptedPDFError as exc:
        logger.warning("Encrypted PDF '%s': %s", name, exc)
        raise HTTPException(
            400, f"'{name}' is encrypted; encrypted PDFs are not supported."
        ) from None
    except PDFParseError as exc:
        logger.warning("Could not parse '%s': %s", name, exc)
        raise HTTPException(
            400, f"'{name}' is not a readable PDF (corrupt or unsupported)."
        ) from None
    if not res.skipped:
        added.append(res.doc_id)
        if res.chunks == 0:
            raise HTTPException(422, NO_TEXT_MSG.format(name=name))
    return DocumentUploadResult(
        doc_id=res.doc_id, filename=res.filename, chunks=res.chunks,
        pages=store.documents[res.doc_id]["pages"], skipped=res.skipped,
    )


def create_app(
    settings: Settings | None = None,
    *,
    embeddings: Embeddings | None = None,
    reranker: Any | None = None,
    llm: LLMClient | None = None,
) -> FastAPI:
    """Build the app. Models load in the lifespan; injected fakes skip the real ones.

    `reranker` may be a Reranker or a cross-encoder-like object with `.predict`.
    """
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Load models and the store once at startup and attach them to app.state."""
        emb = embeddings or make_embeddings(cfg)
        store = DocumentStore(emb, cfg.index_path, cfg)
        if reranker is None:
            rr = Reranker(settings=cfg)
        elif hasattr(reranker, "rerank"):
            rr = reranker
        else:
            rr = Reranker(reranker, cfg)
        lock = threading.RLock()
        app.state.settings = cfg
        app.state.lock = lock
        app.state.store = store
        app.state.retriever = LockedRetriever(HybridRetriever(store, cfg), lock)
        app.state.reranker = rr
        app.state.llm = llm or OpenAICompatibleLLM(cfg)
        yield

    app = FastAPI(title="CiteRAG", description=TAGLINE, version=VERSION, lifespan=lifespan)

    @app.exception_handler(LLMError)
    async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
        """Map LLM failures to their HTTP status and an `llm_*` error code."""
        if isinstance(exc, LLMQuotaExceeded):
            code = "llm_quota_exceeded"
        elif isinstance(exc, LLMUpstreamError):
            code = "llm_upstream_error"
        else:
            code = "llm_error"
        logger.warning("LLM error %s: %s", code, exc.message)
        return _err(exc.http_status, exc.message, code)

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Render HTTP errors in the standard ErrorResponse shape."""
        return JSONResponse(
            ErrorResponse(detail=str(exc.detail), code=f"http_{exc.status_code}").model_dump(),
            status_code=exc.status_code,
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Render request validation failures (422) with per-field errors."""
        body = ValidationErrorResponse(
            detail="Request validation failed", code="validation_error",
            errors=jsonable_encoder(exc.errors(), custom_encoder={Exception: str}),
        )
        return JSONResponse(body.model_dump(), status_code=422)

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Log unexpected errors; return a generic 500 without internals."""
        logger.error("Unhandled error on %s %s", request.method, request.url.path,
                     exc_info=exc)
        return _err(500, "Internal server error", "internal_error")

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        """Liveness check."""
        return HealthResponse()

    @app.post(
        "/documents",
        response_model=DocumentUploadResponse,
        tags=["documents"],
        responses={
            400: {"model": ErrorResponse, "description": "Empty or corrupt PDF"},
            413: {
                "model": ErrorResponse,
                "description": "A file exceeds MAX_UPLOAD_MB or the request exceeds MAX_REQUEST_MB",
            },
            415: {"model": ErrorResponse, "description": "Not a PDF"},
            500: {"model": ErrorResponse, "description": "Embedding, index or save failure"},
            422: {
                "model": ValidationErrorResponse,
                "description": (
                    "PDF has no extractable text (ErrorResponse), or request validation "
                    "failed (ErrorResponse plus `errors`)"
                ),
            },
        },
    )
    def upload_documents(
        request: Request, files: list[UploadFile] = File(...)
    ) -> DocumentUploadResponse:
        """Upload one or more PDFs (field `files`). Duplicates return `skipped=true`.

        The request is all-or-nothing: if any file fails, documents newly added by
        this request are removed again. Always returns 200 on success (deliberate:
        a batch may mix newly added and skipped duplicates; see each result's `skipped`).

        Size limits: `max_upload_mb` per file and `max_request_mb` across all files.
        The framework has already received and spooled the multipart body before this
        handler runs, so the Content-Length pre-check below is not an early network
        guard: it only avoids staging and ingesting an oversized request (413). The
        streaming caps cover missing or lying headers.
        """
        st = request.app.state
        limit = int(st.settings.max_upload_mb * 1024 * 1024)
        total = int(st.settings.max_request_mb * 1024 * 1024)
        declared = request.headers.get("content-length", "")
        if declared.isascii() and declared.isdigit() and int(declared) > total:
            raise HTTPException(
                413, f"Request exceeds the maximum total upload size of {total / 1048576:g} MB."
            )
        budget = [total]
        with tempfile.TemporaryDirectory(prefix="citerag-") as tmp:
            staged = [
                (name, _stage_upload(up, name, Path(tmp) / f"{i}.pdf", limit, budget))
                for i, up in enumerate(files)
                for name in [safe_basename(up.filename)]
            ]
            results: list[DocumentUploadResult] = []
            added: list[str] = []
            with st.lock:
                try:
                    for name, path in staged:
                        results.append(_ingest(st.store, path, name, added))
                except BaseException:
                    # Best-effort rollback: a failing delete must not stop the rest
                    # nor mask the original error (re-raised below).
                    for doc_id in added:
                        try:
                            st.store.delete_document(doc_id)
                        except Exception:
                            logger.exception("Rollback failed to delete document %s", doc_id)
                    raise
        return DocumentUploadResponse(results=results)

    @app.get("/documents", response_model=DocumentListResponse, tags=["documents"])
    def list_documents(request: Request) -> DocumentListResponse:
        """List ingested documents."""
        with request.app.state.lock:
            docs = [DocumentInfo(**d) for d in request.app.state.store.list_documents()]
        return DocumentListResponse(documents=docs, count=len(docs))

    @app.delete(
        "/documents/{doc_id}",
        response_model=DeleteResponse,
        tags=["documents"],
        responses={
            404: {"model": ErrorResponse, "description": "Unknown doc_id"},
            422: {"model": ValidationErrorResponse, "description": "Request validation failed"},
        },
    )
    def delete_document(doc_id: str, request: Request) -> DeleteResponse:
        """Remove a document from the FAISS index and the BM25 corpus."""
        # Lock trade-off (deliberate): uploads hold the store lock for the whole ingest,
        # so a /query's retrieval step waits behind an in-flight upload.
        store = request.app.state.store
        with request.app.state.lock:
            info = store.documents.get(doc_id)
            if info is None:
                raise HTTPException(404, f"Document '{doc_id}' not found.")
            removed = info["chunks"]
            store.delete_document(doc_id)
        return DeleteResponse(doc_id=doc_id, chunks_removed=removed)

    @app.post(
        "/query",
        response_model=QueryResponse,
        tags=["query"],
        responses={
            422: {"model": ValidationErrorResponse, "description": "Request validation failed"},
            429: {"model": ErrorResponse, "description": "LLM quota exceeded"},
            500: {"model": ErrorResponse, "description": "LLM misconfigured / internal error"},
            502: {"model": ErrorResponse, "description": "LLM provider upstream failure"},
        },
    )
    def query(body: QueryRequest, request: Request) -> QueryResponse:
        """Answer a question with validated inline citations.

        `top_k` is applied at rerank over the fused candidate pool (up to
        vector_k + bm25_k unique chunks), so fewer than `top_k` chunks can be returned
        when the pool is smaller.
        """
        # Benign race: a document deleted after retrieval (retrieval lock is released
        # before rerank/LLM) may still be cited in this one answer; it is gone next query.
        st = request.app.state
        result = answer_question(
            body.question, st.retriever, st.reranker, st.llm, body.top_k
        )
        return QueryResponse(
            answer=result.answer,
            citations=[Citation(**vars(c)) for c in result.citations],
            retrieved_chunks=[_to_chunk(c) for c in result.reranked],
            latency_ms=round(result.latency_ms, 2),
        )

    return app


app = create_app()
