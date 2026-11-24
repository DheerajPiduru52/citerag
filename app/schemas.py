"""Pydantic v2 request/response models for the HTTP API."""
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

MAX_QUESTION_CHARS = 2000

Question = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS)
]


class HealthResponse(BaseModel):
    """Liveness payload."""

    status: str = "ok"
    service: str = "citerag"


class DocumentUploadResult(BaseModel):
    """Outcome for one uploaded PDF."""

    doc_id: str
    filename: str
    chunks: int
    pages: int
    skipped: bool = Field(description="True if identical content was already ingested.")


class DocumentUploadResponse(BaseModel):
    """Outcome for a (possibly multi-file) upload."""

    results: list[DocumentUploadResult]


class DocumentInfo(BaseModel):
    """An ingested document."""

    doc_id: str
    filename: str
    chunks: int
    pages: int
    sha256: str


class DocumentListResponse(BaseModel):
    """All ingested documents."""

    documents: list[DocumentInfo]
    count: int


class DeleteResponse(BaseModel):
    """Result of deleting a document."""

    doc_id: str
    deleted: bool = True
    chunks_removed: int


class QueryRequest(BaseModel):
    """A question over the ingested documents."""

    question: Question
    top_k: int | None = Field(
        default=None, ge=1, le=20,
        description=(
            "Max chunks kept after reranking. Applied over the fused candidate pool "
            "(up to vector_k + bm25_k unique chunks), so fewer than top_k may be returned."
        ),
    )


class Citation(BaseModel):
    """A citation marker resolved against the retrieved sources."""

    marker: int
    filename: str | None
    page: int | None
    chunk_id: str | None
    snippet: str | None
    invalid: bool = False


class RetrievedChunk(BaseModel):
    """A source chunk passed to the LLM, with per-stage scores."""

    chunk_id: str
    doc_id: str
    filename: str
    page: int
    text: str
    vector_score: float | None = None
    bm25_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None


class QueryResponse(BaseModel):
    """Answer with validated citations and the supporting chunks."""

    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    latency_ms: float


class ErrorResponse(BaseModel):
    """Error body."""

    detail: str
    code: str | None = None
