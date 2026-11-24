"""Phase 4 tests: the FastAPI app with fake embeddings, cross-encoder and LLM."""
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.citations import IDK_ANSWER  # noqa: E402
from app.config import Settings  # noqa: E402
from app.llm import LLMError, LLMQuotaExceeded, LLMUpstreamError  # noqa: E402
from app.main import create_app, safe_basename  # noqa: E402
from helpers import FakeEmbeddings  # noqa: E402
from test_ingest_store import make_pdf  # noqa: E402

ERROR_PAGE = "The controller shows error E-4012 when suction pressure is too low. " * 3
OTHER_PAGE = "Zephyr gearbox lubrication uses grease GR-77 every season. " * 5


class FakeLLM:
    """Records calls; replies with a canned string or raises a given error."""

    def __init__(self, reply: str = "Error E-4012 means low suction pressure [1].") -> None:
        self.reply: str = reply
        self.error: Exception | None = None
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self.error:
            raise self.error
        return self.reply


class FakeCrossEncoder:
    """Scores by word overlap with the query."""

    def predict(self, pairs):
        return [len(set(q.lower().split()) & set(t.lower().split())) for q, t in pairs]


def make_settings(tmp_path: Path, **kw: object) -> Settings:
    return Settings(_env_file=None, index_path=str(tmp_path / "idx"), **kw)  # type: ignore[call-arg]


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def client(tmp_path: Path, llm: FakeLLM) -> Iterator[TestClient]:
    app = create_app(
        make_settings(tmp_path), embeddings=FakeEmbeddings(),
        reranker=FakeCrossEncoder(), llm=llm,
    )
    with TestClient(app) as c:
        yield c


@pytest.fixture
def pdf_a(tmp_path: Path) -> Path:
    return make_pdf(tmp_path / "manual.pdf", [ERROR_PAGE, "Pumps need regular maintenance."])


@pytest.fixture
def pdf_b(tmp_path: Path) -> Path:
    return make_pdf(tmp_path / "gearbox.pdf", [OTHER_PAGE])


def upload(client: TestClient, *paths: Path, name: str | None = None):
    files = [("files", (name or p.name, p.read_bytes(), "application/pdf")) for p in paths]
    return client.post("/documents", files=files)


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "service": "citerag"}


def test_import_loads_no_models(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.main as m

    assert not hasattr(m.app.state, "store")  # only the lifespan creates state

    def boom(*a: object, **k: object) -> None:
        raise AssertionError("model constructed outside the lifespan")

    monkeypatch.setattr(m, "make_embeddings", boom)
    monkeypatch.setattr(m, "Reranker", boom)
    built = m.create_app()  # building the app (no lifespan) must not load models
    assert not hasattr(built.state, "store")


def test_upload_pdf(client: TestClient, pdf_a: Path) -> None:
    r = upload(client, pdf_a)
    assert r.status_code == 200
    res = r.json()["results"][0]
    assert res["doc_id"] and res["chunks"] >= 2 and res["pages"] == 2
    assert res["filename"] == "manual.pdf" and res["skipped"] is False


def test_upload_multiple(client: TestClient, pdf_a: Path, pdf_b: Path) -> None:
    r = upload(client, pdf_a, pdf_b)
    assert r.status_code == 200
    results = r.json()["results"]
    assert [x["filename"] for x in results] == ["manual.pdf", "gearbox.pdf"]
    assert len({x["doc_id"] for x in results}) == 2
    assert client.get("/documents").json()["count"] == 2


def test_duplicate_skipped(client: TestClient, pdf_a: Path) -> None:
    first = upload(client, pdf_a).json()["results"][0]
    r = upload(client, pdf_a, name="renamed.pdf")
    assert r.status_code == 200
    again = r.json()["results"][0]
    assert again["skipped"] is True and again["doc_id"] == first["doc_id"]
    assert client.get("/documents").json()["count"] == 1


def test_non_pdf_rejected(client: TestClient) -> None:
    r = client.post("/documents", files=[("files", ("notes.txt", b"hello", "text/plain"))])
    assert r.status_code == 415
    assert "not a PDF" in r.json()["detail"]
    r = client.post("/documents", files=[("files", ("fake.pdf", b"hello world", "application/pdf"))])
    assert r.status_code == 415  # .pdf name but no PDF magic bytes


def test_empty_file_rejected(client: TestClient) -> None:
    r = client.post("/documents", files=[("files", ("empty.pdf", b"", "application/pdf"))])
    assert r.status_code == 400
    assert "empty" in r.json()["detail"]


def test_corrupt_pdf_rejected(client: TestClient) -> None:
    bad = b"%PDF-1.4\nthis is not a real pdf body\n"
    r = client.post("/documents", files=[("files", ("bad.pdf", bad, "application/pdf"))])
    assert r.status_code == 400
    assert "readable PDF" in r.json()["detail"]
    assert client.get("/documents").json()["count"] == 0


def test_oversize_rejected(tmp_path: Path, pdf_a: Path) -> None:
    app = create_app(
        make_settings(tmp_path, max_upload_mb=0.001), embeddings=FakeEmbeddings(),
        reranker=FakeCrossEncoder(), llm=FakeLLM(),
    )
    with TestClient(app) as c:
        r = upload(c, pdf_a)
        assert r.status_code == 413
        assert "maximum upload size" in r.json()["detail"]
        assert c.get("/documents").json()["count"] == 0


def test_no_text_pdf_rejected_and_not_stored(client: TestClient, tmp_path: Path) -> None:
    from fpdf import FPDF

    blank = FPDF()
    blank.add_page()
    blank.output(str(tmp_path / "blank.pdf"))
    r = upload(client, tmp_path / "blank.pdf")
    assert r.status_code == 422
    assert "No extractable text" in r.json()["detail"]
    assert client.get("/documents").json()["count"] == 0


def test_batch_is_all_or_nothing(client: TestClient, pdf_a: Path) -> None:
    bad = b"%PDF-1.4\nnot a pdf\n"
    r = client.post("/documents", files=[
        ("files", ("manual.pdf", pdf_a.read_bytes(), "application/pdf")),
        ("files", ("bad.pdf", bad, "application/pdf")),
    ])
    assert r.status_code == 400
    assert client.get("/documents").json()["count"] == 0


def test_client_filename_is_basename_only(client: TestClient, pdf_a: Path) -> None:
    assert safe_basename("..\\..\\evil/../x.pdf") == "x.pdf"
    r = upload(client, pdf_a, name="../../etc/passwd.pdf")
    assert r.status_code == 200
    assert r.json()["results"][0]["filename"] == "passwd.pdf"


def test_list_documents(client: TestClient, pdf_a: Path) -> None:
    assert client.get("/documents").json() == {"documents": [], "count": 0}
    doc_id = upload(client, pdf_a).json()["results"][0]["doc_id"]
    body = client.get("/documents").json()
    assert body["count"] == 1
    doc = body["documents"][0]
    assert doc["doc_id"] == doc_id and doc["filename"] == "manual.pdf" and doc["pages"] == 2


def test_delete_then_404(client: TestClient, pdf_a: Path) -> None:
    doc_id = upload(client, pdf_a).json()["results"][0]["doc_id"]
    r = client.delete(f"/documents/{doc_id}")
    assert r.status_code == 200
    assert r.json()["doc_id"] == doc_id and r.json()["chunks_removed"] >= 2
    assert client.get("/documents").json()["count"] == 0
    r2 = client.delete(f"/documents/{doc_id}")
    assert r2.status_code == 404
    assert "not found" in r2.json()["detail"]


def test_query(client: TestClient, pdf_a: Path, pdf_b: Path, llm: FakeLLM) -> None:
    upload(client, pdf_a, pdf_b)
    r = client.post("/query", json={"question": "What does error E-4012 mean?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Error E-4012 means low suction pressure [1]."
    assert body["latency_ms"] > 0
    chunks = body["retrieved_chunks"]
    assert chunks and len(chunks) <= 5
    top = chunks[0]
    assert top["filename"] == "manual.pdf" and top["page"] == 1
    assert top["vector_score"] is not None
    assert top["bm25_score"] is not None
    assert top["rerank_score"] is not None
    assert body["citations"] == [{
        "marker": 1, "filename": "manual.pdf", "page": 1, "chunk_id": top["chunk_id"],
        "snippet": body["citations"][0]["snippet"], "invalid": False,
    }]
    assert len(llm.calls) == 1


def test_query_top_k(client: TestClient, pdf_a: Path, pdf_b: Path) -> None:
    upload(client, pdf_a, pdf_b)
    r = client.post("/query", json={"question": "error pumps grease", "top_k": 1})
    assert len(r.json()["retrieved_chunks"]) == 1


def test_invalid_citation_flagged(client: TestClient, pdf_a: Path, llm: FakeLLM) -> None:
    upload(client, pdf_a)
    llm.reply = "It means low pressure [1] and also something else [9]."
    body = client.post("/query", json={"question": "error E-4012"}).json()
    by_marker = {c["marker"]: c for c in body["citations"]}
    assert by_marker[1]["invalid"] is False
    assert by_marker[9]["invalid"] is True
    assert by_marker[9]["chunk_id"] is None


def test_query_empty_index_no_llm_call(client: TestClient, llm: FakeLLM) -> None:
    r = client.post("/query", json={"question": "anything at all?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == IDK_ANSWER
    assert body["citations"] == [] and body["retrieved_chunks"] == []
    assert llm.calls == []


@pytest.mark.parametrize("payload", [
    {"question": ""}, {"question": "   "}, {"question": "x" * 5000},
    {"question": "ok?", "top_k": 0}, {"question": "ok?", "top_k": 100}, {},
])
def test_query_validation_errors(client: TestClient, payload: dict) -> None:
    assert client.post("/query", json=payload).status_code == 422


@pytest.mark.parametrize("error,status,code", [
    (LLMQuotaExceeded("LLM quota exceeded; try again later."), 429, "llm_quota_exceeded"),
    (LLMUpstreamError("The LLM provider failed upstream (status 502)."), 502,
     "llm_upstream_error"),
    (LLMError("LLM_API_KEY is missing or a placeholder.", 500), 500, "llm_error"),
    (LLMError("LLM unavailable.", 503), 503, "llm_error"),
])
def test_llm_errors_mapped(
    client: TestClient, pdf_a: Path, llm: FakeLLM, error: Exception, status: int, code: str
) -> None:
    upload(client, pdf_a)
    llm.error = error
    r = client.post("/query", json={"question": "error E-4012"})
    assert r.status_code == status
    assert r.json() == {"detail": str(error), "code": code}


def test_deleted_document_not_retrievable(
    client: TestClient, pdf_a: Path, pdf_b: Path
) -> None:
    ids = {x["filename"]: x["doc_id"] for x in upload(client, pdf_a, pdf_b).json()["results"]}
    client.delete(f"/documents/{ids['manual.pdf']}")
    body = client.post("/query", json={"question": "error E-4012 suction pressure"}).json()
    assert body["retrieved_chunks"]
    assert all(c["doc_id"] != ids["manual.pdf"] for c in body["retrieved_chunks"])


def test_persists_across_restart(tmp_path: Path, pdf_a: Path) -> None:
    def build() -> TestClient:
        return TestClient(create_app(
            make_settings(tmp_path), embeddings=FakeEmbeddings(),
            reranker=FakeCrossEncoder(), llm=FakeLLM(),
        ))

    with build() as c:
        upload(c, pdf_a)
    with build() as c:
        assert c.get("/documents").json()["count"] == 1


def test_batch_rollback_survives_failing_delete(
    client: TestClient, pdf_a: Path, pdf_b: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = client.app.state.store
    real_delete = store.delete_document
    calls: list[str] = []

    def flaky(doc_id: str) -> bool:
        calls.append(doc_id)
        if len(calls) == 1:
            raise OSError("disk gone")
        return real_delete(doc_id)

    monkeypatch.setattr(store, "delete_document", flaky)
    bad = b"%PDF-1.4\nnot a pdf\n"
    r = client.post("/documents", files=[
        ("files", ("manual.pdf", pdf_a.read_bytes(), "application/pdf")),
        ("files", ("gearbox.pdf", pdf_b.read_bytes(), "application/pdf")),
        ("files", ("bad.pdf", bad, "application/pdf")),
    ])
    assert r.status_code == 400  # the ORIGINAL error, not the rollback OSError
    assert "readable PDF" in r.json()["detail"]
    assert len(calls) == 2  # second delete still attempted
    assert [d["filename"] for d in store.list_documents()] == ["manual.pdf"]  # first failed


def test_batch_failure_keeps_preexisting_document(
    client: TestClient, pdf_a: Path, pdf_b: Path
) -> None:
    upload(client, pdf_a)
    bad = b"%PDF-1.4\nnot a pdf\n"
    r = client.post("/documents", files=[
        ("files", ("manual.pdf", pdf_a.read_bytes(), "application/pdf")),  # skipped dup
        ("files", ("gearbox.pdf", pdf_b.read_bytes(), "application/pdf")),
        ("files", ("bad.pdf", bad, "application/pdf")),
    ])
    assert r.status_code == 400
    assert [d["filename"] for d in client.get("/documents").json()["documents"]] == ["manual.pdf"]


def test_encrypted_pdf_rejected(client: TestClient, pdf_a: Path, tmp_path: Path) -> None:
    from pypdf import PdfReader, PdfWriter

    w = PdfWriter()
    for p in PdfReader(str(pdf_a)).pages:
        w.add_page(p)
    w.encrypt("secret", algorithm="RC4-128")
    enc = tmp_path / "enc.pdf"
    with open(enc, "wb") as f:
        w.write(f)
    r = upload(client, enc)
    assert r.status_code == 400
    assert "encrypted PDFs are not supported" in r.json()["detail"]
    assert client.get("/documents").json()["count"] == 0


def test_embedding_failure_is_500_not_client_error(
    tmp_path: Path, pdf_a: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    emb = FakeEmbeddings()
    app = create_app(make_settings(tmp_path), embeddings=emb,
                     reranker=FakeCrossEncoder(), llm=FakeLLM())
    with TestClient(app, raise_server_exceptions=False) as c:
        def boom(*a: object, **k: object) -> None:
            raise ValueError("embedding backend exploded")

        monkeypatch.setattr(emb, "embed_documents", boom)
        r = upload(c, pdf_a)
        assert r.status_code == 500
        assert r.json() == {"detail": "Internal server error", "code": "internal_error"}
        assert "exploded" not in r.text
        assert c.get("/documents").json()["count"] == 0


def test_content_length_precheck_413(tmp_path: Path, pdf_a: Path) -> None:
    app = create_app(
        make_settings(tmp_path, max_request_mb=0.001), embeddings=FakeEmbeddings(),
        reranker=FakeCrossEncoder(), llm=FakeLLM(),
    )
    with TestClient(app) as c:
        r = upload(c, pdf_a)
        assert r.status_code == 413
        assert "maximum total upload size" in r.json()["detail"]
        assert c.get("/documents").json()["count"] == 0


def test_total_budget_enforced_across_files(tmp_path: Path) -> None:
    import io

    from fastapi import HTTPException, UploadFile

    from app.main import _stage_upload

    budget = [150]
    for i in range(2):
        up = UploadFile(io.BytesIO(b"%PDF-" + b"x" * 95), filename=f"{i}.pdf")
        if i == 0:
            _stage_upload(up, "a.pdf", tmp_path / "0.pdf", 1000, budget)
        else:
            with pytest.raises(HTTPException) as ei:
                _stage_upload(up, "b.pdf", tmp_path / "1.pdf", 1000, budget)
            assert ei.value.status_code == 413


def test_top_k_larger_than_pool(tmp_path: Path, pdf_a: Path) -> None:
    app = create_app(
        make_settings(tmp_path, vector_k=1, bm25_k=1), embeddings=FakeEmbeddings(),
        reranker=FakeCrossEncoder(), llm=FakeLLM(),
    )
    with TestClient(app) as c:
        upload(c, pdf_a)
        r = c.post("/query", json={"question": "error E-4012 pumps", "top_k": 20})
        assert r.status_code == 200
        assert 1 <= len(r.json()["retrieved_chunks"]) <= 2  # pool is at most vector_k+bm25_k


def test_unhandled_error_is_generic_500(
    tmp_path: Path, pdf_a: Path
) -> None:
    class BadReranker:
        def predict(self, pairs):
            raise RuntimeError("secret internal detail")

    app = create_app(make_settings(tmp_path), embeddings=FakeEmbeddings(),
                     reranker=BadReranker(), llm=FakeLLM())
    with TestClient(app, raise_server_exceptions=False) as c:
        upload(c, pdf_a)
        r = c.post("/query", json={"question": "error E-4012"})
        assert r.status_code == 500
        assert r.json() == {"detail": "Internal server error", "code": "internal_error"}
        assert "secret" not in r.text


def test_validation_error_shape(client: TestClient) -> None:
    r = client.post("/query", json={"question": ""})
    assert r.status_code == 422
    body = r.json()
    assert body["code"] == "validation_error" and body["detail"]
    assert body["errors"] and "loc" in body["errors"][0] and "msg" in body["errors"][0]
    spec = client.get("/openapi.json").json()
    ref = spec["paths"]["/query"]["post"]["responses"]["422"]["content"]["application/json"]
    assert ref["schema"]["$ref"].endswith("/ValidationErrorResponse")


def test_openapi_metadata(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    assert spec["info"]["title"] == "CiteRAG"
    assert spec["info"]["description"] == "Hybrid-Search Document Q&A with Verified Citations"
    assert {"429", "502"} <= set(spec["paths"]["/query"]["post"]["responses"])
    assert {"400", "413", "415", "422"} <= set(spec["paths"]["/documents"]["post"]["responses"])


@pytest.mark.parametrize("reply", ["", "  \n "])
def test_empty_llm_reply_is_502_json(
    client: TestClient, pdf_a: Path, llm: FakeLLM, reply: str
) -> None:
    upload(client, pdf_a)
    llm.reply = reply
    r = client.post("/query", json={"question": "error E-4012"})
    assert r.status_code == 502
    assert r.json() == {"detail": "LLM returned an empty reply", "code": "llm_error"}


def test_openapi_documents_real_status_codes(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert "503" not in paths["/query"]["post"]["responses"]
    assert "500" in paths["/documents"]["post"]["responses"]

