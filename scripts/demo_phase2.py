"""Phase 2 demo: vector-only vs BM25 vs hybrid+rerank on the sample PDF."""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.rerank import Reranker  # noqa: E402
from app.retrieval import Candidate, HybridRetriever  # noqa: E402
from app.store import DocumentStore, make_embeddings  # noqa: E402

PDF = ROOT / "data" / "sample.pdf"
QUERIES = [
    "What does error E-4012 mean?",
    "why does the pump keep switching off by itself when the tank is nearly empty",
    "How do I replace board PN-4471-B after a firmware update fails?",
]


def fmt(x: float | None) -> str:
    """Format an optional score."""
    return "   -   " if x is None else f"{x:7.3f}"


def show(title: str, cands: list[Candidate]) -> None:
    """Print ranked candidates with page and all scores."""
    print(f"  {title}")
    for i, c in enumerate(cands, 1):
        fused = "   -   " if c.fused_score is None else f"{c.fused_score:7.4f}"
        print(
            f"    {i}. p.{c.page:<2} vec={fmt(c.vector_score)} bm25={fmt(c.bm25_score)} "
            f"fused={fused} rerank={fmt(c.rerank_score)}  {c.text[:60]!r}"
        )


def main() -> None:
    """Ingest the sample PDF into a temp index and run the queries."""
    store = DocumentStore(make_embeddings(), tempfile.mkdtemp(prefix="citerag_demo2_"))
    store.add_document(PDF)
    retriever, reranker = HybridRetriever(store), Reranker()
    for q in QUERIES:
        print(f"\nQUERY: {q}")
        show("vector-only top 5", retriever.retrieve_vector(q)[:5])
        show("BM25 top 5", retriever.retrieve_bm25(q)[:5])
        final = reranker.rerank(q, retriever.retrieve_hybrid(q), 5)
        show("hybrid (RRF) + rerank top 5", final)


if __name__ == "__main__":
    main()
