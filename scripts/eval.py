"""Retrieval evaluation over data/sample.pdf (no LLM).

Modes (all cut at the top 5 chunks):
  vector          FAISS top 5 by vector score
  bm25            BM25 top 5 (extra row, for context)
  hybrid+rerank   vector 20 + BM25 20 -> RRF (k=60) -> cross-encoder top 5

A question is a hit at rank r if the r-th chunk is the first whose
(filename, page) is in the question's accepted set. MRR = mean of 1/r,
with 0 for questions that have no match in the top 5.

Usage: python scripts/eval.py [--json out.json] [--questions path]
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.rerank import Reranker  # noqa: E402
from app.retrieval import Candidate, HybridRetriever  # noqa: E402
from app.store import DocumentStore, make_embeddings  # noqa: E402

PDF = ROOT / "data" / "sample.pdf"
QUESTIONS = ROOT / "data" / "eval_questions.json"
K = 5
MODES = ["vector", "bm25", "hybrid+rerank"]


def first_rank(cands: list[Candidate], accepted: set[tuple[str, int]]) -> int | None:
    """1-based rank of the first candidate whose (filename, page) is accepted."""
    for r, c in enumerate(cands[:K], 1):
        if (c.filename, c.page) in accepted:
            return r
    return None


def summarize(ranks: list[int | None]) -> dict[str, float]:
    """hit@1, hit@5 and MRR over a list of first-match ranks (None = miss)."""
    n = len(ranks)
    return {
        "hit@1": sum(r == 1 for r in ranks) / n,
        "hit@5": sum(r is not None for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
    }


def run(questions: list[dict[str, Any]], filename: str) -> list[dict[str, Any]]:
    """Index the sample PDF in a temp dir and rank every question in each mode."""
    with tempfile.TemporaryDirectory(prefix="citerag_eval_") as tmp:
        return _rank_all(DocumentStore(make_embeddings(), tmp), questions, filename)


def _rank_all(
    store: DocumentStore, questions: list[dict[str, Any]], filename: str
) -> list[dict[str, Any]]:
    """Ingest the PDF into `store` and rank each question in every mode."""
    store.add_document(PDF)
    retriever, reranker = HybridRetriever(store), Reranker()
    rows = []
    for q in questions:
        text = q["question"]
        accepted = {(filename, p) for p in q["pages"]}
        results = {
            "vector": retriever.retrieve_vector(text, K),
            "bm25": retriever.retrieve_bm25(text, K),
            "hybrid+rerank": reranker.rerank(text, retriever.retrieve_hybrid(text), K),
        }
        rows.append({
            **q,
            "ranks": {m: first_rank(results[m], accepted) for m in MODES},
            "top_pages": {m: [c.page for c in results[m]] for m in MODES},
        })
    return rows


def fmt_rank(r: int | None) -> str:
    """Rank or '-' for a miss."""
    return "-" if r is None else str(r)


def print_per_question(rows: list[dict[str, Any]]) -> None:
    """Print the per-question rank table."""
    print(f"{'id':<4} {'type':<10} {'pages':<7} {'vector':>6} {'bm25':>5} {'hybrid':>7}  question")
    for r in rows:
        k = r["ranks"]
        pages = ",".join(map(str, r["pages"]))
        print(f"{r['id']:<4} {r['type']:<10} {pages:<7} {fmt_rank(k['vector']):>6} "
              f"{fmt_rank(k['bm25']):>5} {fmt_rank(k['hybrid+rerank']):>7}  {r['question']}")


def summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per-mode metrics."""
    return {m: summarize([r["ranks"][m] for r in rows]) for m in MODES}


def print_summary(stats: dict[str, dict[str, float]], n: int) -> None:
    """Print the summary table and the README markdown table."""
    print(f"\nSummary ({n} questions, top {K})")
    print(f"{'mode':<15} {'hit@1':>6} {'hit@5':>6} {'MRR':>6}")
    for m in MODES:
        s = stats[m]
        print(f"{m:<15} {s['hit@1']:>6.2f} {s['hit@5']:>6.2f} {s['mrr']:>6.3f}")
    print("\nMarkdown table:\n")
    print("| Mode | hit@1 | hit@5 | MRR |")
    print("|---|---|---|---|")
    for m in MODES:
        s = stats[m]
        print(f"| {m} | {s['hit@1']:.2f} | {s['hit@5']:.2f} | {s['mrr']:.3f} |")


def main() -> None:
    """Run the evaluation and print the tables."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", type=Path, help="also write results to this JSON file")
    ap.add_argument("--questions", type=Path, default=QUESTIONS)
    args = ap.parse_args()
    data = json.loads(args.questions.read_text(encoding="utf-8"))
    rows = run(data["questions"], data["filename"])
    stats = summary(rows)
    print_per_question(rows)
    print_summary(stats, len(rows))
    if args.json:
        args.json.write_text(
            json.dumps({"summary": stats, "questions": rows}, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
