"""Phase 3 demo: ingest sample.pdf, answer 2 questions with citations."""
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.generate import answer_question  # noqa: E402
from app.llm import LLMClient, OpenAICompatibleLLM, is_placeholder_key  # noqa: E402
from app.rerank import Reranker  # noqa: E402
from app.retrieval import HybridRetriever  # noqa: E402
from app.store import DocumentStore, make_embeddings  # noqa: E402

PDF = ROOT / "data" / "sample.pdf"
QUESTIONS = [
    "What does error E-4012 mean?",
    "Who won the 1998 football world cup?",
]


class ScriptedFakeLLM:
    """NOT a real model: cites the top source if the question words overlap, else says IDK."""

    def complete(self, system: str, user: str) -> str:
        question = user.rsplit("Question:", 1)[1].strip()
        first = re.search(r"<source id=\"1\"[^>]*>\n(.*?)\n</source>", user, re.S)
        words = {w for w in re.findall(r"[a-z0-9-]+", question.lower()) if len(w) > 3}
        if first and any(w in first.group(1).lower() for w in words):
            snippet = " ".join(first.group(1).split())[:160]
            return f"[scripted fake] According to the top source: {snippet} [1]"
        return "I don't know based on the provided documents"


def pick_llm() -> tuple[LLMClient, str]:
    """Real LLM if a non-placeholder key is configured, else the scripted fake."""
    settings = get_settings()
    if is_placeholder_key(settings.llm_api_key.get_secret_value()):
        return ScriptedFakeLLM(), "SCRIPTED FAKE LLM (no real LLM_API_KEY configured)"
    return OpenAICompatibleLLM(settings), f"REAL LLM ({settings.llm_model})"


def main() -> None:
    """Run the demo."""
    llm, mode = pick_llm()
    print(f"LLM MODE: {mode}")
    store = DocumentStore(make_embeddings(), tempfile.mkdtemp(prefix="citerag_demo3_"))
    store.add_document(PDF)
    retriever, reranker = HybridRetriever(store), Reranker()
    for q in QUESTIONS:
        res = answer_question(q, retriever, reranker, llm)
        print(f"\nQUESTION: {q}\nANSWER:\n{res.answer}\n\nCITATIONS:")
        for c in res.citations:
            if c.invalid:
                print(f"  [{c.marker}] INVALID (no such source)")
            else:
                print(f"  [{c.marker}] {c.filename} p.{c.page} ({c.chunk_id})\n      {c.snippet}")
        if not res.citations:
            print("  (none)")
        print("SOURCES USED (reranked):")
        for n, s in res.sources.items():
            print(f"  [{n}] {s.filename} p.{s.page} rerank={s.rerank_score:.3f}")
        print(f"LATENCY: {res.latency_ms:.0f} ms")


if __name__ == "__main__":
    main()
