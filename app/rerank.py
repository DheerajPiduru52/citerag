"""Cross-encoder reranking of retrieval candidates."""
from dataclasses import replace
from typing import Any

from app.config import Settings, get_settings
from app.retrieval import Candidate


class Reranker:
    """Scores (query, chunk) pairs with a cross-encoder loaded once."""

    def __init__(self, model: Any | None = None, settings: Settings | None = None) -> None:
        """Use `model` (anything with .predict(pairs)) or load the configured one."""
        self.settings = settings or get_settings()
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(self.settings.rerank_model)
        self.model = model

    def rerank(
        self, query: str, candidates: list[Candidate], top_k: int | None = None
    ) -> list[Candidate]:
        """Return the top_k candidates by cross-encoder score, best first."""
        if top_k is None:
            top_k = self.settings.top_k
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
        if not candidates or top_k == 0:
            return []
        scores = self.model.predict([(query, c.text) for c in candidates])
        scored = [replace(c, rerank_score=float(s)) for c, s in zip(candidates, scores)]
        scored.sort(key=lambda c: -c.rerank_score)  # stable: ties keep input order
        return scored[:top_k]
