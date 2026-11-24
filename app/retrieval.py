"""Hybrid retrieval: FAISS vector search + BM25 keyword search + RRF fusion."""
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

from app.config import Settings, get_settings
from app.store import DocumentStore

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*")
_PART = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase tokenizer keeping compound IDs whole and adding their parts.

    "E-4012" -> ["e-4012", "e", "4012"], so exact IDs match as a unit while
    partial mentions still match on sub-tokens.
    """
    out: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        parts = _PART.findall(tok)
        if len(parts) > 1:
            out.append(tok)
        out.extend(parts)
    return out


@dataclass
class Candidate:
    """A retrieved chunk with per-stage scores (None = not in that list)."""

    chunk_id: str
    text: str
    doc_id: str
    filename: str
    page: int
    vector_score: float | None = None
    bm25_score: float | None = None
    fused_score: float | None = None
    vector_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Fuse ranked id lists: score(id) = sum 1 / (k + rank), rank 1-based.

    Returns (id, score) sorted by score desc; ties break by best (lowest)
    rank achieved in any list, then by id, so the result is deterministic.
    """
    scores: dict[str, float] = {}
    best: dict[str, int] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, cid in enumerate(ranking, 1):
            if cid in seen:
                continue
            seen.add(cid)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            best[cid] = min(best.get(cid, rank), rank)
    order = sorted(scores, key=lambda c: (-scores[c], best[c], c))
    return [(c, scores[c]) for c in order]


class BM25Index:
    """BM25 over a store's chunks; rebuilds itself when the chunk set changes."""

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self._ids: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._token_sets: list[frozenset[str]] = []
        self._signature: frozenset[str] | None = None

    def rebuild(self) -> None:
        """Rebuild the index from the store's current chunks."""
        chunks = self.store.chunks
        self._ids = sorted(chunks)
        corpus = [tokenize(chunks[c]["text"]) for c in self._ids]
        self._token_sets = [frozenset(t) for t in corpus]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._signature = frozenset(chunks)

    def _ensure_fresh(self) -> None:
        """Rebuild if chunks were added or deleted since the last build."""
        if self._signature != frozenset(self.store.chunks):
            self.rebuild()

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return up to k (chunk_id, score) for chunks sharing a query token, best first.

        Filtering is by token overlap, not score sign: Okapi IDF goes zero or
        negative for terms in half or more of the chunks (tiny corpora).
        """
        self._ensure_fresh()
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        query_tokens = set(tokens)
        hits = [
            (cid, float(s))
            for cid, s, toks in zip(self._ids, scores, self._token_sets)
            if query_tokens & toks
        ]
        hits.sort(key=lambda p: (-p[1], p[0]))
        return hits[:k]


class HybridRetriever:
    """Vector, BM25 and fused (RRF) retrieval over a DocumentStore."""

    def __init__(self, store: DocumentStore, settings: Settings | None = None) -> None:
        self.store = store
        self.settings = settings or get_settings()
        self.bm25 = BM25Index(store)

    def _candidate(self, chunk_id: str, **fields: Any) -> Candidate:
        c = self.store.chunks[chunk_id]
        return Candidate(
            chunk_id=chunk_id,
            text=c["text"],
            doc_id=c["doc_id"],
            filename=c["filename"],
            page=c["page"],
            **fields,
        )

    def vector_search(self, query: str, k: int | None = None) -> list[tuple[str, float]]:
        """Top-k (chunk_id, inner-product score) from FAISS."""
        if k is None:
            k = self.settings.vector_k
        hits = self.store.similarity_search(query, k)
        return [(d.metadata["chunk_id"], float(s)) for d, s in hits]

    def bm25_search(self, query: str, k: int | None = None) -> list[tuple[str, float]]:
        """Top-k (chunk_id, BM25 score)."""
        if k is None:
            k = self.settings.bm25_k
        return self.bm25.search(query, k)

    def retrieve_vector(self, query: str, k: int | None = None) -> list[Candidate]:
        """Vector-only mode: candidates ranked by vector score."""
        return [
            self._candidate(cid, vector_score=s, vector_rank=r)
            for r, (cid, s) in enumerate(self.vector_search(query, k), 1)
        ]

    def retrieve_bm25(self, query: str, k: int | None = None) -> list[Candidate]:
        """BM25-only mode: candidates ranked by BM25 score."""
        return [
            self._candidate(cid, bm25_score=s, bm25_rank=r)
            for r, (cid, s) in enumerate(self.bm25_search(query, k), 1)
        ]

    def retrieve_hybrid(
        self,
        query: str,
        vector_k: int | None = None,
        bm25_k: int | None = None,
        rrf_k: int | None = None,
    ) -> list[Candidate]:
        """Merge vector and BM25 lists with RRF; sorted by fused score."""
        if rrf_k is None:
            rrf_k = self.settings.rrf_k
        vec = self.vector_search(query, vector_k)
        kw = self.bm25_search(query, bm25_k)
        vscore, bscore = dict(vec), dict(kw)
        vrank = {c: r for r, (c, _) in enumerate(vec, 1)}
        brank = {c: r for r, (c, _) in enumerate(kw, 1)}
        fused = reciprocal_rank_fusion(
            [[c for c, _ in vec], [c for c, _ in kw]], rrf_k
        )
        return [
            self._candidate(
                cid,
                vector_score=vscore.get(cid),
                bm25_score=bscore.get(cid),
                fused_score=score,
                vector_rank=vrank.get(cid),
                bm25_rank=brank.get(cid),
            )
            for cid, score in fused
        ]
