"""Ranking, reciprocal rank fusion, and reranking."""
from __future__ import annotations

from collections import defaultdict

from .chunk import Chunk
from .index import Index
from .text import expand, tokenize


def ranked(scores: list[float]) -> list[int]:
    return [i for i, s in sorted(enumerate(scores), key=lambda x: -x[1]) if s > 0]


def rrf(rankings: list[list[int]], weights: list[float], k: int = 60) -> list[int]:
    fused: dict[int, float] = defaultdict(float)
    for ranking, w in zip(rankings, weights):
        for r, i in enumerate(ranking):
            fused[i] += w / (k + r + 1)
    return sorted(fused, key=fused.get, reverse=True)


def bm25_search(ix: Index, q: str) -> list[int]:
    return ranked(ix.lexical(q))


def semantic_search(ix: Index, q: str) -> list[int]:
    return ranked(ix.semantic(q))


def hybrid_search(ix: Index, q: str, w_sem: float = 0.5) -> list[int]:
    return rrf([bm25_search(ix, q), semantic_search(ix, q)], [1 - w_sem, w_sem])


def heuristic_rerank_score(chunk: Chunk, query: str) -> float:
    """Transparent stand-in for a cross-encoder: coverage + heading match + focus."""
    qs = set(expand(query))
    ct = set(tokenize(chunk.text))
    coverage = len(qs & ct) / max(1, len(qs))
    heading = chunk.text.split(".", 1)[0]
    head_match = 1.0 if qs & set(tokenize(heading)) else 0.0
    focus = chunk.sections[chunk.primary] / max(1, chunk.n_words)
    return 0.6 * coverage + 0.25 * head_match + 0.15 * focus


def rerank(ix: Index, q: str, candidates: list[int], top_n: int = 8, model: str | None = None) -> list[int]:
    head, tail = candidates[:top_n], candidates[top_n:]
    if model:
        from sentence_transformers import CrossEncoder

        scores = CrossEncoder(model).predict([(q, ix.chunks[i].text) for i in head])
    else:
        scores = [heuristic_rerank_score(ix.chunks[i], q) for i in head]
    return [i for _, i in sorted(zip(scores, head), key=lambda x: -x[0])] + tail


def search(ix: Index, q: str, k: int = 4, w_sem: float = 0.5, use_rerank: bool = True) -> list[Chunk]:
    ids = hybrid_search(ix, q, w_sem)
    if use_rerank:
        ids = rerank(ix, q, ids)
    return [ix.chunks[i] for i in ids[:k]]
