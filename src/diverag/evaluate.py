"""Retrieval metrics: recall@k, MRR, top-1 purity, context size."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .index import Index
from .retrieve import bm25_search, hybrid_search, rerank, semantic_search


@dataclass
class Scores:
    recall: float
    mrr: float
    purity: float
    context_words: float


def load_questions(path: str | Path = "eval/questions.jsonl") -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def score(ix: Index, questions: list[dict], retrieve: Callable[[Index, str], list[int]], k: int = 4) -> Scores:
    hits = rr = pur = ctx = 0.0
    for ex in questions:
        gold = set(ex["gold"])
        ids = retrieve(ix, ex["q"])[:k]
        rank = next((r for r, i in enumerate(ids) if gold & set(ix.chunks[i].sections)), None)
        if rank is not None:
            hits += 1
            rr += 1 / (rank + 1)
        if ids:
            pur += ix.chunks[ids[0]].purity(gold)
        ctx += sum(ix.chunks[i].n_words for i in ids)
    n = len(questions)
    return Scores(hits / n, rr / n, pur / n, ctx / n)


METHODS: dict[str, Callable[[Index, str], list[int]]] = {
    "bm25": bm25_search,
    "semantic": semantic_search,
    "hybrid": hybrid_search,
    "hybrid+rerank": lambda ix, q: rerank(ix, q, hybrid_search(ix, q)),
}


def compare(ix: Index, questions: list[dict], k: int = 4) -> dict[str, Scores]:
    return {name: score(ix, questions, fn, k) for name, fn in METHODS.items()}
