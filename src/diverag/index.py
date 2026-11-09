"""BM25 (lexical) and an embedding index (semantic) over chunks.

The default embedder is a dependency-free TF-IDF scorer with slang expansion, so
tests run anywhere. Install the `embed` extra to use a real embedding model:
    pip install -e ".[embed]"   then   Index(chunks, embedder="st")
"""
from __future__ import annotations

import math
from collections import Counter

from .chunk import Chunk
from .text import expand, tokenize


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.N = len(docs)
        self.avgdl = sum(self.lens) / max(1, self.N)
        self.df = Counter(t for d in docs for t in set(d))

    def scores(self, q: list[str]) -> list[float]:
        out = []
        for tf, dl in zip(self.tfs, self.lens):
            s = 0.0
            for t in set(q):
                if t in tf:
                    idf = math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
                    s += idf * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


class TfidfEmbedder:
    """Lexical stand-in for an embedding model: TF-IDF cosine over slang-expanded queries."""

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        docs = [tokenize(t) for t in texts]
        self.N = len(docs)
        self.df = Counter(t for d in docs for t in set(d))
        self.vecs = [self._vec(Counter(d)) for d in docs]
        return self

    def idf(self, t: str) -> float:
        return math.log((self.N + 1) / (self.df.get(t, 0) + 1)) + 1

    def _vec(self, tf: Counter) -> dict[str, float]:
        v = {t: (1 + math.log(c)) * self.idf(t) for t, c in tf.items()}
        n = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / n for t, x in v.items()}

    def scores(self, query: str) -> list[float]:
        q = self._vec(Counter(expand(query)))
        return [sum(w * d.get(t, 0.0) for t, w in q.items()) for d in self.vecs]


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)

    def fit(self, texts: list[str]) -> "SentenceTransformerEmbedder":
        self.emb = self.model.encode(texts, normalize_embeddings=True)
        return self

    def scores(self, query: str) -> list[float]:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        return (self.emb @ q).tolist()


class Index:
    def __init__(self, chunks: list[Chunk], embedder: str = "tfidf"):
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.bm25 = BM25([tokenize(t) for t in texts])
        self.tfidf = TfidfEmbedder().fit(texts)  # also used for confidence weighting
        self.embedder = self.tfidf if embedder == "tfidf" else SentenceTransformerEmbedder().fit(texts)

    def lexical(self, query: str) -> list[float]:
        return self.bm25.scores(tokenize(query))

    def semantic(self, query: str) -> list[float]:
        return self.embedder.scores(query)
