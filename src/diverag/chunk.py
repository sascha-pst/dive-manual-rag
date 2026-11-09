"""Section-aware and fixed-size chunking."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .parse import Section


@dataclass
class Chunk:
    id: int
    text: str
    source_id: str
    revision: str
    sections: Counter = field(default_factory=Counter)  # section -> word count

    @property
    def primary(self) -> str:
        return self.sections.most_common(1)[0][0]

    @property
    def n_words(self) -> int:
        return sum(self.sections.values())

    def purity(self, gold: set[str]) -> float:
        return sum(self.sections[g] for g in gold) / max(1, self.n_words)


def _heading(s: Section) -> str:
    return f"§{s.section} {s.title}."


def section_chunks(sections: list[Section], max_words: int = 90, overlap: int = 10) -> list[Chunk]:
    """One chunk per section; long sections are windowed. Each chunk carries a heading breadcrumb."""
    chunks: list[Chunk] = []
    for s in sections:
        head = _heading(s)
        n_head = len(head.split())
        body = s.text.split()
        room = max(5, max_words - n_head)
        step = max(1, room - overlap)
        i = 0
        while True:
            words = body[i : i + room]
            chunks.append(
                Chunk(len(chunks), f"{head} {' '.join(words)}", s.source_id, s.revision,
                      Counter({s.section: len(words) + n_head}))
            )
            if i + room >= len(body):
                break
            i += step
    return chunks


def fixed_chunks(sections: list[Section], size: int = 90, overlap: int = 10) -> list[Chunk]:
    """Fixed word windows across the whole document, ignoring section boundaries."""
    words = [(w, s) for s in sections for w in f"{_heading(s)} {s.text}".split()]
    step = max(1, size - overlap)
    chunks: list[Chunk] = []
    i = 0
    while True:
        window = words[i : i + size]
        first = window[0][1]
        chunks.append(
            Chunk(len(chunks), " ".join(w for w, _ in window), first.source_id, first.revision,
                  Counter(s.section for _, s in window))
        )
        if i + size >= len(words):
            break
        i += step
    return chunks
