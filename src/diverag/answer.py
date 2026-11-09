"""Route the question, then answer: computed lookup, referral, refusal, or grounded answer."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from .chunk import Chunk
from .index import Index
from .retrieve import search
from .text import expand, tokenize

ADVICE = re.compile(
    r"\b(can i|should i|is it safe|am i ok|my dive|my dives|for me|plan (my|a) dive|how long should i)\b", re.I
)
DEPTH = re.compile(r"(\d+(?:\.\d+)?)\s*(fsw|feet|ft|msw|meters?|m)\b", re.I)
PRESSURE_WORDS = re.compile(r"\b(pressure|ata|atm|po2|partial)\b", re.I)
PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(%|percent)", re.I)
CONFIDENCE_THRESHOLD = 0.35

SYSTEM_PROMPT = """You answer questions about diving using ONLY the provided context from approved manuals.
- Cite every claim as [§section, source revision].
- If the context does not answer the question, reply exactly: "Not covered by the approved sources."
- Never compute decompression schedules, no-decompression limits or flying intervals.
- Never give personal dive-planning or medical advice; point to the cited sections and a qualified professional."""


@dataclass
class Answer:
    route: str  # lookup | advice | refuse | explain
    text: str
    citations: list[str] = field(default_factory=list)
    context: list[Chunk] = field(default_factory=list)


def weighted_coverage(ix: Index, chunk: Chunk, query: str) -> float:
    """Share of the query's IDF weight found in the chunk. Rare words count more."""
    ct = set(tokenize(chunk.text))
    terms = set(expand(query))
    total = sum(ix.tfidf.idf(t) for t in terms)
    hit = sum(ix.tfidf.idf(t) for t in terms if t in ct)
    return hit / total if total else 0.0


def table_lookup(q: str) -> Answer | None:
    m = DEPTH.search(q)
    if not (m and PRESSURE_WORDS.search(q)):
        return None
    depth, unit = float(m[1]), m[2].lower()
    metric = unit in {"msw", "m", "meter", "meters"}
    ata = depth / (10 if metric else 33) + 1
    text = f"At {depth:g} {'msw' if metric else 'fsw'}, absolute pressure = {ata:.2f} ata."
    cites = ["§2-1"]
    if re.search(r"\b(po2|partial)\b", q, re.I):
        pct = PERCENT.search(q)
        fo2 = float(pct[1]) / 100 if pct else 0.21
        text += f" PO2 = {fo2:.2f} × {ata:.2f} = {fo2 * ata:.2f} ata" + ("" if pct else " (air, 21% O2)") + "."
        cites.append("§2-3")
    return Answer("lookup", text + " Computed, not generated.", cites)


def extractive(q: str, ctx: list[Chunk]) -> tuple[str, list[str]]:
    qs = set(expand(q))
    cands = []
    for c in ctx:
        body = c.text.split(".", 1)[1] if c.text.startswith("§") else c.text
        for s in re.findall(r"[^.]+\.", body):
            hits = len(qs & set(tokenize(s)))
            if hits:
                cands.append((hits, s.strip(), c.primary))
    cands.sort(key=lambda x: -x[0])
    seen, picks = set(), []
    for _, s, sec in cands:
        if s not in seen:
            seen.add(s)
            picks.append((s, sec))
        if len(picks) == 2:
            break
    return " ".join(f"{s} [§{sec}]" for s, sec in picks), [f"§{sec}" for _, sec in picks]


def llm_answer(q: str, ctx: list[Chunk]) -> str:
    import anthropic

    context = "\n\n".join(f"[§{c.primary}, {c.source_id} {c.revision}]\n{c.text}" for c in ctx)
    msg = anthropic.Anthropic().messages.create(
        model=os.environ.get("DIVERAG_MODEL", "claude-sonnet-5-5"),
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q}"}],
    )
    return msg.content[0].text


def answer(ix: Index, q: str, k: int = 4, use_llm: bool = False) -> Answer:
    ctx = search(ix, q, k=k)
    if ADVICE.search(q):
        top = weighted_coverage(ix, ctx[0], q) if ctx else 0
        secs = list(dict.fromkeys(f"§{c.primary}" for c in ctx if weighted_coverage(ix, c, q) >= 0.5 * top))
        return Answer(
            "advice",
            "This assistant doesn't give personal dive-planning or medical advice. "
            f"Relevant sections: {', '.join(secs) or 'none found'}. "
            "Use your governing procedures and talk to your dive supervisor or a diving medicine physician.",
            secs, ctx,
        )
    if (lookup := table_lookup(q)) is not None:
        return lookup
    if not ctx or weighted_coverage(ix, ctx[0], q) < CONFIDENCE_THRESHOLD:
        return Answer("refuse", "Not covered by the approved sources.", [], ctx)
    if use_llm:
        return Answer("explain", llm_answer(q, ctx), [f"§{c.primary}" for c in ctx], ctx)
    text, cites = extractive(q, ctx)
    return Answer("explain", text or "Not covered by the approved sources.", cites, ctx)
