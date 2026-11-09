"""Turn ePub/PDF files into clean text split into numbered sections.

ePub and PDF support are optional extras:  pip install -e ".[parse]"
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Manual headings look like "3-4 DECOMPRESSION SICKNESS" or "9-8.2 Ascent Rate".
SECTION_RE = re.compile(r"^(\d{1,2}-\d{1,3}(?:\.\d+)*)\s+([A-Z][^\n]{2,120})$", re.M)


@dataclass
class Section:
    source_id: str
    revision: str
    section: str
    title: str
    text: str


# ---------- cleaning ----------
def strip_boilerplate(pages: list[str], threshold: float = 0.5, edge: int = 2) -> list[str]:
    """Drop header/footer lines: among the first and last `edge` lines of each page,
    those that repeat on more than `threshold` of pages. Body lines are never touched."""

    def edges(p: str) -> list[str]:
        lines = [l.strip() for l in p.splitlines() if l.strip()]
        return lines[:edge] + lines[-edge:]

    freq = Counter(l for p in pages for l in set(edges(p)))
    boiler = {l for l, c in freq.items() if c > threshold * len(pages)}
    cleaned = []
    for p in pages:
        lines = [l for l in p.splitlines() if l.strip()]
        head = [l for l in lines[:edge] if l.strip() not in boiler]
        mid = lines[edge:-edge] if len(lines) > 2 * edge else []
        tail = [l for l in lines[-edge:] if l.strip() not in boiler] if len(lines) > edge else []
        cleaned.append("\n".join(head + mid + tail))
    return cleaned


def dehyphenate(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def join_wrapped_lines(text: str) -> str:
    """Join single newlines inside paragraphs; keep blank lines and headings."""
    out = []
    for para in re.split(r"\n\s*\n", text):
        lines = para.splitlines()
        buf = []
        for line in lines:
            if SECTION_RE.match(line.strip()):
                if buf:
                    out.append(" ".join(buf))
                    buf = []
                out.append(line.strip())
            else:
                buf.append(line.strip())
        if buf:
            out.append(" ".join(buf))
    return "\n".join(x for x in out if x)


def clean_pages(pages: list[str]) -> str:
    return join_wrapped_lines(dehyphenate("\n".join(strip_boilerplate(pages))))


# ---------- readers ----------
def read_pdf(path: Path) -> list[str]:
    import fitz  # pymupdf

    with fitz.open(path) as doc:
        return [page.get_text() for page in doc]


def read_epub(path: Path) -> list[str]:
    import ebooklib
    from bs4 import BeautifulSoup
    from ebooklib import epub

    book = epub.read_epub(str(path))
    return [
        BeautifulSoup(item.get_content(), "html.parser").get_text("\n")
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
    ]


def extract_tables(path: Path) -> list[list[list[str]]]:
    """Tables as rows, for exact lookups. Never embed decompression tables as prose."""
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        return [t for page in pdf.pages for t in page.extract_tables()]


# ---------- sectioning ----------
def split_sections(text: str, source_id: str, revision: str) -> list[Section]:
    marks = list(SECTION_RE.finditer(text))
    sections = []
    for m, nxt in zip(marks, marks[1:] + [None]):
        body = text[m.end() : nxt.start() if nxt else None].strip()
        if body:
            sections.append(Section(source_id, revision, m[1], m[2].strip().title(), body))
    return sections


def parse_file(path: str | Path, source_id: str, revision: str) -> list[Section]:
    path = Path(path)
    if path.suffix.lower() == ".epub":
        pages = read_epub(path)
    elif path.suffix.lower() == ".pdf":
        pages = read_pdf(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return split_sections(clean_pages(pages), source_id, revision)


def load_sample(path: str | Path = "data/sample_corpus.json") -> list[Section]:
    data = json.loads(Path(path).read_text())
    return [
        Section(data["source_id"], data["revision"], s["section"], s["title"], s["text"])
        for s in data["sections"]
    ]
