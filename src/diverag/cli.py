"""Command line:  diverag ask "what causes the bends"  |  diverag eval  |  diverag ingest"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .answer import answer
from .chunk import fixed_chunks, section_chunks
from .evaluate import compare, load_questions
from .index import Index
from .parse import load_sample, parse_file
from .sources import load_manifest

PARSED = Path("data/parsed_sections.json")


def load_sections(corpus: str):
    if corpus == "sample":
        return load_sample()
    from .parse import Section

    return [Section(**s) for s in json.loads(PARSED.read_text())]


def build(args) -> Index:
    sections = load_sections(args.corpus)
    chunker = section_chunks if args.chunker == "section" else fixed_chunks
    return Index(chunker(sections, args.size, args.overlap), embedder=args.embedder)


def cmd_ingest(args):
    out = []
    for src in load_manifest(args.manifest):
        path = Path(src.path)
        if not path.exists():
            print(f"skip {src.id}: download it to {path} (from {src.url})")
            continue
        sections = parse_file(path, src.id, src.revision)
        print(f"{src.id}: {len(sections)} sections")
        out += [s.__dict__ for s in sections]
    PARSED.write_text(json.dumps(out, indent=1))
    print(f"wrote {len(out)} sections to {PARSED}")


def cmd_ask(args):
    a = answer(build(args), args.question, k=args.k, use_llm=args.llm)
    print(f"[{a.route}] {a.text}")
    if a.citations:
        print("citations:", ", ".join(a.citations))


def cmd_eval(args):
    ix = build(args)
    print(f"{len(ix.chunks)} chunks ({args.chunker}, size={args.size}, overlap={args.overlap}), k={args.k}\n")
    print(f"{'method':<15}{'recall@k':>10}{'MRR':>8}{'purity':>8}{'ctx words':>11}")
    for name, s in compare(ix, load_questions(args.questions), args.k).items():
        print(f"{name:<15}{s.recall:>10.2f}{s.mrr:>8.2f}{s.purity:>8.2f}{s.context_words:>11.0f}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="diverag")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--corpus", choices=["sample", "parsed"], default="sample")
        sp.add_argument("--chunker", choices=["section", "fixed"], default="section")
        sp.add_argument("--size", type=int, default=90)
        sp.add_argument("--overlap", type=int, default=10)
        sp.add_argument("--embedder", choices=["tfidf", "st"], default="tfidf")
        sp.add_argument("-k", type=int, default=4)

    sp = sub.add_parser("ingest", help="parse sources listed in the manifest")
    sp.add_argument("--manifest", default="sources.toml")
    sp.set_defaults(fn=cmd_ingest)

    sp = sub.add_parser("ask", help="answer one question")
    sp.add_argument("question")
    sp.add_argument("--llm", action="store_true", help="generate with Claude (needs ANTHROPIC_API_KEY)")
    common(sp)
    sp.set_defaults(fn=cmd_ask)

    sp = sub.add_parser("eval", help="compare retrieval methods on the test questions")
    sp.add_argument("--questions", default="eval/questions.jsonl")
    common(sp)
    sp.set_defaults(fn=cmd_eval)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
