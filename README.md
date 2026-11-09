# dive-manual-rag

Retrieval-augmented question answering over **approved, public diving manuals**: the US Navy Diving Manual, NOAA's Diving Standards and Safety Manual, and the NPS diving field manual.

The pipeline gates sources on license, parses PDFs and ePubs into numbered sections, chunks by section, runs hybrid retrieval (BM25 plus embeddings, fused with RRF), reranks, routes each question, and answers with section citations. An evaluation harness scores every configuration the same way.

There is also an **interactive explorer** (`explorer/index.html`): open it in a browser to watch each stage work and change the settings live.

> ⚠️ This is a learning and portfolio project, not diving guidance. It refuses personal dive-planning and medical questions by design.

## Quick start

```bash
pip install -e ".[dev]"
pytest -q                                    # 12 tests, runs on the bundled sample corpus
diverag eval                                 # compare retrieval methods
diverag eval --chunker fixed                 # see what fixed-size chunking does
diverag ask "what causes the bends"
diverag ask "PO2 of 32% oxygen at 99 fsw"    # computed, not generated
diverag ask "can I fly 6 hours after my dive"  # referred, not advised
```

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| 1. Sources | `sources.py`, `sources.toml` | Manifest of approved sources. Ingestion fails if any license isn't allow-listed. |
| 2. Parse | `parse.py` | ePub/PDF → text. Strips repeated headers and footers, rejoins hyphenated words, splits on section numbers (`3-4`, `9-8.2`). Tables are extracted as rows. |
| 3. Chunk | `chunk.py` | Section-aware chunks with a `§section title` breadcrumb. Fixed-size chunking is included for comparison. |
| 4. Retrieve | `index.py`, `retrieve.py` | BM25 plus a semantic scorer, merged with reciprocal rank fusion. |
| 5. Rerank | `retrieve.py` | Heuristic reranker by default, or any cross-encoder (`rerank(..., model="BAAI/bge-reranker-base")`). |
| 6. Answer | `answer.py` | Routes to: **lookup** (depth/pressure math is computed), **advice** (cite and refer), **refuse** (low IDF-weighted coverage), or **explain** (grounded, cited answer; extractive by default, Claude with `--llm`). |
| 7. Evaluate | `evaluate.py`, `eval/questions.jsonl` | Recall@k, MRR, top-1 purity, and context size for each retrieval method. |

## Results on the sample corpus (k = 4)

| Method | Recall@4 | MRR | Top-1 purity | Context words |
|---|---|---|---|---|
| BM25 | 0.88 | 0.84 | 0.81 | 176 |
| Semantic | 1.00 | 1.00 | 1.00 | 206 |
| Hybrid (RRF) | 1.00 | 0.97 | 0.94 | 207 |
| Hybrid + rerank | 1.00 | 1.00 | 1.00 | 207 |

Switching to fixed-size chunks drops top-1 purity to about 0.54 for every method, because chunks straddle section boundaries.

**Read these numbers skeptically.** The default "semantic" scorer is TF-IDF with a diver-slang synonym map (`text.py`), and I wrote that map and the test questions myself. That leaks test knowledge into the system, so its perfect score is optimistic. The meaningful findings are the *relative* ones: BM25 misses slang ("tipsy", "airplane"), and fixed chunking hurts purity. Next steps are a real embedding model (`--embedder st`) and held-out questions written by someone else.

## Using the real manuals

The manuals are not committed. Download them from the URLs in `sources.toml` into `data/raw/`, then:

```bash
pip install -e ".[parse,embed]"
diverag ingest                       # writes data/parsed_sections.json
diverag eval --corpus parsed --embedder st
diverag ask "symptoms of oxygen toxicity" --corpus parsed --embedder st
```

For generated answers, install `.[llm]`, set `ANTHROPIC_API_KEY`, and add `--llm`. The model defaults to `claude-sonnet-5-5`; override it with `DIVERAG_MODEL`.

## Source licensing

| Source | Status |
|---|---|
| US Navy Diving Manual Rev 7 Change A, NAVSEA contaminated-water guidance | Released to the public by NAVSEA |
| NOAA Diving Standards & Safety Manual, NPS FM-4 | US government works |
| NOAA *Diving Manual* (commercial edition), PADI/SSI/DAN material | Copyrighted: not ingested without written permission |

The sample corpus in `data/sample_corpus.json` was written for this project. It paraphrases general diving physics and physiology and is not quoted from any manual.

## Layout

```
src/diverag/   sources · parse · chunk · index · retrieve · answer · evaluate · cli
data/          sample_corpus.json, raw/ (gitignored manuals)
eval/          questions.jsonl (16 questions with gold sections)
explorer/      interactive single-page walkthrough
tests/         pytest suite, run in CI on Python 3.11 and 3.12
```
