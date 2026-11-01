# 2EPUB

[English](README.md) | [Русский](README.ru.md)

**Local Python CLI that merges an English original book and its Russian literary translation into a single bilingual EPUB with parallel text.**

Each English paragraph is rendered **in bold**, with the Russian translation directly beneath it in normal font — designed for comfortable reading in English while keeping the translation always at hand.

> Status: **MVP v0.1** — a working, experimental command-line converter.
> Long-term this is meant to become one feature of a larger application
> for reading and language learning (see [Roadmap](#roadmap)).

---

## Table of contents

- [Current status](#current-status)
- [Quick start (macOS)](#quick-start-macos)
- [Working with books](#working-with-books)
- [Commands](#commands)
- [Project structure](#project-structure)
- [How it works](#how-it-works)
- [Known limitations](#known-limitations-mvp-v01)
- [First test results](#first-test-results)
- [Roadmap](#roadmap)
- [Technical documentation](#technical-documentation)

## Current status

Implemented and verified end to end on a real book:

- FB2 source parsing (chapters, typed paragraphs, sentences).
- Chapter-level alignment via dynamic programming.
- Sentence segmentation for English and Russian.
- Multilingual embeddings with **Qwen3-Embedding-0.6B** (Apple Silicon MPS).
- Disk cache for embeddings — reruns skip model inference.
- Sentence-level DP alignment with ops `1:1`, `1:2`, `2:1`, `skip_en`, `skip_ru`.
- Reconstruction of bilingual paragraphs (EN paragraph structure is authoritative).
- EPUB generation with cover, navigation and per-block styling.
- EN chapters without an RU counterpart are kept (EN-only).
- Automatic post-write checks: structure, reading order, text completeness.
- Two modes: **preview** (first 2 chapter pairs) and **full book** (`--all`).

## Quick start (macOS)

Requires Python — developed and verified on **Python 3.14** (Apple Silicon); other recent versions are untested.

```bash
git clone git@github.com:aleksandr88p/2EPUB.git
cd 2EPUB
```

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The Qwen3-Embedding-0.6B model (~1 GB) is **downloaded automatically**
from Hugging Face into `cache/models/` on first run — no separate setup
step, but the first generation needs an internet connection.

Sanity check (optional but recommended, ~1 s, no model needed):

```bash
venv/bin/python -m pytest
```

Preflight check of a book pair (optional, reports structural issues):

```bash
venv/bin/python -m bookparse check "books/<english>.fb2" "books/<russian>.fb2"
```

Generate the preview (first two aligned chapter pairs):

```bash
venv/bin/python experiments/build_preview_epub.py
```

Generate the full book:

```bash
venv/bin/python experiments/build_preview_epub.py --all
```

Output lands in `output/`:

| Mode    | Command adds | Output file                 |
|---------|--------------|-----------------------------|
| preview | —            | `output/2epub_preview.epub` |
| full    | `--all`      | `output/2epub_full.epub`    |

> The first run is slow: the model computes embeddings for every
> sentence of the book. Results are cached in `cache/embeddings/`, so
> rerunning the same book pair is much faster.

## Working with books

- Input books live in `books/` (gitignored — books are not committed).
- Supported input format for the builder: **FB2** (both EPUB and FB2 are
  supported by the `bookparse` CLI for parsing/checking).
- **Important:** the builder script does *not* detect languages and does
  *not* accept book paths via CLI. The English and Russian files are
  hardcoded as constants at the top of `experiments/build_preview_epub.py`:

  ```python
  EN_PATH = PROJECT_ROOT / "books" / "blade itslef.fb2"
  RU_PATH = PROJECT_ROOT / "books" / "1 Krov i zhelezo.fb2"
  ```

  To convert a different book pair, put both `.fb2` files into `books/`
  and edit `EN_PATH` / `RU_PATH`. Whichever file `EN_PATH` points to is
  treated as English and defines the structure; `RU_PATH` is the
  translation.

- Output file names are likewise constants (`PREVIEW_PATH`,
  `FULL_PATH`). There is **no `--output` flag** yet.
- Supported CLI flags today: `--all` and `-h/--help`. Nothing else.

## Commands

```bash
# preview EPUB (first 2 aligned chapter pairs)
venv/bin/python experiments/build_preview_epub.py

# full-book EPUB (all pairs + EN-only chapters)
venv/bin/python experiments/build_preview_epub.py --all

# builder help
venv/bin/python experiments/build_preview_epub.py --help

# full test suite
venv/bin/python -m pytest

# inspect a single book's structure -> cache/<name>.json
venv/bin/python -m bookparse parse "books/book.fb2"

# preflight structural report on a book pair
venv/bin/python -m bookparse check "books/en.fb2" "books/ru.fb2"

# chapter alignment result -> cache/align_<en>--<ru>.json
venv/bin/python -m bookparse align-chapters "books/en.fb2" "books/ru.fb2"
```

## Project structure

```
2EPUB/
├── bookparse/                  # the library: parsing → alignment → checks
│   ├── models.py               # Book / Chapter / Paragraph / Sentence
│   ├── fb2_parser.py           # FB2 structure extraction (BeautifulSoup)
│   ├── epub_parser.py          # EPUB structure extraction (EbookLib)
│   ├── segment.py              # sentence splitting, word counting
│   ├── chapter_alignment.py    # DP chapter alignment + quality report
│   ├── sentence_alignment.py   # DP sentence alignment (AlignOp list)
│   ├── paragraph_reconstruction.py  # EN paragraphs + attached RU text
│   ├── embedding_cache.py      # per-sentence .npy embedding cache
│   ├── checks.py               # preflight structural checks
│   └── cli.py                  # `python -m bookparse ...`
├── experiments/                # runnable scripts / diagnostics
│   ├── build_preview_epub.py   # ★ the EPUB generator (preview + --all)
│   ├── paragraph_preview.py    # bilingual paragraph preview -> txt
│   ├── sentence_alignment_diagnostic.py
│   ├── sentence_match_diagnostic.py
│   └── test_embeddings.py
├── tests/                      # pytest suite
├── books/                      # input FB2 files        (gitignored)
├── cache/
│   ├── models/                 # HF model weights       (gitignored)
│   ├── embeddings/             # .npy vectors per text  (gitignored)
│   └── *.json                  # parsed books, alignment results
├── output/                     # generated EPUBs        (gitignored)
├── docs/ARCHITECTURE.md        # technical documentation
├── requirements.txt
└── pytest.ini
```

- `books/` — *your* input files. Nothing is required to be committed here.
- `bookparse/` — the reusable library: everything except the final EPUB
  assembly. Pure data structures in, data structures out.
- `experiments/` — executable scripts that wire the library together,
  including the actual EPUB builder.
- `cache/` — everything expensive: model weights and per-sentence
  embedding vectors, so reruns don't redo inference.
- `output/` — final artifacts (EPUB) and diagnostic reports.
- `tests/` — unit tests for parsing, alignment, reconstruction, cache
  and the EPUB builder.

## How it works

```
EN.fb2 ──┐                          ┌── embeddings (Qwen, cached)
         ├─► parse ─► chapters ─► sentences ─► DP sentence alignment
RU.fb2 ──┘        └─► DP chapter alignment ─┘          │
                                                       ▼
EPUB ◄─ ebooklib ◄─ bilingual paragraphs ◄─ paragraph reconstruction
```

1. **Parse.** Both FB2 files become `Book` objects: chapters → typed
   paragraphs → sentences.
2. **Chapter alignment.** A DP pass (Needleman–Wunsch style) pairs
   titled main-body chapters using text volume, position in the book
   and nesting level. Unpaired/service chapters are kept aside, never
   silently dropped.
3. **Sentence alignment.** For each chapter pair, sentence embeddings
   (Qwen3-Embedding-0.6B, unit-norm) give a similarity matrix; a second
   DP pass produces monotonic ops: `1:1` pairs, `1:2`/`2:1` merges
   (scored on the *joined* text of the group) and explicit
   `skip_en`/`skip_ru`. Embeddings are cached on disk.
4. **Paragraph reconstruction.** RU sentences are distributed over the
   EN paragraph skeleton. The EN book defines structure and order —
   nothing is lost or duplicated (verified by contract).
5. **EPUB assembly.** EN paragraph in **bold**, RU translation below in
   normal font; headings, epigraphs and the EN cover are preserved;
   unpaired EN chapters ship as EN-only sections.
6. **Verification.** The written EPUB is re-opened and checked: exact
   chapter set, spine order, well-formed XHTML, complete/ordered
   EN + RU text per chapter, resolvable image references.

Why embeddings + DP: embedding similarity alone gives pairwise scores,
but choosing the best *global* monotonic mapping (with merges and
skips) is exactly what dynamic programming solves cheaply and
deterministically.

## Known limitations (MVP v0.1)

Working but experimental — a correct result is not guaranteed:

- Alignment is not perfect: a Russian sentence can land under the
  *neighboring* English paragraph. Short replies, standalone lines,
  letters and epigraphs are the hardest cases.
- `skip_en` / `skip_ru` ops need careful interpretation; a small number
  of EN paragraphs legitimately end up without translation.
- EN chapters without an RU counterpart are included **without**
  translation (front matter, `Acknowledgments`).
- RU chapters without an EN counterpart (notes, front matter) are **not**
  included in the EPUB — they are listed in the build report.
- FB2 formatting is only partially preserved: italics (`<emphasis>`)
  are flattened to plain text, and RU verse lines merge into a single
  paragraph following the EN paragraph structure.
- Text completeness is verified **against the parser output**, not the
  raw FB2 — a loss during parsing itself would not be caught.
- The first book passed a *partial* semantic review; a full independent
  audit is still pending.
- Book paths and output names are hardcoded in
  `experiments/build_preview_epub.py` (see
  [Working with books](#working-with-books)).

## First test results

First full test book: **The Blade Itself / «Кровь и железо»**
(Joe Abercrombie).

| Metric                                  | Result |
|-----------------------------------------|--------|
| XHTML sections in EPUB                  | 52     |
| English text blocks                     | 5 056  |
| Russian text blocks                     | 5 033  |
| EN blocks without Russian text          | 23     |
| Confirmed problems in partial review    | 5 MAJOR + 1 MINOR |

These numbers describe this specific book/translation pair only — they
do not characterize the algorithm's accuracy in general.

## Roadmap

### Converter improvements

- Higher alignment quality; fix paragraph-boundary shifts.
- Better formatting preservation (italics, verse lines).
- Arbitrary input files and output names via CLI (`--en`, `--ru`,
  `--output`).
- Richer QA reports per book.
- Testing across different books and translations.

### The bigger application (plans, not features)

- Own web reader UI.
- Backend for uploading and storing books.
- Reading English with parallel translation; show/hide translation.
- LLM integration for translation.
- Text simplification to the reader's level (e.g. B2 → B1).
- Long term: a commercial service.

## Technical documentation

Internals — modules, data structures, pipeline I/O, cache design, the
DP cost model and integrity guarantees — are documented in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
