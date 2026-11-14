# 2EPUB — Architecture

This document describes the implemented pipeline. User commands and setup
are in [README.md](../README.md) and [README.ru.md](../README.ru.md).

## Supported inputs

- Each input is independently `.epub` or `.fb2`; all four format
  combinations are supported.
- Validated languages are English (`en`), Russian (`ru`), and Spanish
  (`es`) in all six directed pairs.
- Language comes from EPUB/FB2 metadata unless overridden. Aliases and
  regional tags normalize to the base language. Missing metadata uses the
  compatibility fallback EN→RU.
- Sentence segmentation has language-specific abbreviation rules for the
  three supported languages; it remains heuristic.

## Pipeline

```text
SOURCE + TRANSLATION
       │
       ├─ parse_epub / parse_fb2 ─► Book models
       ├─ preflight checks ───────► INFO / WARNING / ERROR
       ├─ align_chapters ─────────► chapter pairs and unmatched sections
       ├─ Qwen embeddings/cache ──► similarity matrices
       ├─ align_sentences ────────► ordered AlignOp operations
       ├─ rebuild_chapter ────────► source-shaped bilingual paragraphs
       └─ EPUB render/write ──────► post-write verification
```

The source book defines output chapter order and paragraph/block structure.
The translation is attached to source content by alignment operations.
Operations must cover both sentence streams exactly once and in order;
reconstruction raises `ValueError` if that invariant is broken.

## Modules and data model

| Module | Responsibility |
|---|---|
| `models.py` | `Book → Chapter → Paragraph → Sentence` data model. |
| `epub_parser.py` | Reads EPUB spine XHTML, TOC markers, headings, paragraphs and image references. |
| `fb2_parser.py` | Reads FB2 bodies, nested sections, typed blocks and image references. |
| `languages.py`, `segment.py` | Language normalization, metadata fallback, sentence splitting and word counting. |
| `checks.py` | Heuristic single-book and cross-book preflight checks. |
| `chapter_alignment.py` | Monotonic chapter pairing and quality report. |
| `sentence_alignment.py` | Pure DP over sentence similarity values; returns `AlignOp`s and does not load a model. |
| `embedding_cache.py` | Persistent cache for sentence and joined-text vectors. |
| `paragraph_reconstruction.py` | Assigns translation sentences and optionally splits source blocks. |
| `cli.py` | Public `python -m bookparse` commands: `check`, `build`, `align-chapters`, `parse`. |
| `experiments/build_preview_epub.py` | Model wiring, asset extraction, XHTML/EPUB generation and verification. |

Core records:

```text
Book:      source_file, format, metadata, chapters
Chapter:   id, order, title, level, source_id, body, paragraphs
Paragraph: id, order, type, text, source_id, images, sentences
Sentence:  id, order, text
AlignOp:   kind, en_idx, ru_idx, score, en_text, ru_text, quality
```

`Chapter.body` is `"main"` or an FB2 body name such as `notes`.
`Paragraph.images` contains references, not image bytes. Internal field names
`en`/`ru` are historical; the public pipeline is source-language neutral.

## Parsing

### EPUB

`parse_epub` reads linear XHTML documents from the spine with EbookLib and
parses their block content with BeautifulSoup. The EbookLib TOC is flattened
into chapter markers; anchors can start multiple chapters inside one XHTML
file. If there are no TOC markers, `h1`–`h3` headings are used when present;
otherwise each spine document starts a chapter.

When any TOC markers exist, they take precedence over heading fallback.
Spine documents without matching TOC markers do not necessarily start a new
chapter and may be attached to the preceding chapter. EPUB service/front
matter is not always reliably classified by its TOC or markup.

Paragraph text, block type, IDs and local image references are extracted.
Scripts and styles are skipped; inline formatting is flattened to plain
text. Chapters are assigned `body="main"`.

### FB2

`parse_fb2` processes every `<body>`. Each `<section>` becomes a chapter;
nested sections increase `level`, and `<body name>` is recorded as the
chapter body. Section titles, epigraphs, citations, poems/verse, subtitles,
images, empty lines, tables and text-author blocks are represented by block
types. Stray body-level content is retained in an untitled chapter.

FB2 image references are recorded during parsing. The builder later decodes
referenced `<binary>` data and the cover-page reference. Inline XML styling
is not retained in the text model.

## CLI and build modes

| Command | Behavior |
|---|---|
| `python -m bookparse check SOURCE TRANSLATION` | Parse both books and print preflight results; `--output` optionally saves the report. |
| `python -m bookparse build SOURCE TRANSLATION` | Run preflight and build the full source-ordered EPUB; default output `output/2epub.epub`. |
| `build ... --preview` | Emit the first two aligned chapter pairs only. |
| `build ... --short-paragraphs` | Apply source-word block segmentation after sentence alignment. |
| `build ... --output PATH` | Set the output EPUB path. |
| `check`, `build`, `align-chapters` | Accept `--source-lang` and `--translation-lang`. |
| `python -m bookparse align-chapters SOURCE TRANSLATION` | Save chapter alignment JSON under `cache/` by default. |
| `python -m bookparse parse BOOK` | Save one parsed book as JSON; `--lang` overrides segmentation language. |

The public `build` command defaults to a full book. It passes the internal
builder's `--all` option; invoking `experiments/build_preview_epub.py`
directly defaults to its two-pair preview mode. Build preflight runs before
chapter/sentence alignment and before model loading. `ERROR` stops the build;
warnings are printed and conversion continues.

## Chapter alignment

Only chapters with a title and `body == "main"` participate in chapter DP.
Untitled main sections and auxiliary bodies are returned as service entries.
The public alignment is monotonic: it pairs one chapter at a time or leaves
one unmatched; chapter merges are not implemented.

The pair score combines normalized word-volume similarity, relative book
position and nesting level:

```text
2.0 × volume_ratio + 1.0 × position_proximity + 0.5 × level_match
```

An unmatched chapter has cost `1.0 + 0.5 × min(words / 3000, 1)`.
Pairs are labeled `confident` or `doubtful` with structural reasons such as
`volume-mismatch`, `level-mismatch` or `position-drift`. These labels are
not semantic validation.

In full build mode, every source chapter is emitted in source order.
Unpaired source chapters remain source-only; unmatched translation chapters
are reported but are not added to the source-defined EPUB.

## Sentence alignment and Qwen cache

The builder loads `Qwen/Qwen3-Embedding-0.6B` through SentenceTransformers
when at least one chapter pair will be processed. It uses MPS when available,
otherwise CPU. The model is downloaded on first use to `cache/models/`.
Preflight `ERROR` exits before model loading.

`embed_with_cache` keys each vector by a truncated SHA-256 of model name,
vector-affecting parameters and text. It stores `.npy` files in
`cache/embeddings/`, writes atomically, deduplicates repeated texts in a
call, validates encoded arrays, and recomputes unreadable entries. The
builder normalizes vectors; pairwise dot products are cosine similarities.
Joined adjacent sentence pairs used by the group scorer use the same cache.

`align_sentences` is a model-free monotonic DP that receives the similarity
matrix and group scorer. It emits ordered operations:

| Operation | Meaning |
|---|---|
| `1:1` | One source sentence to one translation sentence. |
| `1:2`, `2:1` | One sentence grouped with two adjacent sentences on the other side. |
| `skip_en`, `skip_ru` | Explicit unmatched source or translation sentence. |

The current DP supports groups only up to 1:2/2:1. Its costs are
`1 - similarity` for 1:1, `2 × (1 - group_similarity) + 0.05` for groups,
and `0.35` per skipped sentence. Group scoring is limited to a 15-cell
diagonal bandwidth. A score threshold of `0.55` separates confident and
doubtful matches. Every sentence index must occur in exactly one operation.

## Paragraph reconstruction and short blocks

`rebuild_chapter` assigns translation sentences to the source paragraph
containing the first source index of each matched operation. A 2:1 operation
crossing a source paragraph boundary is attached to the earlier paragraph.
`skip_ru` text attaches to the next matched paragraph, falling back to the
last matched paragraph at the end; `skip_en` leaves source text without an
invented translation. Translation order follows its original stream.

With `short_paragraphs=True`, segmentation runs on the already computed
`AlignOp` list; sentence alignment is not repeated. Word counts include only
source text and use the language-neutral Unicode word counter. Candidate
boundaries are between complete operations. A partition optimizer targets
the midpoint of 35–40 words (37.5) while keeping ordinary groups at or below
40 words; when no boundary lands in the target interval, it chooses the
lowest-deviation safe partition.

`1:2` and `2:1` operations are indivisible. Operations over 40 source words
remain whole and may form longer blocks. Paragraphs touched by an operation
crossing source paragraph boundaries, plus headings, images, empty lines,
subtitles, verse, tables and preformatted blocks, are not split. The
translation is never truncated to fit. If a source paragraph is split,
source images are included only with its first fragment.

## EPUB rendering, assets and verification

Source prose is bold, while source headings retain heading markup. Its
aligned translation is a plain paragraph immediately below when available.
Quote-like blocks render as blockquotes; supported source block types are
retained. Output language metadata and XHTML `lang` attributes identify
both languages.

The builder can copy the source cover and referenced local images from
either FB2 binaries or EPUB image items. Translation images are not copied.
It creates one XHTML document per emitted source chapter, an EPUB navigation
document and NCX, and a spine in source order. Chapter titles populate the
generated navigation.

After writing, `_verify` reopens the EPUB and checks the expected chapter
set and spine order, XHTML well-formedness, source/translation text against
the parsed expected sentences, image targets and presence of navigation and
NCX. These checks establish consistency with parser output, not completeness
against the raw books or semantic translation quality.

## Technical limitations

- Supported languages are limited to EN/RU/ES; Qwen being multilingual does
  not validate other languages in this converter.
- Sentence splitting and alignment are heuristic. Monotonic alignment
  cannot reorder content; chapter merges and sentence groups larger than
  1:2/2:1 are unsupported.
- Only titled main-body chapters enter chapter alignment. EPUB TOC/spine
  differences can cause service documents to be combined with a neighboring
  chapter or treated as ordinary main content.
- Translation paragraph structure is discarded. Verse or other blocks may
  be flattened, and inline formatting such as italics is not preserved.
- Short blocks may exceed 40 source words when a group is indivisible,
  cross-paragraph, or belongs to an unsplittable structural block.
- Verification cannot detect text lost during parsing or determine whether
  each translation sentence is semantically correct.
