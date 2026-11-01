# 2EPUB — Architecture

Technical reference for the bilingual EPUB pipeline. Read this before
modifying alignment, reconstruction or EPUB assembly code.

Pipeline overview:

```
EN.fb2 ──┐
         ├─► parse_fb2 ─► Book ─► align_chapters ─► pairs
RU.fb2 ──┘                                       │
                                                 ▼
        sentences ─► embed (cached) ─► sim matrix ─► align_sentences
                                                 │   (AlignOp list)
                                                 ▼
                              rebuild_chapter ─► BilingualParagraphs
                                                 │
                                                 ▼
                            render + ebooklib ─► output/*.epub ─► _verify
```

**Design invariant:** the *English* book is the structural skeleton —
chapter order, paragraph order and paragraph boundaries come from EN.
RU text is distributed onto that skeleton by alignment ops. No
sentence may be lost or duplicated; violations raise `ValueError`,
and the post-write `_verify` re-checks everything against the packaged
EPUB.

---

## Modules (`bookparse/`)

| Module | Role |
|---|---|
| `models.py` | Dataclasses: `Book → Chapter → Paragraph → Sentence`. |
| `fb2_parser.py` | FB2 → `Book` via BeautifulSoup (`lxml-xml`). Each `<section>` → chapter; titles, epigraphs, poems, citations, images, empty lines become typed paragraphs. All `<body>` elements processed (incl. auxiliary bodies like `notes`). |
| `epub_parser.py` | EPUB → `Book` via EbookLib + BeautifulSoup. Chapters delimited by TOC anchors; falls back to headings, then spine files. |
| `segment.py` | Rule-based sentence splitter (EN+RU, abbreviation-aware) and `count_words`. |
| `chapter_alignment.py` | Needleman–Wunsch DP over plot chapters; quality report. |
| `sentence_alignment.py` | DP over sentence sequences → `AlignOp` list; no ML inside. |
| `embedding_cache.py` | Disk cache for embedding vectors. |
| `paragraph_reconstruction.py` | Distributes RU sentences over EN paragraphs → `BilingualParagraph`. |
| `checks.py` | Preflight heuristics on parsed books + cross-book comparison. |
| `cli.py` | `python -m bookparse parse / check / align-chapters`. |

`experiments/build_preview_epub.py` is the only place these are wired
into an actual EPUB; it also owns FB2 `<binary>` extraction (the
parser records image *references* only), XHTML rendering and
post-write verification.

## Data structures

```python
Sentence:  id, order, text
Paragraph: id, order, type, text, source_id, images, sentences
Chapter:   id, order, title, level, source_id, body, paragraphs
Book:      source_file, format, metadata, chapters
```

- `Paragraph.type`: `heading | paragraph | epigraph | cite | subtitle |
  verse | text-author | image | empty-line | quote | table | ...` —
  preserved from the source markup.
- `Chapter.body`: `"main"` or the FB2 `<body name>` (e.g. `notes`).
  Only *titled* main-body chapters participate in chapter alignment.

```python
AlignOp(kind, en_idx, ru_idx, score, en_text, ru_text, quality)
# kind: "1:1" | "1:2" | "2:1" | "skip_en" | "skip_ru"
# quality: "confident" | "doubtful" | "unmatched"

BilingualParagraph(index, type, en_sentences, ru_sentences, ru_op_kinds)
```

`align_chapters()` returns a JSON-ready dict: `pairs` (each with ids,
titles, word counts, `quality`, `reasons`), `unmatched.english /
russian`, `service.english / russian`, `quality` totals.

## Stage I/O

| Stage | Input | Output |
|---|---|---|
| `parse_fb2` | `.fb2` path | `Book` (chapters → paragraphs → sentences) |
| `align_chapters` | `en: Book`, `ru: Book` | dict with `pairs`, `unmatched`, `service` |
| `embed_with_cache` | `encode_fn`, texts, cache dir, model, params | vectors, n_new, seconds |
| `align_sentences` | EN/RU sentences, sim matrix, `group_scorer` | ordered `AlignOp` list |
| `rebuild_chapter` | EN paragraphs (as sentence lists), RU sentences, ops | `BilingualParagraph` list |
| builder | aligned chapters + FB2 binaries | `.epub` file |
| `_verify` | written `.epub` + expected `BilingualParagraph`s | (label, ok) check list |

## Embedding cache (`embedding_cache.py`)

- Key: `sha256(json{model, params, text})[:24]` — encode params that
  affect the vector (`prompt`, `normalize_embeddings`) are part of the
  key, so changing them invalidates entries instead of serving stale
  ones.
- Storage: one `.npy` per key in `cache/embeddings/`; written
  atomically (temp file + `os.replace`), corrupted entries fall back
  to recompute.
- `embed_with_cache(encode_fn, texts, ...)` deduplicates within the
  call, loads hits, batch-encodes only misses, validates the encoder's
  output shape/finiteness, returns vectors in input order.
- The builder uses params `{"prompt": None, "normalize_embeddings": True}`
  — unit-norm vectors, so `E @ R.T` is already the cosine matrix.
- Model weights live in `cache/models/` (`cache_folder` argument),
  downloaded by `SentenceTransformer` on first use.

## Chapter DP (`chapter_alignment.py`)

- Universe: titled main-body chapters only; untitled front matter and
  auxiliary bodies are returned as `service` entries — kept aside,
  never ignored.
- Pair score: `2.0·volume_ratio + 1.0·position_proximity + 0.5·level_match`
  — text volume dominates, then relative position, then nesting level.
- Skip cost: `1.0 + 0.5·min(words/3000, 1)` — large unmatched chapters
  hurt more.
- Quality assessment per pair: `confident` or `doubtful` + reasons
  (`volume-mismatch`, `level-mismatch`, `position-drift`).
- Chapter merges (1:N) are intentionally **not** implemented.

## Sentence DP (`sentence_alignment.py`)

Monotonic alignment, cost-minimizing; every EN and RU index appears in
exactly one op (pair, merge or explicit skip).

| Transition | Cost |
|---|---|
| `1:1` | `1 − sim[i][j]` |
| `1:2` / `2:1` | `2·(1 − group_sim) + group_penalty (0.05)` |
| `skip_*` | `0.35` per sentence |

- Group costs are scaled by the covered sentence count — otherwise
  merges would be undercharged and absorb stray sentences.
- `group_scorer(en_idx, ru_idx)` scores the *joined* group text, not a
  max of member sims. In production it uses embeddings of joined
  adjacent pairs, precomputed in one batch through the same cache;
  `bandwidth=15` restricts group queries to cells near the diagonal,
  bounding scorer calls.
- `confident_min=0.55` splits matches into `confident`/`doubtful`.
- The function is pure: similarity matrix and scorer are injected, no
  model is loaded inside.

## Paragraph reconstruction (`paragraph_reconstruction.py`)

`rebuild_chapter(en_paragraphs, ru_sentences, ops)` →
`BilingualParagraph`s, one per EN paragraph.

- **Contract:** ops must cover every EN and RU index exactly once, in
  order — otherwise `ValueError`. An empty EN chapter with RU text also
  raises `ValueError` (there is nowhere to attach it).
- Matched ops (`1:1`, `1:2`, `2:1`): RU goes to the paragraph of the
  op's *first* EN index — a merge crossing a paragraph boundary lands
  in the earlier paragraph.
- `skip_ru` (translator insertions): attaches to the *next* matched
  paragraph; trailing ones fall back to the last matched paragraph
  (or paragraph 0 if nothing matched at all).
- `skip_en`: the paragraph keeps its EN text, gets no translation —
  rendered bare, never fabricated.
- Chapter-scoped: passing ops from one pair makes cross-chapter
  mixing impossible.

## EPUB assembly & verification

Rendering (`_render_paragraph`) maps EN block types to XHTML:
`heading → <h2>`, `epigraph|cite|quote → <blockquote>`, `image →
<img>`, `empty-line → spacer`, else `<p><strong>…`. RU always renders
as a plain `<p class="ru">` right below, and only when RU text exists.
CSS classes `.en`/`.ru` are both the styling hook and the verification
markers.

Assembly: `epub.EpubBook` + one `EpubHtml` per chapter (`chap_N.xhtml`),
`set_cover` for the FB2 coverpage binary, `EpubNcx` + `EpubNav`,
spine `[cover?, nav, chap_1 … chap_N]`.

`_verify` re-opens the written file and checks:

- the chapter document set equals expectation **as a set** (file-name
  sort would misorder `chap_10+`);
- chapter order follows the **spine**, resolved via idrefs;
- each `chap_*.xhtml` is well-formed XML;
- per chapter, `.en`/`.ru` text equals the expected joined sentences —
  completeness, order and no-duplication in one comparison;
- every `<img src>` resolves to a packaged item;
- nav document and NCX exist.

## Known architectural limitations

- Alignment is **monotonic**: reordering sentences/chapters is
  impossible by design.
- Chapter-level merges (1 EN : N RU) are not implemented.
- RU paragraph structure is discarded — RU sentences re-flow into EN
  paragraph shapes (verse lines merge).
- Inline formatting (`<emphasis>`) is flattened at parse time; the
  parsed model holds plain text only.
- The builder extracts binaries from **FB2** only; an EPUB source
  would lose its images/cover silently.
- The group scorer covers only *adjacent* pairs — groups larger than
  2:1/1:2 are impossible.
- The builder's CLI covers `--en/--ru/--output/--all`; model and
  alignment parameters remain fixed constants in the script.
- Verification compares EPUB text to *parser output*, not to the raw
  FB2 — parser-level losses are invisible to it.
