# 2EPUB — Project Progress

Last updated: 2026-11-02  
Project status: **Experimental MVP v0.1**

## Implemented and working

- `bookparse` builds the data model `Book → Chapter → Paragraph → Sentence`.
  The FB2 parser preserves block types and image references; the EPUB
  parser extracts structure from the TOC, headings, or spine.
- Chapters are aligned monotonically with DP using text volume, position,
  and nesting level. The result distinguishes paired, doubtful, unmatched,
  and service chapters.
- Sentences are segmented with a rule-based splitter that accounts for
  selected English and Russian abbreviations.
- `experiments/build_preview_epub.py` requires `--en` and `--ru`, accepts
  `--output` and `--all`, and independently selects the FB2 or EPUB parser
  for each input. All four format combinations are supported. Preview
  includes the first two aligned chapter pairs; `--all` processes every
  pair and keeps remaining English chapters without a translation.
- Sentence alignment uses local Qwen3-Embedding-0.6B, a disk cache, and a
  joined-text group scorer. Sentence DP emits `1:1`, `1:2`, `2:1`,
  `skip_en`, and `skip_ru`.
- `rebuild_chapter` distributes Russian sentences over the English
  paragraph structure and validates complete monotonic index coverage.
  Missing EN/RU text is not replaced with invented text.
- The builder writes EPUB navigation, NCX, and spine; carries over the
  English cover and referenced English images from either FB2 or EPUB;
  and checks chapter count/order, XHTML, text against the parsed
  representation, and image references.
- After parsing both books, the builder runs the existing
  `bookparse.checks.build_report()` preflight before chapter alignment or
  loading Qwen. It prints the report, stops with a non-zero exit status
  for ERROR/STOP, and continues with a visible warning summary for
  WARNING. The separate `python -m bookparse check` command is unchanged.
- A full test book, **The Blade Itself / «Кровь и железо»**, was generated:
  52 XHTML sections, 48 paired chapters, and 4 EN-only chapters. The
  generator log reports successful structural and text-completeness checks.

## Implemented separately from the main converter

- `parse_epub` and `bookparse parse/check/align-chapters` also work as
  separate tools.
- `checks.py` contains heuristic source checks and cross-book comparison.
  These are also available through the separate `bookparse check` command.
- `bookparse align-chapters` separately saves a JSON alignment result.
  The builder performs chapter alignment directly.
- `experiments/paragraph_preview.py`,
  `experiments/sentence_alignment_diagnostic.py`,
  `experiments/sentence_match_diagnostic.py`, and
  `experiments/test_embeddings.py` are diagnostic/experimental scripts,
  not stages of the conversion pipeline.

## Known issues and limitations

- Semantic alignment is not guaranteed: a translation may end up under a
  neighboring paragraph. Short dialogue, standalone lines, letters, and
  epigraphs need particular attention; `skip_en`/`skip_ru` require
  contextual interpretation.
- English paragraph structure is retained, but Russian paragraph
  boundaries are not reconstructed. Verse lines may be merged; FB2 inline
  formatting, including `<emphasis>`, is flattened to plain text by the
  parser.
- Chapter DP aligns titled chapters in the main body only; chapter
  merges (one chapter to multiple chapters) are not implemented.
- Sentence DP supports only `1:2` and `2:1` groups, not larger groups.
- Completeness checks compare the EPUB with parser output, not the raw FB2.
  They do not establish semantic correctness.
- A full independent semantic audit of the book is incomplete; a previous
  partial review reported 5 MAJOR and 1 MINOR issues.
- The model runs locally, but initial download and generation can take
  time and require sufficient machine resources.

## Required work to complete the MVP

- Add and run an automated **full CLI end-to-end scenario** covering
  argument handling, preflight, conversion, and result verification.
- Update both `README.md` and `README.ru.md` to document the final,
  verified workflow and supported input combinations.
- Complete and document an independent QA audit of the current bilingual
  EPUB; investigate confirmed alignment errors.
- Evaluate alignment on additional books and translations.
- Improve paragraph-boundary alignment based on QA findings.
- Define MVP acceptance criteria and check generated EPUBs in target
  reading applications.

These items are planned work, not currently available features.

## Tests and verification

Most recent full test run:

```text
venv/bin/python -m pytest
97 passed in 0.99s
```

Tests cover chapter alignment, sentence alignment, parsers, heuristic
checks, cache, paragraph reconstruction, and builder/EPUB verification.
Builder tests exercise all four input format combinations, unsupported
formats and missing paths, plus image/cover extraction. Alignment and
builder tests use controlled test data. Preflight tests cover OK,
WARNING, and ERROR (including stopping before alignment); pytest does
not rerun the full-book conversion.

## Key files

- `experiments/build_preview_epub.py` — EPUB conversion CLI.
- `bookparse/` — parsing, models, segmentation, alignment, cache, and checks.
- `tests/` — automated tests.
- `README.md`, `README.ru.md` — user documentation.
- `docs/ARCHITECTURE.md` — module architecture and contracts.

Personal books, model files, cache, and generated EPUBs belong in `books/`,
`cache/`, and `output/`. User data and generated files are ignored by Git;
`.gitkeep` files retain the input and output directories in the repository.