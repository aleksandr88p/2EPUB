# 2EPUB — Project Progress

Last updated: 2026-11-14
Project status: **Experimental MVP v0.1 — implementation complete; validation ongoing**

## Implemented

- EPUB and FB2 parsing, including all four source/translation format
  combinations. EPUB chapters are read from TOC markers, headings, or spine
  fallback; FB2 sections, nested sections, block types and image references
  are represented in the book model.
- Conservative single-section FB2 recovery using repeated spacing and
  compact-heading patterns. Recovered parts and chapters retain hierarchy;
  uncertain candidates are reported without changing the original section.
  Preflight corroborates accepted structure against a normal counterpart
  with the existing chapter DP and does not launch Qwen.
- Source-language-neutral CLI for EN, RU and ES, including all six directed
  language pairs, metadata detection, language overrides, and the fallback
  EN→RU pair when metadata is absent.
- Public commands `check`, `build`, `align-chapters`, and `parse`.
  `build` creates the full source-ordered EPUB by default; `--preview`
  selects the first two aligned pairs.
- Heuristic preflight checks, monotonic chapter alignment, and monotonic
  sentence alignment. Sentence operations include `1:1`, `1:2`, `2:1`,
  `skip_en` and `skip_ru`.
- Qwen3-Embedding-0.6B sentence/group scoring with a persistent disk cache.
- Paragraph reconstruction that preserves source order and validates exact
  sentence-index coverage.
- Optional `--short-paragraphs`: target about 35–40 source words using only
  boundaries between existing `AlignOp`s. Group operations remain atomic;
  translation text is not truncated, and images appear only with the first
  split fragment. A synthetic 400-word case produced blocks of
  30, 30, 50, 60, 30, 30, 50, 60, 30 and 30 words; the 50- and 60-word
  blocks were indivisible operations.
- EPUB generation with source chapter order, navigation/NCX, supported
  source block formatting, and source cover/referenced images from EPUB or
  FB2 when available. Post-write checks verify document order, XHTML, text
  against parsed input, navigation and image references.

## Confirmed full-book conversions

| Book | Inputs and result |
|---|---|
| *The Blade Itself* / «Кровь и железо» | FB2 + FB2; 52 XHTML chapter documents, 48 paired chapters and 4 source-only chapters. Structural and text checks passed. |
| *Lost Gods* | EPUB + FB2; 116 XHTML chapter documents, 110 paired chapters and 6 source-only chapters. Structure, text and image checks passed; preflight had 6 warnings and chapter alignment had 1 doubtful pair. |
| *Yo, robot* / «Я, робот» | `books/01_Yo_robot_Isaac_Asimov.epub` + `books/я робот.fb2`; the user-provided run log confirms successful `--short-paragraphs` conversion to `output/Yo_robot_bilingual.epub` (965,856 bytes). All 9 stories were paired in order. All 14 chapter documents passed the listed post-write checks: 10 paired (the introduction plus 9 stories) and 4 Spanish-only. Preflight reported 6 warnings; 6 chapter pairs were labeled doubtful for position drift. |

The `Yo, robot` build is recorded as successful because the supplied
terminal output includes `Successfully converted`, all post-write checks
as `[OK]`, and exit code 0. These checks confirm consistency with parsed
text; they are not a semantic review.

## Structural recovery validation

`books/Wraiths of the Broken Land.fb2` and
`books/Крэйг С Залер - Духи рваной земли .fb2` were checked without Qwen or
EPUB generation. The English FB2 has 4,862 paragraphs, 8,258 sentences and
91,121 words. It recovered 48 chapter headings and 5 parts from its single
main section, while retaining separate front-matter and author sections.
The Russian book has 5,196 paragraphs, 8,708 sentences and 71,074 words;
its normal structure contains the same 48 chapters and 5 parts. Existing
chapter DP paired all 53 sections confidently, with no unmatched sections
and 100% word coverage on both sides. `check` now returns
`PROCEED WITH CAUTION` (3 source-quality warnings), not `STOP: no-chapters`.

A small synthetic EPUB test confirms nested Part → Chapter links in both
EPUB3 navigation and NCX, with chapter documents in spine order. The full
real-book EPUB has not been built.

## Automated tests

Most recent full test run:

```text
venv/bin/python -m pytest
156 passed in 1.07s
```

The suite covers parsers, structural recovery, preflight, all four format
combinations, all six language directions, alignment, paragraph
reconstruction, CLI forwarding, nested EPUB navigation and verification.
Tests stub model inference; they do not run a full-book conversion.

## Known limitations and remaining work

- Semantic alignment and sentence segmentation remain heuristic. Review
  warnings and doubtful pairs; text can be attached to a neighboring
  paragraph.
- Chapter alignment uses titled main-body chapters and does not merge
  chapters. Sentence alignment supports groups only up to 1:2/2:1.
- FB2 recovery depends on recurring structural evidence and may leave an
  unusual one-section layout unsplit. Cross-book volume/order alignment is
  corroboration, not semantic proof; inline XML styles remain flattened by
  the existing parser.
- Translation paragraph boundaries and some inline formatting are not
  preserved. EPUB TOC/spine differences can affect chapter boundaries and
  service-section classification.
- Short blocks can exceed 40 source words when an operation is indivisible,
  crosses source paragraphs, or belongs to an unsplittable structural block.
- Verification compares generated text with parser output, not raw book
  contents, and does not assess translation meaning.
- An independent semantic QA audit remains incomplete; a previous partial
  review of *The Blade Itself* reported 5 major and 1 minor issue.

Next validation work:

1. Complete and document semantic review of generated books, especially
   doubtful chapter/paragraph alignments.
2. Inspect output in target reading applications, including a 6-inch Kindle.
3. Expand QA across additional book pairs and structural variants.

## Documentation and user data

- `README.md`, `README.ru.md` — concise user guides in English and Russian.
- `docs/ARCHITECTURE.md` — implementation details and limitations.
- `AGENTS.MD` — repository-specific agent workflow and testing rules.

Keep personal books, model files, caches and generated EPUBs in `books/`,
`cache/` and `output/`; these are ignored by Git.