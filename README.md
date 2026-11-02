# 2EPUB

[English](README.md) | [Русский](README.ru.md)

**2EPUB creates one bilingual EPUB from an English book and its Russian translation.** English text is shown in bold, with the aligned Russian text directly below it. The English book determines chapter order and paragraph structure.

**Status:** experimental MVP v0.1. The converter works on real books, but alignment is not guaranteed to be semantically perfect.

## Quick start (macOS)

The project is developed and tested with Python 3.14. Git and an internet connection for the first model download are required.

Clone the repository and enter the project directory:

```bash
git clone https://github.com/aleksandr88p/2EPUB.git
cd 2EPUB
```

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

The Qwen3-Embedding-0.6B model downloads automatically the first time you run `build`. It is stored in `cache/models/`; sentence embeddings are cached in `cache/embeddings/`. The first full conversion can take a while. Repeating a conversion reuses cached embeddings.

## Prepare your books

Put your books in `books/` if convenient. The directory is kept in Git by `.gitkeep`, while personal book files are ignored and are not committed. You can also pass files from another location.

Both inputs can be FB2 or EPUB, independently:

| English original | Russian translation | Supported |
|---|---|---|
| FB2 | FB2 | Yes |
| FB2 | EPUB | Yes |
| EPUB | FB2 | Yes |
| EPUB | EPUB | Yes |

The first positional input is always the English original; the second is the Russian translation. The program does not guess which language a file contains. Use quotation marks around paths containing spaces or non-Latin characters.

## Typical workflow

First check the two source books:

```bash
python -m bookparse check \
  "books/Lost_Gods_-_Brom.epub" \
  "books/потерянные боги Бром.fb2"
```

Then create the full bilingual book:

```bash
python -m bookparse build \
  "books/Lost_Gods_-_Brom.epub" \
  "books/потерянные боги Бром.fb2"
```

The default output is `output/2epub.epub`. The output directory is created automatically if necessary.

To create a quick preview containing the first two aligned chapter pairs:

```bash
python -m bookparse build \
  "books/Lost_Gods_-_Brom.epub" \
  "books/потерянные боги Бром.fb2" \
  --preview
```

To choose a different output file:

```bash
python -m bookparse build \
  "books/Lost_Gods_-_Brom.epub" \
  "books/потерянные боги Бром.fb2" \
  --output "output/Lost_Gods_bilingual.epub"
```

Open the resulting `.epub` in a regular ebook reader or transfer it to an e-reader. The `output/` directory is ignored by Git.

## CLI reference

For ordinary use, the `check` and `build` commands are enough. All paths below may point to either `.fb2` or `.epub`, except `parse` which accepts one book at a time.

### `check EN RU`

Run source preflight checks and print a report before conversion. An optional `-o/--output PATH` saves the report to a text file.

```bash
python -m bookparse check "books/english.epub" "books/russian.fb2"
python -m bookparse check "books/english.epub" "books/russian.fb2" \
  --output "output/preflight.txt"
```

`build` runs the same preflight automatically, so a separate check is optional but useful when you want to inspect source warnings first.

### `build EN RU`

Build the full bilingual EPUB by default. Add `--preview` for only the first two aligned chapter pairs. Use `-o/--output PATH` to choose a destination; otherwise the file is `output/2epub.epub`.

```bash
python -m bookparse build "books/english.fb2" "books/russian.epub"
python -m bookparse build "books/english.fb2" "books/russian.epub" --preview
python -m bookparse build "books/english.fb2" "books/russian.epub" \
  --output "output/My_Bilingual_Book.epub"
```

The builder accepts either format for each input, including mixed pairs. The English original defines the structure. Its cover and referenced images are carried into the result when present.

### `align-chapters EN RU`

Align chapters and save a JSON report. This is a diagnostic command; it does not create an EPUB. Optional `-o/--output PATH` changes the report path. By default, the JSON is saved as `cache/align_<en>--<ru>.json`.

```bash
python -m bookparse align-chapters "books/english.epub" "books/russian.fb2"
python -m bookparse align-chapters "books/english.epub" "books/russian.fb2" \
  --output "output/chapter_alignment.json"
```

### `parse BOOK`

Parse one EPUB or FB2 and save its extracted structure as JSON. By default, the JSON is written to `cache/<book-name>.json`. `-o/--output PATH` changes the destination; `--lang LANG` overrides the source language used for sentence segmentation (normally `en` or `ru`).

```bash
python -m bookparse parse "books/english.epub"
python -m bookparse parse "books/russian.fb2" --lang ru \
  --output "cache/russian_structure.json"
```

### Help and tests

```bash
python -m bookparse --help
python -m bookparse build --help
python -m pytest
```

## Reading the reports

### `check`

The report has issue levels and an overall verdict:

- `INFO` — informational note; not a blocker.
- `WARNING` — source feature that may affect quality; review it. The build continues.
- `ERROR` — source issue considered critical by the current checks.
- `VERDICT: OK` — no warnings or errors were detected (informational notes may still be listed).
- `VERDICT: PROCEED WITH CAUTION` — warnings were found; inspect them before relying on the output.
- `VERDICT: STOP` — an error was found. `build` stops before alignment and returns a non-zero status.

These are heuristic checks. A warning may be a false positive, and an `OK` verdict is not a semantic translation review.

### `align-chapters`

- **Paired** — chapter pairs selected by the alignment pass.
- **Confident / doubtful** — quality labels for those pairs, based on structural signals; they are not proof of semantic correctness.
- **Unmatched** — titled main-body chapters without a counterpart on the other side.
- **Service sections** — untitled front matter or auxiliary sections such as notes; they are listed separately from plot chapter matching.
- **Word coverage** — the proportion of each book's words that belong to matched chapters.

### `build`

- **`Pair`** — the English and Russian chapters being processed together.
- **`EN-only`** — an English chapter without a Russian pair; it is retained without invented translation.
- **`embeddings`** — newly computed sentence and joined-text vectors; the rest came from cache.
- **`ops`** — counts of sentence alignment operations (`1:1`, `1:2`, `2:1`, `skip_en`, `skip_ru`) and match quality labels.
- **`[OK]` / `[FAIL]`** — post-write EPUB checks, including chapter set/order, XHTML, parsed-text completeness/order, and internal image links. A failed check gives a non-zero exit status and is not reported as a successful conversion.
- **`TOTAL`** — chapter, sentence, EN-only chapter, and English paragraph-without-translation totals. Russian chapters not represented in the English structure are listed separately.

Text-preservation checks compare the EPUB with the parser's extracted text. They cannot prove that each Russian sentence is the correct translation of the English sentence above it.

## Tested conversions

| Book | Inputs | Result |
|---|---|---|
| *The Blade Itself* / «Кровь и железо» | FB2 + FB2 | Full EPUB created; 52 XHTML sections, 48 paired chapters, 4 EN-only chapters. Structural and text checks passed. |
| *Lost Gods* | EPUB + FB2 | Full EPUB created with the real Qwen model; 116 XHTML sections, 110 paired chapters and 6 EN-only chapters. Structure, text and image-link checks passed. |

The `Lost Gods` preflight returned `PROCEED WITH CAUTION` because it found source/chapter-structure warnings; generation and post-write checks still completed. Its chapter alignment included one doubtful pair. These book-specific results do not measure universal alignment accuracy.

Latest recorded automated test run: **105 passed**.

## Limitations

- Semantic alignment is experimental. A translation can appear under a neighboring English paragraph, especially for short dialogue, standalone lines, letters and epigraphs.
- `skip_en` and `skip_ru` are alignment operations, not semantic judgements. Review the resulting paragraphs when these are frequent.
- The English book defines paragraph structure; Russian paragraph boundaries are not reconstructed. Verse lines may be combined.
- Some source formatting, including inline italics, is flattened by the parsers.
- The automatic checks verify structure and text relative to parser output. They do not verify the meaning or literary quality of the translation, nor do they catch text already lost by parsing.
- Russian chapters without an English counterpart are reported but are not added to the generated book.

MVP v0.1 is a working converter, not a guarantee of perfect alignment. Review the generated EPUB before relying on it for close reading.

## Project structure

```text
2EPUB/
├── bookparse/       # Parsers, CLI, checks, alignment and reconstruction
├── experiments/     # EPUB builder and diagnostic scripts
├── tests/           # Automated tests, including end-to-end CLI cases
├── docs/            # Architecture notes
├── books/           # User input books (ignored by Git)
├── cache/           # Model weights and embedding cache (ignored by Git)
├── output/          # Generated EPUBs and reports (ignored by Git)
├── requirements.txt
├── README.md
└── README.ru.md
```

For implementation details, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Personal books, model files and generated output are not committed to the repository.

## Next steps

Converter work may improve alignment, preserve more formatting and expand QA across books. A future, separate reading app is a plan—not an implemented feature—and may include a web reader, parallel-text controls, optional LLM translation and text-level adaptation.
