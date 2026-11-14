# 2EPUB

[English](README.md) | [Русский](README.ru.md)

2EPUB is an experimental Python CLI that creates a bilingual EPUB from an
original book and its translation. The original defines chapter order and
paragraph structure; its text is bold, with the corresponding translation
immediately below.

## Features

- EPUB and FB2 inputs in any combination: EPUB+EPUB, EPUB+FB2, FB2+EPUB,
  or FB2+FB2.
- English, Russian, and Spanish in all six directions: EN→RU, RU→EN,
  EN→ES, ES→EN, RU→ES, and ES→RU.
- Language detection from book metadata, with `--source-lang` and
  `--translation-lang` overrides.
- Source chapter structure, supported block formatting, cover and
  referenced images are carried over when available. The output includes
  EPUB navigation and a table of contents.
- A conservative FB2 fallback can recover repeated part/chapter structure
  from a single malformed section; uncertain boundaries are left intact and
  reported.
- Optional `--short-paragraphs` mode targets about 35–40 source words per
  block. It keeps the complete aligned translation without a word limit
  and splits only at sentence and alignment-operation boundaries; 1:2 and
  2:1 groups stay intact.

## Requirements and installation

Python 3.14 is the tested version. The commands below use macOS/Linux.
An internet connection is needed the first time the local Qwen embedding
model is downloaded.

```bash
git clone https://github.com/aleksandr88p/2EPUB.git
cd 2EPUB
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

The model is stored in `cache/models/` and sentence embeddings in
`cache/embeddings/`. Later builds reuse the embedding cache.

## Quick start

Put your `.epub` or `.fb2` files in `books/` (or use paths elsewhere).
The first path is always the original; the second is its translation.

```bash
python -m bookparse check "books/original.epub" "books/translation.fb2"
python -m bookparse build "books/original.epub" "books/translation.fb2" \
  --short-paragraphs --output "output/bilingual.epub"
```

The build command creates a full book by default. Its default output is
`output/2epub.epub`; `--preview` limits the result to the first two aligned
chapter pairs. The output directory is created automatically.

Languages are read from EPUB/FB2 metadata. If metadata is missing or wrong,
set both overrides, for example:

```bash
python -m bookparse build "books/original.epub" "books/перевод.fb2" \
  --source-lang es --translation-lang ru
```

Supported codes include `en`/`eng`, `ru`/`rus`, and `es`/`spa`; regional
tags such as `es-ES` are accepted. If language metadata is absent, the
legacy fallback is EN→RU.

## CLI commands

All book inputs can independently be EPUB or FB2.

| Command | Purpose | Example |
|---|---|---|
| `check SOURCE TRANSLATION` | Print preflight warnings and errors. `--output PATH` saves the report. | `python -m bookparse check source.epub translation.fb2` |
| `build SOURCE TRANSLATION` | Create the bilingual EPUB. Supports `--preview`, `--short-paragraphs`, `--output PATH`, and language overrides. | `python -m bookparse build source.epub translation.fb2 --preview` |
| `align-chapters SOURCE TRANSLATION` | Save a chapter-alignment JSON report; `--output PATH` changes its location. | `python -m bookparse align-chapters source.epub translation.fb2` |
| `parse BOOK` | Save one book's parsed structure as JSON; supports `--output PATH` and `--lang CODE` (`en`, `ru`, or `es`). | `python -m bookparse parse source.epub --output cache/source.json` |

For command help, run `python -m bookparse --help` or
`python -m bookparse build --help`.

## Limitations

- Only English, Russian, and Spanish are supported. Sentence segmentation
  and semantic alignment are heuristic; inspect warnings and review the
  generated book.
- Chapter matching uses titled main-body chapters and does not merge
  chapters. Translation chapters without a source counterpart are not added
  to the output.
- The source controls paragraph structure; translation paragraph breaks
  are not retained. Some inline formatting, such as italics, is flattened.
- `--short-paragraphs` cannot split a grouped alignment operation, a
  group crossing source paragraphs, or structural blocks. An indivisible
  operation may therefore exceed 40 source words.
- Verification compares the generated EPUB with parsed text. It cannot
  detect text already lost by a parser or judge translation meaning.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for implementation details
and [PROGRESS.md](PROGRESS.md) for current verification status.
