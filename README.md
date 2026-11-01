
# 2EPUB

**2EPUB** is a Python application for creating bilingual ebooks with parallel English and Russian text.

The application takes two books — an English original and its Russian translation — aligns their content, and generates a single EPUB file.

In the resulting book:

- **English paragraphs are displayed in bold.**
- The corresponding Russian translation appears directly below each English paragraph.
- The chapter order and structure follow the English original.
- The resulting EPUB can be opened in a standard EPUB reader.

**Status:** MVP v0.1 — working experimental version.

[English](README.md) | [Русский](README.ru.md)

## Installation and Usage

Instructions for macOS. Git and Python are required (the project was developed using Python 3.14).

### 1. Clone the repository

Open Terminal and run:

```bash
git clone https://github.com/aleksandr88p/2EPUB.git
cd 2EPUB
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

On the first conversion, the application automatically downloads the Qwen embedding model. An internet connection is required for this initial download.

### 4. Add your books

Place the English original and Russian translation in `.fb2` format inside the `books/` directory.

For example:

```text
2EPUB/
├── books/
│   ├── The Blade Itself.fb2
│   └── Кровь и железо.fb2
├── experiments/
├── output/
└── ...
```

Create the `books/` directory manually if it does not exist.

You can use any filenames. There is no need to rename your books to `english.fb2` or `russian.fb2`.

### 5. Run the converter

From the project directory, with the virtual environment activated, run:

```bash
python experiments/build_preview_epub.py \
  --en "books/The Blade Itself.fb2" \
  --ru "books/Кровь и железо.fb2" \
  --output "output/My Book.epub" \
  --all
```

Replace the example filenames with your own.

**Command-line options:**

| Option | Description |
|---|---|
| `--en` | Path to the English original |
| `--ru` | Path to the Russian translation |
| `--output` | Path and filename of the generated EPUB |
| `--all` | Process the entire book instead of only the first two aligned chapters |

Use quotation marks around paths containing spaces or non-Latin characters.

### 6. Open the generated book

After successful conversion, the EPUB file will be saved at the specified location:

```text
output/My Book.epub
```

You can open it in an EPUB reader or transfer it to your e-reader.

### Additional commands

Generate a preview using only the first two aligned chapters:

```bash
python experiments/build_preview_epub.py \
  --en "books/The Blade Itself.fb2" \
  --ru "books/Кровь и железо.fb2" \
  --output "output/preview.epub"
```

Display all available options:

```bash
python experiments/build_preview_epub.py --help
```

Run automated tests:

```bash
python -m pytest
```

Running the converter without arguments uses the default test books from the `books/` directory, if those files are available.

## How 2EPUB Works

The application processes books through several stages.

### 1. Book parsing

The FB2 parser extracts chapters, paragraphs, and sentences from both books.

The English original defines the structure of the resulting bilingual book.

### 2. Chapter alignment

The application identifies corresponding chapters in the English and Russian books.

This is necessary because translations may differ from the original in their chapter structure.

### 3. Embedding generation

The application generates multilingual sentence embeddings using the local **Qwen3-Embedding-0.6B** model.

These embeddings help estimate semantic similarity between English and Russian sentences.

Generated embeddings are cached on disk, allowing alignment experiments without repeatedly running the model.

### 4. Sentence alignment

A dynamic programming algorithm finds an ordered alignment between the original and translated sentences.

Supported alignment operations include:

- `1:1` — one sentence aligned with one sentence.
- `1:2` — one sentence aligned with two sentences.
- `2:1` — two sentences aligned with one sentence.
- `skip_en` and `skip_ru` — unmatched sentences on either side.

### 5. Paragraph reconstruction

After sentence alignment, the application reconstructs bilingual paragraphs using the English book's paragraph structure as the reference.

### 6. EPUB generation

The application uses EbookLib to generate the final EPUB containing English paragraphs and their Russian translations.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| EbookLib | EPUB generation |
| BeautifulSoup | Ebook structure parsing |
| Qwen3-Embedding-0.6B | Multilingual sentence embeddings |
| NumPy | Numerical processing |
| Dynamic programming | Text alignment |
| pytest | Automated testing |

The embedding model runs locally. After the initial download, no cloud API is required to run the model.

## Current Project Status

### Implemented features

- FB2 parsing and text extraction.
- English–Russian chapter alignment.
- Embedding generation and caching.
- Sentence and sentence-group alignment.
- Bilingual paragraph reconstruction.
- EPUB generation.
- Structural and text-preservation checks.
- Command-line interface with configurable input and output paths.
- Automated tests.

### Initial testing

The first experimental book was **The Blade Itself / «Кровь и железо»**.

A complete bilingual EPUB was successfully generated.

Structural and text-preservation checks were performed. A partial independent semantic review also identified several paragraph-boundary alignment errors.

These results demonstrate that the processing pipeline works, but they do not guarantee perfect alignment.

### Known limitations

- The converter currently accepts FB2 input only.
- Alignment errors may occur between neighboring paragraphs, particularly short dialogue lines.
- Some original formatting may not be preserved perfectly, including italics and poetry line breaks.
- Translations can differ significantly from the original in structure.
- Automated validation does not guarantee that every paragraph is semantically aligned correctly.
- Alignment quality across other books and translations has not yet been thoroughly evaluated.

**Important:** This MVP is intended for experimentation and personal use. Generated books should be reviewed for alignment quality.

## Project Structure

```text
2EPUB/
├── bookparse/       # Book parsing and processing
├── experiments/     # Experiments and EPUB generation scripts
├── tests/           # Automated tests
├── docs/            # Technical documentation
├── books/           # Input books
├── cache/           # Model files and embedding cache
├── output/          # Generated EPUB files
├── requirements.txt
├── README.md
└── README.ru.md
```

The `books/`, `cache/`, and `output/` directories are excluded from Git. Input books, downloaded models, and generated EPUB files are not committed to the repository.

For implementation details, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Roadmap

### Converter improvements

- Improve alignment accuracy and fix paragraph-boundary shifts.
- Preserve more of the original formatting.
- Add full EPUB input support.
- Expand alignment quality reports.
- Evaluate the algorithm on additional books and translations.

### Future Smart Reader

2EPUB is intended to become the foundation of a larger language-learning application.

Potential features include:

- Reading books directly in a web browser.
- Switching between original text, translation, and parallel reading modes.
- Translating individual words, sentences, and paragraphs.
- Adapting English text difficulty using an LLM, for example from B2 to B1.
- Saving vocabulary and maintaining a personal word list.
- Uploading and processing personal ebooks.

**Next development stage:** designing the backend and building a minimal web reader.

## Development Principles

The project is developed in Python with the assistance of the Windsurf AI IDE.

Core principles:

- The English original determines the structure of the generated book.
- Preserve all original and translated text without unnecessary loss or duplication.
- Keep parsing, alignment, and EPUB generation separate.
- Cache embeddings to make experiments faster.
- Validate results using automated tests and independent quality reviews.

The current release is an experimental MVP, not a finished commercial product.
