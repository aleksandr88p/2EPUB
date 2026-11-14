# 2EPUB — Agent Instructions

## Project

2EPUB is a Python CLI that combines an original book and its translation
into one bilingual EPUB. Source and translation may each be EPUB or FB2.
The MVP supports English, Russian and Spanish in all six directions.
The source defines output chapter order and paragraph structure; source
prose is bold and its aligned translation follows below.

## Documentation sources

- Read `PROGRESS.md` at the start of a task; keep it factual and current.
- `README.md` and `README.ru.md` are user-facing guides.
- `docs/ARCHITECTURE.md` is the technical reference for parsing, alignment,
  reconstruction and EPUB generation.
- Avoid duplicating architecture detail in the README or progress log.

## Development workflow

- Work only on the requested task; do not refactor unrelated code or add
  unrequested features.
- Keep changes simple and follow existing conventions. Use type hints,
  descriptive names and focused tests.
- Preserve source/translation order and do not silently lose or duplicate
  text. Do not change alignment behavior unless the task calls for it.
- Do not add dependencies without a clear need.
- Run the relevant tests with `venv/bin/python -m pytest`. The test suite
  stubs model inference; it does not require a real Qwen download.
- Do not launch a real full-book conversion or model download unless the
  task explicitly requests it. Such a run can load Qwen and write large
  output files.
- Keep personal books, caches, model files and generated EPUBs in the
  ignored `books/`, `cache/` and `output/` directories.
- After each task, update `PROGRESS.md` with verified changes, test results
  and the next outstanding validation task. Do not report a build as
  successful without a confirmed completion result.