# 2EPUB — Agent Instructions

## Project
Python CLI tool that combines English and Russian EPUB books into one bilingual EPUB.

English paragraphs must be bold, followed by their Russian translations.

## Tech Stack
- Python, EbookLib, BeautifulSoup
- Qwen3-Embedding-0.6B for multilingual embeddings
- Dynamic programming for sentence alignment
- Local development on MacBook Air M4 (16 GB RAM)

## Architecture
- English EPUB is the source of chapter structure, images and formatting.
- Align chapters first, then sentences and sentence groups.
- Preserve original paragraph boundaries when generating the final EPUB.
- Never silently lose or duplicate text.
- Cache embeddings to avoid repeated model inference.
- Generate alignment quality reports.

## Development Rules
- Keep code simple, readable and modular.
- Use type hints and descriptive function names.
- Write tests for parsing and alignment.
- Do not add unnecessary dependencies.
- Work in small iterations.
- Do not implement features beyond the current task.
- Explain important architectural decisions.
- Never modify unrelated files without permission.