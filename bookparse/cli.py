"""Command line interface: `python -m bookparse parse <book>`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .chapter_alignment import align_chapters, format_report
from .checks import ERROR, build_report
from .epub_parser import parse_epub
from .fb2_parser import parse_fb2
from .languages import (
    UnsupportedLanguageError,
    language_name,
    normalize_language,
    resolve_language_pair,
)
from .models import Book

PARSERS = {".epub": parse_epub, ".fb2": parse_fb2}
DEFAULT_BUILD_OUTPUT = Path("output") / "2epub.epub"


def _add_language_overrides(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--source-lang", default=None, metavar="CODE",
        help="override original language (en, ru, es; default: book metadata)",
    )
    command.add_argument(
        "--translation-lang", default=None, metavar="CODE",
        help="override translation language (en, ru, es; default: book metadata)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bookparse",
        description="Parse, check, align, and build bilingual EPUB books "
                    "from EPUB/FB2 files.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    parse_cmd = sub.add_parser("parse", help="parse a book file")
    parse_cmd.add_argument("path", type=Path, help="path to .epub or .fb2 file")
    parse_cmd.add_argument(
        "-o", "--output", type=Path, default=None,
        help="output JSON path (default: cache/<book-name>.json)",
    )
    parse_cmd.add_argument(
        "--lang", default=None,
        help="override segmentation language (en, ru, es; default: metadata)",
    )
    check_cmd = sub.add_parser(
        "check", help="preflight validation of a source and its translation",
    )
    check_cmd.add_argument(
        "english", type=Path, metavar="SOURCE",
        help="original/source book (.epub or .fb2)",
    )
    check_cmd.add_argument(
        "russian", type=Path, metavar="TRANSLATION",
        help="translated book (.epub or .fb2)",
    )
    check_cmd.add_argument(
        "-o", "--output", type=Path, default=None,
        help="optional path to save the text report",
    )
    _add_language_overrides(check_cmd)
    build_cmd = sub.add_parser(
        "build", help="build a bilingual EPUB from an original and translation",
    )
    build_cmd.add_argument(
        "english", type=Path, metavar="SOURCE",
        help="original/source book (.epub or .fb2)",
    )
    build_cmd.add_argument(
        "russian", type=Path, metavar="TRANSLATION",
        help="translated book (.epub or .fb2)",
    )
    build_cmd.add_argument(
        "--preview", action="store_true",
        help="build only the first two aligned chapter pairs",
    )
    build_cmd.add_argument(
        "-o", "--output", type=Path, default=None,
        help=f"output EPUB path (default: {DEFAULT_BUILD_OUTPUT})",
    )
    _add_language_overrides(build_cmd)
    align_cmd = sub.add_parser(
        "align-chapters",
        help="align chapters of two books (JSON result)",
    )
    align_cmd.add_argument(
        "english", type=Path, metavar="SOURCE",
        help="original/source book (.epub or .fb2)",
    )
    align_cmd.add_argument(
        "russian", type=Path, metavar="TRANSLATION",
        help="translated book (.epub or .fb2)",
    )
    align_cmd.add_argument(
        "-o", "--output", type=Path, default=None,
        help="output JSON path (default: cache/align_<en>--<ru>.json)",
    )
    _add_language_overrides(align_cmd)
    return parser


def run_parse(args: argparse.Namespace) -> Book:
    path: Path = args.path
    if not path.is_file():
        raise SystemExit(f"error: file not found: {path}")
    parser_fn = PARSERS.get(path.suffix.lower())
    if parser_fn is None:
        raise SystemExit(
            f"error: unsupported format {path.suffix!r} (expected .epub or .fb2)"
        )
    try:
        lang = normalize_language(args.lang)
        book = parser_fn(path, lang=lang)
    except UnsupportedLanguageError as exc:
        raise SystemExit(f"error: {exc}") from None

    out = args.output or Path("cache") / f"{path.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(book.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stats = book.stats()
    print(f"Parsed: {path.name} ({book.format})")
    print(f"  chapters:   {stats['chapters']}")
    print(f"  paragraphs: {stats['paragraphs']}")
    print(f"  sentences:  {stats['sentences']}")
    print(f"  words:      {stats['words']}")
    print(f"Saved to: {out}")
    return book


def _parse_any(path: Path, lang: str | None = None) -> Book:
    if not path.is_file():
        raise SystemExit(f"error: file not found: {path}")
    parser_fn = PARSERS.get(path.suffix.lower())
    if parser_fn is None:
        raise SystemExit(
            f"error: unsupported format {path.suffix!r} (expected .epub or .fb2)"
        )
    try:
        normalized_lang = normalize_language(lang)
        return parser_fn(path, lang=normalized_lang)
    except UnsupportedLanguageError as exc:
        raise SystemExit(f"error: {exc}") from None


def _pair_labels(
    source: Book,
    translation: Book,
    source_override: str | None,
    translation_override: str | None,
) -> tuple[str, str]:
    try:
        source_lang, translation_lang = resolve_language_pair(
            source, translation, source_override, translation_override,
        )
    except UnsupportedLanguageError as exc:
        raise SystemExit(f"error: {exc}") from None
    return language_name(source_lang), language_name(translation_lang)


def run_check(args: argparse.Namespace) -> int:
    source = _parse_any(args.english, args.source_lang)
    translation = _parse_any(args.russian, args.translation_lang)
    source_label, translation_label = _pair_labels(
        source, translation, args.source_lang, args.translation_lang,
    )
    report, issues = build_report(
        source, translation, source_label, translation_label,
    )
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"\nReport saved to: {args.output}")
    return 2 if any(i.level == ERROR for i in issues) else 0


def run_align(args: argparse.Namespace) -> int:
    source = _parse_any(args.english, args.source_lang)
    translation = _parse_any(args.russian, args.translation_lang)
    source_label, translation_label = _pair_labels(
        source, translation, args.source_lang, args.translation_lang,
    )
    result = align_chapters(source, translation)

    out = args.output or (
        Path("cache") / f"align_{args.english.stem}--{args.russian.stem}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(format_report(result, source_label, translation_label))
    print(f"\nSaved to: {out}")
    return 0


def run_build(args: argparse.Namespace) -> int:
    """Delegate conversion to the existing EPUB builder."""
    from experiments.build_preview_epub import main as build_epub

    argv = ["--en", str(args.english), "--ru", str(args.russian)]
    if not args.preview:
        argv.append("--all")
    if args.source_lang is not None:
        argv.extend(["--source-lang", args.source_lang])
    if args.translation_lang is not None:
        argv.extend(["--translation-lang", args.translation_lang])
    argv.extend(["--output", str(args.output or DEFAULT_BUILD_OUTPUT)])
    return build_epub(argv)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "parse":
        run_parse(args)
    elif args.command == "check":
        return run_check(args)
    elif args.command == "build":
        return run_build(args)
    elif args.command == "align-chapters":
        return run_align(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
