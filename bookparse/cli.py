"""Command line interface: `python -m bookparse parse <book>`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .checks import ERROR, build_report
from .epub_parser import parse_epub
from .fb2_parser import parse_fb2
from .models import Book

PARSERS = {".epub": parse_epub, ".fb2": parse_fb2}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bookparse",
        description="Extract structure (chapters, paragraphs, sentences) "
                    "from EPUB/FB2 books into JSON.",
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
        help="override language for sentence splitting (default: book metadata)",
    )
    check_cmd = sub.add_parser(
        "check", help="preflight validation of English + Russian sources",
    )
    check_cmd.add_argument("english", type=Path, help="path to English .epub/.fb2")
    check_cmd.add_argument("russian", type=Path, help="path to Russian .epub/.fb2")
    check_cmd.add_argument(
        "-o", "--output", type=Path, default=None,
        help="optional path to save the text report",
    )
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
    book = parser_fn(path, lang=args.lang)

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


def _parse_any(path: Path) -> Book:
    if not path.is_file():
        raise SystemExit(f"error: file not found: {path}")
    parser_fn = PARSERS.get(path.suffix.lower())
    if parser_fn is None:
        raise SystemExit(
            f"error: unsupported format {path.suffix!r} (expected .epub or .fb2)"
        )
    return parser_fn(path)


def run_check(args: argparse.Namespace) -> int:
    en = _parse_any(args.english)
    ru = _parse_any(args.russian)
    report, issues = build_report(en, ru)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"\nReport saved to: {args.output}")
    return 2 if any(i.level == ERROR for i in issues) else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "parse":
        run_parse(args)
    elif args.command == "check":
        return run_check(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
