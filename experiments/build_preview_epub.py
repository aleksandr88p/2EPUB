"""Build a bilingual EPUB from a source book and its translation.

Two modes:

  * preview (default): the first two aligned chapter pairs ->
    output/2epub_preview.epub
  * full (--all): every aligned pair plus every source chapter that has
    no translation counterpart (front matter, unmatched plot chapters,
    auxiliary bodies), in the EN book's original order ->
    output/2epub_full.epub

Both run the existing pipeline end to end:

    chapter alignment -> Qwen embeddings (disk cache) -> DP sentence
    alignment with the joined-text group scorer -> paragraph
    reconstruction -> EPUB assembly.

Layout contract: the source book defines structure and chapter order.
Every source paragraph is rendered bold, its aligned translation
sentences follow in a normal paragraph. Source cover and images are
carried over; translation images are not included yet. Paragraphs
without a translation are left bare — no placeholder text is invented.
Translation chapters without a source counterpart are not added to the
EPUB (the source book defines its
structure); they are listed in the run report instead.

Run:  venv/bin/python experiments/build_preview_epub.py \
          --en books/source.fb2 --ru books/translation.epub [--all]
      venv/bin/python experiments/build_preview_epub.py \
          --en books/english.epub --ru books/russian.fb2 \
          --output output/my_book.epub --all
"""

from __future__ import annotations

import argparse
import base64
import posixpath
import sys
import uuid
from pathlib import Path
from pathlib import PurePosixPath

import numpy as np
from bs4 import BeautifulSoup
from ebooklib import epub, ITEM_COVER, ITEM_DOCUMENT
from html import escape

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bookparse.chapter_alignment import align_chapters  # noqa: E402
from bookparse.checks import ERROR, WARNING, build_report  # noqa: E402
from bookparse.embedding_cache import embed_with_cache  # noqa: E402
from bookparse.epub_parser import parse_epub  # noqa: E402
from bookparse.fb2_parser import parse_fb2  # noqa: E402
from bookparse.languages import (  # noqa: E402
    UnsupportedLanguageError,
    language_name,
    normalize_language,
    resolve_language_pair,
)
from bookparse.models import Book, Chapter, Paragraph  # noqa: E402
from bookparse.paragraph_reconstruction import (  # noqa: E402
    BilingualParagraph,
    rebuild_chapter,
)
from bookparse.sentence_alignment import (  # noqa: E402
    align_sentences,
    summarize_ops,
)

MODELS_DIR = PROJECT_ROOT / "cache" / "models"
EMB_DIR = PROJECT_ROOT / "cache" / "embeddings"
PREVIEW_PATH = PROJECT_ROOT / "output" / "2epub_preview.epub"
FULL_PATH = PROJECT_ROOT / "output" / "2epub_full.epub"

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
ENCODE_PARAMS: dict = {"prompt": None, "normalize_embeddings": True}

N_CHAPTERS = 2


def _rel(path: Path) -> Path | str:
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_preview_epub",
        description="Build a bilingual EPUB from an original and "
                    "translation (EPUB or FB2).",
    )
    parser.add_argument(
        "--all", action="store_true", dest="all_chapters",
        help="process every aligned pair plus source-only chapters "
             "(default: first %d pairs only)"
             % N_CHAPTERS,
    )
    parser.add_argument(
        "--en", type=Path, required=True, metavar="PATH",
        help="legacy --en option: path to the source/original book",
    )
    parser.add_argument(
        "--ru", type=Path, required=True, metavar="PATH",
        help="legacy --ru option: path to the translation book",
    )
    parser.add_argument(
        "--source-lang", metavar="CODE",
        help="override the original language (en, ru, es; default: metadata)",
    )
    parser.add_argument(
        "--translation-lang", metavar="CODE",
        help="override the translation language (en, ru, es; default: metadata)",
    )
    parser.add_argument(
        "--output", type=Path, default=None, metavar="PATH",
        help="output EPUB path (default: %s, or %s with --all)"
             % (_rel(PREVIEW_PATH), _rel(FULL_PATH)),
    )
    return parser


def _output_path(all_chapters: bool) -> Path:
    return FULL_PATH if all_chapters else PREVIEW_PATH


def _validate_book_path(path: Path, label: str) -> Path:
    """Validate an input book path and report user-facing errors."""
    if not path.is_file():
        raise SystemExit(f"error: {label} book not found: {path}")
    if path.suffix.lower() not in {".fb2", ".epub"}:
        raise SystemExit(
            f"error: {label} book must be a .fb2 or .epub file: {path}"
        )
    return path


def _run_preflight(
    en_book: Book,
    ru_book: Book,
    source_label: str = "English",
    translation_label: str = "Russian",
) -> bool:
    """Print source checks and return False when alignment must stop."""
    report, issues = build_report(
        en_book, ru_book, source_label, translation_label,
    )
    print(report)

    has_error = any(issue.level == ERROR for issue in issues)
    has_stop_verdict = "VERDICT: STOP" in report
    if has_error or has_stop_verdict:
        print("Preflight: ERROR — conversion stopped.")
        return False

    warning_count = sum(issue.level == WARNING for issue in issues)
    if warning_count:
        print(f"Preflight: {warning_count} warning(s); continuing.")
    else:
        print("Preflight: OK; continuing.")
    return True


def _parse_book(path: Path, lang: str | None = None) -> Book:
    """Select the existing parser from the input file extension."""
    lang = normalize_language(lang)
    if path.suffix.lower() == ".fb2":
        return parse_fb2(path, lang=lang)
    if path.suffix.lower() == ".epub":
        return parse_epub(path, lang=lang)
    raise ValueError(f"unsupported book format: {path.suffix}")


_CSS = """\
p.source { font-weight: bold; margin: 0.4em 0 0 0; }
h2.source { margin: 1em 0 0.3em 0; }
p.translation { margin: 0.15em 0 0.9em 0; }
blockquote { margin: 0.4em 1.2em; }
p.empty-line { margin: 0.6em 0; }
div.source-img { text-align: center; margin: 0.6em 0; }
"""

_MIME_EXT = {
    "image/jpeg": ".jpg", "image/png": ".png",
    "image/gif": ".gif", "image/svg+xml": ".svg",
}
_QUOTE_TYPES = {"epigraph", "cite", "quote"}


# --- FB2 binary extraction (the parser keeps only image references) --

def _fb2_soup(path: Path) -> BeautifulSoup:
    return BeautifulSoup(path.read_bytes(), "lxml-xml")


def _fb2_binaries(soup: BeautifulSoup) -> dict[str, tuple[str, bytes]]:
    """id -> (media_type, decoded bytes) for every <binary> element."""
    out: dict[str, tuple[str, bytes]] = {}
    for b in soup.find_all("binary"):
        bid = b.get("id")
        if not bid:
            continue
        out[bid] = (
            b.get("content-type") or "image/jpeg",
            base64.b64decode(b.get_text(strip=True)),
        )
    return out


def _fb2_cover_id(soup: BeautifulSoup) -> str | None:
    img = soup.select_one("title-info coverpage image")
    if img is None:
        return None
    href = img.get("l:href") or img.get("xlink:href") or img.get("href")
    return href.lstrip("#") if href else None


def _source_assets(
    path: Path,
) -> tuple[dict[str, tuple[str, bytes]], str | None]:
    """Return image assets keyed by parsed references and the cover key."""
    if path.suffix.lower() == ".fb2":
        soup = _fb2_soup(path)
        return _fb2_binaries(soup), _fb2_cover_id(soup)

    source = epub.read_epub(str(path))
    images = {
        PurePosixPath(item.get_name()).as_posix():
            (item.media_type, item.get_content())
        for item in source.get_items()
        if (item.media_type or "").startswith("image/")
    }

    cover_id = next(
        (
            attrs.get("content")
            for _, attrs in source.get_metadata("OPF", "meta")
            if attrs.get("name") == "cover"
        ),
        None,
    )
    cover_item = source.get_item_with_id(cover_id) if cover_id else None
    if cover_item is None:
        cover_item = next(
            (
                item for item in source.get_items()
                if "cover-image" in getattr(item, "properties", [])
            ),
            None,
        )
    if cover_item is None:
        cover_item = next(
            (
                item for item in source.get_items()
                if item.get_type() == ITEM_COVER
            ),
            None,
        )
    cover_key = (
        PurePosixPath(cover_item.get_name()).as_posix()
        if cover_item is not None
        else None
    )
    return images, cover_key


def _image_key(src: str | None) -> str:
    return src[1:] if src and src.startswith("#") else (src or "")


def _image_file_names(
    image_keys: set[str],
    images: dict[str, tuple[str, bytes]],
) -> dict[str, str]:
    """Assign collision-free output paths to the referenced source images."""
    out: dict[str, str] = {}
    used: set[str] = set()
    for key in sorted(image_keys):
        if key not in images:
            continue
        media_type, _ = images[key]
        name = PurePosixPath(key).name or "image"
        stem, suffix = posixpath.splitext(name)
        if not suffix:
            suffix = _MIME_EXT.get(media_type, ".img")
        candidate = f"images/{stem}{suffix}"
        serial = 2
        while candidate in used:
            candidate = f"images/{stem}_{serial}{suffix}"
            serial += 1
        used.add(candidate)
        out[key] = candidate
    return out


def _cover_file_name(source_key: str, media_type: str) -> str:
    suffix = PurePosixPath(source_key).suffix or _MIME_EXT.get(
        media_type, ".jpg",
    )
    return f"cover{suffix}"


# --- XHTML rendering --------------------------------------------------

def _render_paragraph(
    bp: BilingualParagraph,
    src: Paragraph,
    img_files: dict[str, str],
    source_lang: str = "en",
    translation_lang: str = "ru",
) -> str:
    """One source block plus its translation block.

    `img_files` maps source image references to in-EPUB paths.
    """
    en_text = " ".join(bp.en_sentences)
    ru_text = " ".join(bp.ru_sentences)
    source_lang_attr = escape(source_lang, quote=True)
    translation_lang_attr = escape(translation_lang, quote=True)

    image_tags = []
    for img in src.images:
        source_key = _image_key(img.get("src"))
        file_name = img_files.get(source_key)
        if file_name:
            alt = escape(img.get("alt") or "", quote=True)
            image_tags.append(f'<img src="{file_name}" alt="{alt}" />')
    image_block = (
        f'<div class="source-img">{"".join(image_tags)}</div>'
        if image_tags else ""
    )

    if bp.type == "empty-line" and not en_text:
        en_block = (
            f'<p class="source empty-line" lang="{source_lang_attr}">'
            "&#160;</p>"
        )
    elif bp.type == "image" or (src.images and not en_text):
        en_block = image_block
    elif bp.type == "heading":
        en_block = (
            f'<h2 class="source" lang="{source_lang_attr}">'
            f"{escape(en_text)}</h2>"
        )
    else:
        en_block = (
            f'<p class="source {escape(bp.type, quote=True)}" '
            f'lang="{source_lang_attr}">'
            f"<strong>{escape(en_text)}</strong></p>"
        )
        en_block += image_block

    translation_block = (
        f'<p class="translation" lang="{translation_lang_attr}">'
        f"{escape(ru_text)}</p>"
        if ru_text else ""
    )

    if bp.type in _QUOTE_TYPES:
        return (
            f"<blockquote>{en_block}{translation_block}</blockquote>\n"
        )
    return en_block + translation_block + "\n"


def _chapter_xhtml(body: str) -> str:
    return "<body>\n" + body + "</body>\n"


# --- chapter plan ------------------------------------------------------

def _chapter_plan(
    en_book, pairs: list[dict],
) -> list[tuple[Chapter, dict | None]]:
    """Ordered rendering plan over all source chapters.

    Returns (source_chapter, pair_or_None) in the source book's original
    order. Chapters absent from `pairs` (skipped plot chapters,
    untitled front matter, auxiliary bodies) are emitted source-only —
    the source book defines the output's structure, nothing is dropped.
    """
    pair_by_en = {p["en_order"]: p for p in pairs}
    return [(ch, pair_by_en.get(ch.order)) for ch in en_book.chapters]


def _unpaired_reason(chapter: Chapter) -> str:
    if chapter.body != "main":
        return f"auxiliary-body:{chapter.body}"
    if not chapter.title:
        return "untitled-front-matter"
    return "unmatched"


def _en_only_paragraphs(chapter: Chapter) -> list[BilingualParagraph]:
    """Source chapter without a translation: keep its paragraphs and order."""
    return [
        BilingualParagraph(
            index=i,
            type=p.type,
            en_sentences=[s.text for s in p.sentences],
        )
        for i, p in enumerate(chapter.paragraphs)
    ]


# --- embeddings / alignment (same wiring as the experiments) ---------

def _normed(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def _align_chapter_pair(
    en_ch: Chapter, ru_ch: Chapter, encode,
) -> tuple[list[BilingualParagraph], dict[str, int]]:
    en_paragraphs = [[s.text for s in p.sentences] for p in en_ch.paragraphs]
    para_types = [p.type for p in en_ch.paragraphs]
    en_sents = [s for p in en_paragraphs for s in p]
    ru_sents = [s.text for p in ru_ch.paragraphs for s in p.sentences]

    vectors, n_new, _ = embed_with_cache(
        encode, en_sents + ru_sents, EMB_DIR, MODEL_NAME, ENCODE_PARAMS,
    )
    en_joins = [" ".join(en_sents[i:i + 2])
                for i in range(len(en_sents) - 1)]
    ru_joins = [" ".join(ru_sents[j:j + 2])
                for j in range(len(ru_sents) - 1)]
    join_vecs, n_new_j, _ = embed_with_cache(
        encode, en_joins + ru_joins, EMB_DIR, MODEL_NAME, ENCODE_PARAMS,
    )
    print(f"  embeddings: {n_new} new sentences, "
          f"{n_new_j} new joins (rest cached)")

    E, R = vectors[: len(en_sents)], vectors[len(en_sents):]
    EJ = join_vecs[: len(en_joins)]
    RJ = join_vecs[len(en_joins):]
    sim = E @ R.T

    def group_scorer(ei: tuple[int, ...], rj: tuple[int, ...]) -> float:
        a = _normed(E[ei[0]] if len(ei) == 1 else EJ[ei[0]])
        b = _normed(R[rj[0]] if len(rj) == 1 else RJ[rj[0]])
        return float(a @ b)

    ops = align_sentences(en_sents, ru_sents, sim, group_scorer=group_scorer)
    bps = rebuild_chapter(
        en_paragraphs, ru_sents, ops, para_types=para_types,
    )

    rebuilt_en = [s for p in bps for s in p.en_sentences]
    rebuilt_ru = [s for p in bps for s in p.ru_sentences]
    assert rebuilt_en == en_sents and rebuilt_ru == ru_sents, (
        "paragraph reconstruction lost or reordered sentences"
    )
    return bps, summarize_ops(ops)


# --- verification ------------------------------------------------------

def _verify(
    path: Path,
    expected: list[tuple[str, list[BilingualParagraph]]],
    source_label: str = "EN",
    translation_label: str = "RU",
) -> list[str]:
    """Re-open the EPUB and check structure + text integrity."""
    checks: list[str] = []
    book = epub.read_epub(str(path))
    docs = {
        item.get_name(): item
        for item in book.get_items_of_type(ITEM_DOCUMENT)
    }
    item_names = {i.get_name() for i in book.get_items()}

    exp_names = [f"chap_{k + 1}.xhtml" for k in range(len(expected))]
    checks.append((
        "exactly the selected chapters present",
        {n for n in docs if n.startswith("chap_")} == set(exp_names),
    ))
    # Reading order comes from the spine, not from file-name order
    # (lexicographic sort misorders chap_10+ vs chap_2).
    name_by_id = {i.get_id(): i.get_name() for i in book.get_items()}
    spine_chaps = [
        name for idref, _lin in book.spine
        if (name := name_by_id.get(idref)) and name.startswith("chap_")
    ]
    checks.append((
        "chapters in original order (spine)",
        spine_chaps == exp_names,
    ))

    # On re-read ebooklib maps the nav document to an EpubNav item;
    # the manifest "nav" property is not kept on the object.
    nav_ok = any(isinstance(i, epub.EpubNav) for i in book.get_items())
    checks.append(("EPUB3 nav document present", nav_ok))
    checks.append(("NCX present", "toc.ncx" in item_names))

    for k, (title, bps) in enumerate(expected):
        name = exp_names[k]
        item = docs.get(name)
        if item is None:
            checks.append((f"{name} readable", False))
            continue
        soup = BeautifulSoup(item.get_content(), "lxml-xml")
        checks.append((
            f"{name} well-formed XHTML",
            soup.find("body") is not None,
        ))

        en_got = [
            t for el in soup.select(".source")
            if (t := " ".join(el.get_text().split()))
        ]
        ru_got = [
            " ".join(el.get_text().split())
            for el in soup.select(".translation")
        ]
        en_exp = [" ".join(p.en_sentences) for p in bps if p.en_sentences]
        ru_exp = [" ".join(p.ru_sentences) for p in bps if p.ru_sentences]
        checks.append((
            f"{name}: {source_label} text complete, ordered, unduplicated",
            en_got == en_exp,
        ))
        checks.append((
            f"{name}: {translation_label} text complete, ordered, unduplicated",
            ru_got == ru_exp,
        ))

    # Every <img src> in every document must resolve to a packaged file.
    for name, item in docs.items():
        soup = BeautifulSoup(item.get_content(), "lxml-xml")
        for img in soup.find_all("img"):
            checks.append((
                f"{name}: image {img.get('src')} exists",
                (img.get("src") or "") in item_names,
            ))

    return checks


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    source_path = _validate_book_path(args.en, "source")
    translation_path = _validate_book_path(args.ru, "translation")
    out_path = args.output or _output_path(args.all_chapters)

    try:
        source_override = normalize_language(args.source_lang)
        translation_override = normalize_language(args.translation_lang)
        source_book = _parse_book(source_path, source_override)
        translation_book = _parse_book(
            translation_path, translation_override,
        )
        source_lang, translation_lang = resolve_language_pair(
            source_book, translation_book,
            source_override, translation_override,
        )
    except UnsupportedLanguageError as exc:
        raise SystemExit(f"error: {exc}") from None

    source_label = language_name(source_lang)
    translation_label = language_name(translation_lang)
    if not _run_preflight(
        source_book, translation_book, source_label, translation_label,
    ):
        return 2

    alignment = align_chapters(source_book, translation_book)
    all_pairs = alignment["pairs"]
    pairs = all_pairs if args.all_chapters else all_pairs[:N_CHAPTERS]
    if not args.all_chapters and len(pairs) < N_CHAPTERS:
        raise SystemExit("not enough aligned chapter pairs")

    if args.all_chapters:
        # The EN book defines structure: every EN chapter appears, in
        # order; paired ones get a translation, the rest stay EN-only.
        plan = _chapter_plan(source_book, pairs)
    else:
        plan = [(source_book.chapters[p["en_order"]], p) for p in pairs]

    encode = None
    if any(pair is not None for _, pair in plan):
        import torch
        from sentence_transformers import SentenceTransformer

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = SentenceTransformer(
            MODEL_NAME, device=device, cache_folder=str(MODELS_DIR),
        )

        def encode(batch: list[str]) -> np.ndarray:
            kwargs = {
                k: v for k, v in ENCODE_PARAMS.items() if v is not None
            }
            return model.encode(batch, convert_to_numpy=True, **kwargs)

    # --- align / prepare each chapter ---------------------------------
    aligned: list[tuple[Chapter, Chapter | None, list[BilingualParagraph]]] = []
    for source_ch, pair in plan:
        if pair is None:
            print(f"{source_label}-only: {source_ch.id} "
                  f"{source_ch.title!r} [{_unpaired_reason(source_ch)}]")
            aligned.append((
                source_ch, None, _en_only_paragraphs(source_ch),
            ))
            continue
        translation_ch = translation_book.chapters[pair["ru_order"]]
        print(f"Pair: {source_label} {source_ch.id} "
              f"{source_ch.title!r} <-> {translation_label} "
              f"{translation_ch.id} {translation_ch.title!r} "
              f"({len(source_ch.paragraphs)} source paras, "
              f"{len(translation_ch.paragraphs)} translation paras)")
        bps, op_stats = _align_chapter_pair(
            source_ch, translation_ch, encode,
        )
        print(f"  ops: {op_stats}")
        aligned.append((source_ch, translation_ch, bps))

    # --- collect source images actually referenced --------------------
    images, cover_key = _source_assets(source_path)
    image_keys: set[str] = set()
    for source_ch, _, bps in aligned:
        for bp in bps:
            for img in source_ch.paragraphs[bp.index].images:
                image_keys.add(_image_key(img.get("src")))
    img_files = _image_file_names(image_keys, images)

    # --- assemble the EPUB ---------------------------------------------
    out_book = epub.EpubBook()
    out_book.set_identifier(str(uuid.uuid4()))
    mode = "full" if args.all_chapters else "preview"
    out_book.set_title(
        f"{source_book.metadata.get('title', '')} — bilingual {mode}"
    )
    out_book.set_language(source_lang)
    out_book.add_metadata("DC", "language", translation_lang)
    author = source_book.metadata.get("author")
    if author:
        out_book.add_author(author)

    cover_page = None
    if cover_key and cover_key in images:
        media, data = images[cover_key]
        name = _cover_file_name(cover_key, media)
        out_book.set_cover(name, data)
        cover_page = out_book.get_item_with_id("cover")
        print(f"Cover: {name} ({media}, {len(data)} bytes)")
    else:
        print(f"Cover: none in {source_label} source")

    style = epub.EpubItem(
        uid="style", file_name="style.css",
        media_type="text/css", content=_CSS.encode("utf-8"),
    )
    out_book.add_item(style)

    for image_index, (image_key, file_name) in enumerate(img_files.items(), 1):
        media, data = images[image_key]
        out_book.add_item(epub.EpubItem(
            uid=f"img-{image_index}", file_name=file_name,
            media_type=media, content=data,
        ))
        print(f"Image: {file_name} ({media}, {len(data)} bytes)")

    chapters: list[epub.EpubHtml] = []
    expected: list[tuple[str, list[BilingualParagraph]]] = []
    for k, (source_ch, translation_ch, bps) in enumerate(aligned):
        source_title = source_ch.title or "Untitled"
        title = (
            f"{source_title} / {translation_ch.title or 'Untitled'}"
            if translation_ch is not None else source_title
        )
        body = "".join(
            _render_paragraph(
                bp, source_ch.paragraphs[bp.index], img_files,
                source_lang, translation_lang,
            )
            for bp in bps
        )
        item = epub.EpubHtml(
            uid=f"chap_{k + 1}", file_name=f"chap_{k + 1}.xhtml",
            title=title, lang=source_lang,
        )
        item.add_item(style)
        item.content = _chapter_xhtml(body).encode("utf-8")
        out_book.add_item(item)
        chapters.append(item)
        expected.append((title, bps))

    out_book.toc = tuple(chapters)
    out_book.add_item(epub.EpubNcx())
    out_book.add_item(epub.EpubNav())
    out_book.spine = [x for x in (cover_page, "nav", *chapters) if x]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    epub.write_epub(str(out_path), out_book)

    # --- verification ----------------------------------------------------
    checks = _verify(
        out_path, expected, source_label, translation_label,
    )
    ok = True
    for label, passed in checks:
        ok &= passed
        print(f"  [{'OK' if passed else 'FAIL'}] {label}")
    if not ok:
        print("\nConversion failed: EPUB verification did not pass.")
        return 1

    print(
        f"\nSuccessfully converted: {out_path} "
        f"({out_path.stat().st_size} bytes)"
    )

    # --- summary ----------------------------------------------------------
    print("\nPer-chapter summary:")
    totals = {
        "paired": 0, "source_only": 0,
        "source_s": 0, "translation_s": 0, "bare": 0,
    }
    for (source_ch, translation_ch, bps), (title, _) in zip(aligned, expected):
        source_s = sum(len(p.en_sentences) for p in bps)
        translation_s = sum(len(p.ru_sentences) for p in bps)
        bare = sum(1 for p in bps if p.en_sentences and not p.ru_sentences)
        if translation_ch is None:
            mark = f"  [{source_label}-only: {_unpaired_reason(source_ch)}]"
            totals["source_only"] += 1
        else:
            mark = ""
            totals["paired"] += 1
        totals["source_s"] += source_s
        totals["translation_s"] += translation_s
        totals["bare"] += bare
        print(f"  {title!r}: {len(bps)} paragraphs, "
              f"{source_s} {source_label} / "
              f"{translation_s} {translation_label} sentences, "
              f"{bare} source paragraphs without translation{mark}")
    print(
        f"TOTAL: {len(aligned)} chapters "
        f"({totals['paired']} paired, "
        f"{totals['source_only']} {source_label}-only), "
        f"{totals['source_s']} {source_label} / "
        f"{totals['translation_s']} {translation_label} sentences, "
        f"{totals['bare']} source paragraphs without translation"
    )

    doubtful = [p for p in pairs if p["quality"] == "doubtful"]
    print(f"\nDoubtful chapter pairs: {len(doubtful)}")
    for p in doubtful:
        print(f"  {source_label} {p['en_chapter_id']} "
              f"{p['en_title']!r} <-> {translation_label} "
              f"{p['ru_chapter_id']} {p['ru_title']!r} "
              f"[{', '.join(p['reasons'])}]")

    if args.all_chapters:
        # Translation chapters without a source counterpart are not in the EPUB —
        # surfaced here instead of being dropped silently.
        missing_ru = (
            alignment["unmatched"]["russian"]
            + alignment["service"]["russian"]
        )
        print(
            f"\n{translation_label} chapters not in the EPUB: "
            f"{len(missing_ru)}"
        )
        for c in missing_ru:
            reason = c.get("service_reason") or "unmatched"
            print(f"  {c['chapter_id']} {c['title']!r} "
                  f"({c['words']} w) [{reason}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
