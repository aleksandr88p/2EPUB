"""Build a bilingual EPUB from English and Russian EPUB or FB2 books.

Two modes:

  * preview (default): the first two aligned chapter pairs ->
    output/2epub_preview.epub
  * full (--all): every aligned pair plus every EN chapter that has
    no RU counterpart (front matter, unmatched plot chapters,
    auxiliary bodies), in the EN book's original order ->
    output/2epub_full.epub

Both run the existing pipeline end to end:

    chapter alignment -> Qwen embeddings (disk cache) -> DP sentence
    alignment with the joined-text group scorer -> paragraph
    reconstruction -> EPUB assembly.

Layout contract: the EN book defines structure and chapter order.
Every EN paragraph is rendered bold, its aligned RU sentences follow
in a normal paragraph. EN cover and body images are carried over;
RU images are not included yet. Paragraphs without a translation are
left bare — no placeholder text is invented. RU chapters without an
EN counterpart are not added to the EPUB (the EN book defines its
structure); they are listed in the run report instead.

Run:  venv/bin/python experiments/build_preview_epub.py [--all]
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
from bookparse.embedding_cache import embed_with_cache  # noqa: E402
from bookparse.epub_parser import parse_epub  # noqa: E402
from bookparse.fb2_parser import parse_fb2  # noqa: E402
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
        description="Build a bilingual EPUB from English and Russian "
                    "EPUB or FB2 books.",
    )
    parser.add_argument(
        "--all", action="store_true", dest="all_chapters",
        help="process every aligned chapter pair plus all EN chapters "
             "without an RU counterpart (default: first %d pairs only)"
             % N_CHAPTERS,
    )
    parser.add_argument(
        "--en", type=Path, required=True, metavar="PATH",
        help="path to the English original (.fb2 or .epub)",
    )
    parser.add_argument(
        "--ru", type=Path, required=True, metavar="PATH",
        help="path to the Russian translation (.fb2 or .epub)",
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


def _parse_book(path: Path) -> Book:
    """Select the existing parser from the input file extension."""
    if path.suffix.lower() == ".fb2":
        return parse_fb2(path)
    if path.suffix.lower() == ".epub":
        return parse_epub(path)
    raise ValueError(f"unsupported book format: {path.suffix}")


_CSS = """\
p.en { font-weight: bold; margin: 0.4em 0 0 0; }
h2.en { margin: 1em 0 0.3em 0; }
p.ru { margin: 0.15em 0 0.9em 0; }
blockquote { margin: 0.4em 1.2em; }
p.empty-line { margin: 0.6em 0; }
div.en-img { text-align: center; margin: 0.6em 0; }
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
) -> str:
    """One EN block + its RU translation block.

    `img_files` maps source image references to in-EPUB paths.
    """
    en_text = " ".join(bp.en_sentences)
    ru_text = " ".join(bp.ru_sentences)

    image_tags = []
    for img in src.images:
        source_key = _image_key(img.get("src"))
        file_name = img_files.get(source_key)
        if file_name:
            alt = escape(img.get("alt") or "", quote=True)
            image_tags.append(f'<img src="{file_name}" alt="{alt}" />')
    image_block = (
        f'<div class="en-img">{"".join(image_tags)}</div>'
        if image_tags else ""
    )

    if bp.type == "empty-line" and not en_text:
        en_block = '<p class="en empty-line">&#160;</p>'
    elif bp.type == "image" or (src.images and not en_text):
        en_block = image_block
    elif bp.type == "heading":
        en_block = f'<h2 class="en">{escape(en_text)}</h2>'
    else:
        en_block = (
            f'<p class="en {escape(bp.type, quote=True)}">'
            f"<strong>{escape(en_text)}</strong></p>"
        )
        en_block += image_block

    ru_block = f'<p class="ru">{escape(ru_text)}</p>' if ru_text else ""

    if bp.type in _QUOTE_TYPES:
        return f"<blockquote>{en_block}{ru_block}</blockquote>\n"
    return en_block + ru_block + "\n"


def _chapter_xhtml(body: str) -> str:
    return "<body>\n" + body + "</body>\n"


# --- chapter plan ------------------------------------------------------

def _chapter_plan(
    en_book, pairs: list[dict],
) -> list[tuple[Chapter, dict | None]]:
    """Ordered rendering plan over ALL EN chapters.

    Returns (en_chapter, pair_or_None) in the EN book's original
    order. Chapters absent from `pairs` (skipped plot chapters,
    untitled front matter, auxiliary bodies) are emitted EN-only —
    the EN book defines the output's structure, nothing is dropped.
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
    """EN chapter without an RU counterpart: paragraphs keep their
    text and order, no translation is attached."""
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
            t for el in soup.select(".en")
            if (t := " ".join(el.get_text().split()))
        ]
        ru_got = [
            " ".join(el.get_text().split())
            for el in soup.select(".ru")
        ]
        en_exp = [" ".join(p.en_sentences) for p in bps if p.en_sentences]
        ru_exp = [" ".join(p.ru_sentences) for p in bps if p.ru_sentences]
        checks.append((
            f"{name}: EN text complete, ordered, unduplicated",
            en_got == en_exp,
        ))
        checks.append((
            f"{name}: RU text complete, ordered, unduplicated",
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
    en_path = _validate_book_path(args.en, "English")
    ru_path = _validate_book_path(args.ru, "Russian")
    out_path = args.output or _output_path(args.all_chapters)

    en_book = _parse_book(en_path)
    ru_book = _parse_book(ru_path)
    alignment = align_chapters(en_book, ru_book)
    all_pairs = alignment["pairs"]
    pairs = all_pairs if args.all_chapters else all_pairs[:N_CHAPTERS]
    if not args.all_chapters and len(pairs) < N_CHAPTERS:
        raise SystemExit("not enough aligned chapter pairs")

    if args.all_chapters:
        # The EN book defines structure: every EN chapter appears, in
        # order; paired ones get a translation, the rest stay EN-only.
        plan = _chapter_plan(en_book, pairs)
    else:
        plan = [(en_book.chapters[p["en_order"]], p) for p in pairs]

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
    for en_ch, pair in plan:
        if pair is None:
            print(f"EN-only: {en_ch.id} {en_ch.title!r} "
                  f"[{_unpaired_reason(en_ch)}]")
            aligned.append((en_ch, None, _en_only_paragraphs(en_ch)))
            continue
        ru_ch = ru_book.chapters[pair["ru_order"]]
        print(f"Pair: {en_ch.id} {en_ch.title!r} <-> "
              f"{ru_ch.id} {ru_ch.title!r} "
              f"({len(en_ch.paragraphs)} EN paras, "
              f"{len(ru_ch.paragraphs)} RU paras)")
        bps, op_stats = _align_chapter_pair(en_ch, ru_ch, encode)
        print(f"  ops: {op_stats}")
        aligned.append((en_ch, ru_ch, bps))

    # --- collect EN images actually referenced ------------------------
    images, cover_key = _source_assets(en_path)
    image_keys: set[str] = set()
    for en_ch, _, bps in aligned:
        for bp in bps:
            for img in en_ch.paragraphs[bp.index].images:
                image_keys.add(_image_key(img.get("src")))
    img_files = _image_file_names(image_keys, images)

    # --- assemble the EPUB ---------------------------------------------
    out_book = epub.EpubBook()
    out_book.set_identifier(str(uuid.uuid4()))
    mode = "full" if args.all_chapters else "preview"
    out_book.set_title(
        f"{en_book.metadata.get('title', '')} — bilingual {mode}"
    )
    out_book.set_language("en")
    out_book.add_metadata("DC", "language", "ru")
    author = en_book.metadata.get("author")
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
        print("Cover: none in EN source")

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
    for k, (en_ch, ru_ch, bps) in enumerate(aligned):
        en_t = en_ch.title or "Untitled"
        title = (
            f"{en_t} / {ru_ch.title or 'Untitled'}"
            if ru_ch is not None else en_t
        )
        body = "".join(
            _render_paragraph(bp, en_ch.paragraphs[bp.index], img_files)
            for bp in bps
        )
        item = epub.EpubHtml(
            uid=f"chap_{k + 1}", file_name=f"chap_{k + 1}.xhtml",
            title=title, lang="en",
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
    print(f"\nWritten: {out_path} ({out_path.stat().st_size} bytes)")

    # --- verification ----------------------------------------------------
    checks = _verify(out_path, expected)
    ok = True
    for label, passed in checks:
        ok &= passed
        print(f"  [{'OK' if passed else 'FAIL'}] {label}")

    # --- summary ----------------------------------------------------------
    print("\nPer-chapter summary:")
    totals = {"paired": 0, "en_only": 0, "en_s": 0, "ru_s": 0, "bare": 0}
    for (en_ch, ru_ch, bps), (title, _) in zip(aligned, expected):
        en_s = sum(len(p.en_sentences) for p in bps)
        ru_s = sum(len(p.ru_sentences) for p in bps)
        bare = sum(1 for p in bps if p.en_sentences and not p.ru_sentences)
        if ru_ch is None:
            mark = f"  [EN-only: {_unpaired_reason(en_ch)}]"
            totals["en_only"] += 1
        else:
            mark = ""
            totals["paired"] += 1
        totals["en_s"] += en_s
        totals["ru_s"] += ru_s
        totals["bare"] += bare
        print(f"  {title!r}: {len(bps)} paragraphs, "
              f"{en_s} EN / {ru_s} RU sentences, "
              f"{bare} EN paragraphs without translation{mark}")
    print(
        f"TOTAL: {len(aligned)} chapters "
        f"({totals['paired']} paired, {totals['en_only']} EN-only), "
        f"{totals['en_s']} EN / {totals['ru_s']} RU sentences, "
        f"{totals['bare']} EN paragraphs without translation"
    )

    doubtful = [p for p in pairs if p["quality"] == "doubtful"]
    print(f"\nDoubtful chapter pairs: {len(doubtful)}")
    for p in doubtful:
        print(f"  {p['en_chapter_id']} {p['en_title']!r} <-> "
              f"{p['ru_chapter_id']} {p['ru_title']!r} "
              f"[{', '.join(p['reasons'])}]")

    if args.all_chapters:
        # RU chapters without an EN counterpart are not in the EPUB —
        # surfaced here instead of being dropped silently.
        missing_ru = (
            alignment["unmatched"]["russian"]
            + alignment["service"]["russian"]
        )
        print(f"\nRU chapters not in the EPUB: {len(missing_ru)}")
        for c in missing_ru:
            reason = c.get("service_reason") or "unmatched"
            print(f"  {c['chapter_id']} {c['title']!r} "
                  f"({c['words']} w) [{reason}]")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
