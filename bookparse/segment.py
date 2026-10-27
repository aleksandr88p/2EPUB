"""Sentence splitting and word counting.

Rule-based splitter that works for English and Russian without heavy
dependencies. It splits on sentence-ending punctuation followed by
whitespace, and refuses to split after common abbreviations, initials
("J." in "J. R. R. Tolkien") and before a lowercase continuation.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[\w]+(?:[’'\-][\w]+)*", re.UNICODE)

_BOUNDARY_RE = re.compile(r"[.!?…]+[”’\"'»)\]]*(?=\s|$)")
_CLOSERS = "”’\"'»)]"
_TOKEN_RE = re.compile(r"[\w.]+['’\w]*$")
_INITIALS_RE = re.compile(r"(?:[A-ZА-ЯЁ]\.)+$")

EN_ABBREV = {
    "mr", "mrs", "ms", "dr", "st", "mt", "jr", "sr", "prof", "vs", "etc",
    "e.g", "i.e", "no", "fig", "al", "dept", "univ", "assn", "bros",
    "inc", "ltd", "co", "rev", "gen", "col", "lt", "capt", "maj", "sgt",
    "approx", "est", "min", "max", "misc", "ave", "blvd", "rd", "ph.d",
    "u.s", "u.k", "vol", "pp", "ed", "eds",
}

RU_ABBREV = {
    "т.д", "т.п", "т.е", "т.к", "т.н", "т.о", "т.г", "и.о", "и.т.д",
    "г", "гг", "с", "стр", "гл", "рис", "ил", "св", "им", "др", "ч",
    "см", "н.э", "с.ш", "в.д", "ю.ш", "руб", "коп", "долл", "тыс", "млн",
}


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _last_token(chunk: str) -> str:
    match = _TOKEN_RE.search(chunk.strip())
    return match.group(0) if match else ""


def _is_boundary(text: str, end: int, chunk: str, punct: str, abbrevs: set[str]) -> bool:
    token = _last_token(chunk)
    if token:
        if _INITIALS_RE.fullmatch(token):
            return False
        if token.lower().rstrip(".") in abbrevs:
            return False
    # "." or "…" followed by a lowercase letter is usually mid-sentence.
    if punct in ".…":
        tail = text[end:].lstrip()
        if tail and tail[0].islower():
            return False
    return True


def split_sentences(text: str, lang: str | None = None) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    abbrevs = EN_ABBREV | RU_ABBREV
    if lang == "en":
        abbrevs = EN_ABBREV
    elif lang == "ru":
        abbrevs = RU_ABBREV

    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY_RE.finditer(text):
        end = match.end()
        punct = match.group(0).rstrip(_CLOSERS)[-1]
        chunk = text[start:end]
        if not _is_boundary(text, end, chunk, punct, abbrevs):
            continue
        sentences.append(chunk.strip())
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences
