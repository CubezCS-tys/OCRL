"""Text handling shared by the renderer and the recogniser: charset, cleaning, runs."""

from __future__ import annotations

import re
import string
from pathlib import Path

LETTERS = "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي"
DIGITS = "0123456789٠١٢٣٤٥٦٧٨٩"
MARKS = "ًٌٍَُِّْ"  # tanwin x3, fatha, damma, kasra, shadda, sukun
LATIN = string.ascii_letters
PUNCT = ".،؛؟!:(),-/%[]"
CHARSET = LETTERS + DIGITS + PUNCT + MARKS + LATIN
# The first recogniser was trained before marks, Latin and the extra punctuation existed.
LEGACY_CHARSET = LETTERS + DIGITS + ".،؛؟!:(),"

TATWEEL = "ـ"
NBSP = " "  # holds a Latin phrase together so it is laid out left to right as one piece

_MARKS = re.compile("[ً-ْٰـ]")  # vowel marks and tatweel
_RUN = re.compile(
    f"(?P<ar>[{LETTERS}][{LETTERS}{MARKS}{TATWEEL}]*)"
    f"|(?P<num>[{DIGITS}]+(?:[.,][{DIGITS}]+)*)"
    f"|(?P<lat>[A-Za-z]+(?:[{NBSP}\\-]?[A-Za-z]+)*)"
    f"|(?P<p>[{re.escape(PUNCT)}])"
)
_LATIN_WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")


def strip_marks(text: str) -> str:
    """Remove vowel marks and tatweel: the bare letters, as most print carries them."""
    return _MARKS.sub("", text)


def strip_tatweel(text: str) -> str:
    """Tatweel is a stretch of the pen, not a character: it is never part of a reading."""
    return text.replace(TATWEEL, "")


def is_digit(ch: str) -> bool:
    return ch in DIGITS


def is_ltr(ch: str) -> bool:
    return ch in LATIN


def runs(word: str) -> list[tuple[str, str]]:
    """Split a space-free word into directional runs: ('ar' | 'num' | 'lat' | 'p', text).

    Returns [] if the word contains anything outside the supported charset.
    """
    out, pos = [], 0
    for m in _RUN.finditer(word):
        if m.start() != pos:
            return []
        out.append((m.lastgroup, m.group()))
        pos = m.end()
    return out if pos == len(word) else []


def join_latin(words: list[str]) -> list[str]:
    """Glue neighbouring Latin words into one phrase, as the bidi algorithm would display them."""
    out: list[str] = []
    for w in words:
        if out and _LATIN_WORD.fullmatch(w) and _LATIN_WORD.fullmatch(out[-1].split(NBSP)[-1]):
            out[-1] += NBSP + w
        else:
            out.append(w)
    return out


def clean_sentence(text: str) -> str:
    """Drop marks and any word we cannot render faithfully."""
    words = [w for w in strip_marks(text).split() if runs(w)]
    return " ".join(words)


def load_corpus(path: str | Path, min_words: int = 4) -> list[str]:
    """Leipzig sentence file ('id<TAB>sentence') -> cleaned sentences."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        sent = clean_sentence(line.split("\t", 1)[-1])
        if len(sent.split()) >= min_words:
            out.append(sent)
    return out
