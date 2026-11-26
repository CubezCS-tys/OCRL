"""Script-level rules of Arabic: who joins whom, and what follows from it.

Nothing here is learned. The joining class of every character comes straight
from Unicode's ArabicShaping.txt, and the rest is derived from one predicate:

    joins(a, b)  --  does letter `a` connect to the letter `b` that follows it?

A *unit* is a maximal chain of joined characters. For Arabic letters that is a
sub-word (PAW); for digits and punctuation it is the single character. On clean
ink every unit is exactly one connected body blob, which is what lets the page's
connected components serve as the map.
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache
from pathlib import Path

# Joining types, as named by Unicode.
RIGHT = "R"  # joins to the previous letter only: ا د ذ ر ز و ة ى ...
DUAL = "D"  # joins on both sides: ب ت س ل م ...
CAUSING = "C"  # tatweel: joins on both sides, carries no letter
NONJOINING = "U"  # ء, digits, punctuation, anything non-Arabic
TRANSPARENT = "T"  # vowel marks: invisible to joining
LEFT = "L"  # unused by Arabic, kept for completeness

TATWEEL = "ـ"
HAMZA = "ء"


def _load_table() -> dict[str, str]:
    table: dict[str, str] = {}
    path = Path(__file__).with_name("ArabicShaping.txt")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        code, _name, jtype, _group = (f.strip() for f in line.split(";"))
        table[chr(int(code, 16))] = jtype
    return table


_TABLE = _load_table()


@lru_cache(maxsize=None)
def joining_type(ch: str) -> str:
    """Unicode joining type of a single character."""
    jt = _TABLE.get(ch)
    if jt is not None:
        return jt
    # Unlisted characters: marks and format controls are transparent, the rest don't join.
    return TRANSPARENT if unicodedata.category(ch) in ("Mn", "Me", "Cf") else NONJOINING


def is_transparent(ch: str) -> bool:
    return joining_type(ch) == TRANSPARENT


def joins(a: str, b: str) -> bool:
    """True if base character `a` connects to the base character `b` after it."""
    return joining_type(a) in (DUAL, CAUSING, LEFT) and joining_type(b) in (DUAL, CAUSING, RIGHT)


def unit_spans(token: str) -> list[tuple[int, int]]:
    """Split a space-free token into units; returns [start, end) character spans.

    Transparent marks stay with the base character before them.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    prev_base: str | None = None
    for i, ch in enumerate(token):
        if is_transparent(ch):
            continue
        if prev_base is not None and not joins(prev_base, ch):
            spans.append((start, i))
            start = i
        prev_base = ch
    if token:
        spans.append((start, len(token)))
    return spans


def units(token: str) -> list[str]:
    return [token[a:b] for a, b in unit_spans(token)]


def text_units(text: str) -> list[str]:
    """Units of a whole line of text, in logical order, spaces dropped."""
    return [u for word in text.split() for u in units(word)]


def bases(unit: str) -> list[str]:
    """The letters of a unit: no marks, no tatweel."""
    return [c for c in unit if not is_transparent(c) and c != TATWEEL]


def is_valid_unit(unit: str) -> bool:
    """Could this string be a single connected body?

    Every letter but the last must be dual-joining, and every letter but the
    first must accept a join from the right. A decoder that wants to emit a
    string failing this test is looking at more than one body.
    """
    letters = [c for c in unit if not is_transparent(c)]
    if not letters:
        return False
    return all(joins(a, b) for a, b in zip(letters, letters[1:]))


def first_violation(unit: str) -> int | None:
    """Index of the first letter that cannot follow its predecessor in one body."""
    prev: tuple[int, str] | None = None
    for i, ch in enumerate(unit):
        if is_transparent(ch):
            continue
        if prev is not None and not joins(prev[1], ch):
            return i
        prev = (i, ch)
    return None


def forces_word_break(unit_a: str, unit_b: str) -> bool:
    """Two adjacent, separate bodies whose facing letters *would* have joined.

    Had they been in one word they would be one blob; they are not, so a word
    boundary lies between them. Says nothing (returns False) when the first
    ends in a non-joiner or the second starts with one.
    """
    a, b = bases(unit_a), bases(unit_b)
    return bool(a and b) and joins(a[-1], b[0])


FORMS = ("isolated", "initial", "medial", "final")


def positional_forms(token: str) -> list[str | None]:
    """Shape of each character in a token; None for transparent marks."""
    idx = [i for i, c in enumerate(token) if not is_transparent(c)]
    out: list[str | None] = [None] * len(token)
    for k, i in enumerate(idx):
        prev_j = k > 0 and joins(token[idx[k - 1]], token[i])
        next_j = k + 1 < len(idx) and joins(token[i], token[idx[k + 1]])
        out[i] = FORMS[(2 if prev_j and next_j else 3 if prev_j else 1 if next_j else 0)]
    return out
