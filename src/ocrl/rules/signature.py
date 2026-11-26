"""Ink signature: the satellite marks each character must bring with it.

A letter is a skeleton stroke (rasm) plus a fixed set of dots or a hamza. That
set is known from the text alone, so a recognised string predicts how many
satellite blobs its cluster should hold. Dots may fuse with each other in a
given font (two dots drawn as a bar), so the prediction is a range.
"""

from __future__ import annotations

from .joining import is_transparent, positional_forms

# letter -> (number of dots, 'above' | 'below' | None)
DOTS: dict[str, tuple[int, str | None]] = {
    "ب": (1, "below"), "ت": (2, "above"), "ث": (3, "above"), "ن": (1, "above"),
    "ي": (2, "below"), "ج": (1, "below"), "خ": (1, "above"), "ذ": (1, "above"),
    "ز": (1, "above"), "ش": (3, "above"), "ض": (1, "above"), "ظ": (1, "above"),
    "غ": (1, "above"), "ف": (1, "above"), "ق": (2, "above"), "ة": (2, "above"),
}

# letter -> a non-dot mark that is always a separate blob (hamza, madda)
MARKS: dict[str, str] = {"أ": "above", "إ": "below", "آ": "above", "ؤ": "above", "ئ": "above"}

# Punctuation drawn as a body plus detached pieces.
PUNCT_SATELLITES: dict[str, int] = {":": 1, "؛": 1, "؟": 1, "!": 1, "?": 1, ";": 1, "%": 2, "i": 1, "j": 1}

VOWEL_MARKS = "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652"

# Dotless skeleton classes: letters that are the same stroke once dots are removed.
RASM: dict[str, str] = {
    **dict.fromkeys("اأإآ", "ا"), **dict.fromkeys("بتث", "ب"), **dict.fromkeys("جحخ", "ح"),
    **dict.fromkeys("دذ", "د"), **dict.fromkeys("رز", "ر"), **dict.fromkeys("سش", "س"),
    **dict.fromkeys("صض", "ص"), **dict.fromkeys("طظ", "ط"), **dict.fromkeys("عغ", "ع"),
    **dict.fromkeys("وؤ", "و"), **dict.fromkeys("هة", "ه"), **dict.fromkeys("يىئ", "ى"),
    "ف": "ف", "ق": "ق", "ك": "ك", "ل": "ل", "م": "م", "ن": "ن", "ء": "ء",
}


def satellite_range(unit: str) -> tuple[int, int]:
    """(fewest, most) satellite blobs a clean rendering of `unit` can show."""
    lo = hi = 0
    forms = positional_forms(unit)
    prev_mark = False
    for ch, form in zip(unit, forms):
        if ch in VOWEL_MARKS:
            # A vowel mark is a blob of its own; two on one letter (shadda + vowel) may fuse.
            lo, hi = lo + (0 if prev_mark else 1), hi + 1
            prev_mark = True
            continue
        prev_mark = False
        if is_transparent(ch):
            continue
        n, _ = DOTS.get(ch, (0, None))
        if n:
            lo, hi = lo + 1, hi + n
        if ch in MARKS:
            lo, hi = lo + 1, hi + 1
        if ch == "ك" and form in ("isolated", "final"):
            hi += 1  # the small mark inside a final kaf; some fonts fuse it
        if ch in PUNCT_SATELLITES:
            lo, hi = lo + PUNCT_SATELLITES[ch], hi + PUNCT_SATELLITES[ch]
    return lo, hi


def dot_units(unit: str) -> int:
    """Total dots the text of `unit` calls for."""
    return sum(DOTS.get(c, (0, None))[0] for c in unit)
