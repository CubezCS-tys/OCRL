"""What the structure model is told about a line: where it sits, how big it is, and a few of its characters.

The same code prepares lines from the Desk database (training) and lines found by OCRL itself
(use), so the two cannot drift apart.
"""

from __future__ import annotations

import re

import numpy as np

from ..text import DIGITS, strip_marks

ROLES = [
    "paragraph", "heading", "list-item", "ref", "footnote", "endnote", "caption", "table-cell", "verse-group", "abstract",
    "title", "title-translated", "byline", "front-matter", "back-matter", "keywords", "quote",
    "pageHeader", "pageNumber", "pageFooter", "noise", "pictureText", "pictureJunk", "ornament", "otherArticle",
    "formulaInk", "showThrough",
]
ROLE_ID = {r: i for i, r in enumerate(ROLES)}
FIRST, LAST = 24, 10  # characters kept from the start and the end of a line
N_GEOM = 26

_BULLET = re.compile(r"^\s*[-–—•*·▪◦]\s")
_NUMBERED = re.compile(rf"^\s*\(?\s?[{DIGITS}]{{1,3}}\s?[-.)ـ:]|^\s*\([{DIGITS}\s]{{1,5}}\)|^\s*[أ-ي]\s?[-)]\s")


def text_ids(text: str, vocab: dict[str, int]) -> np.ndarray:
    """First and last characters of a line as ids; 0 pads, 1 is any character the vocabulary lacks."""
    t = strip_marks(text).strip()
    out = np.zeros(FIRST + LAST, np.int16)
    head = t[:FIRST]
    tail = t[-LAST:] if len(t) > FIRST else ""
    out[: len(head)] = [vocab.get(c, 1) for c in head]
    if tail:
        out[FIRST : FIRST + len(tail)] = [vocab.get(c, 1) for c in tail]
    return out


def geometry(boxes: np.ndarray, texts: list[str], page_w: float, page_h: float) -> np.ndarray:
    """What a line looks like, and how it sits against the lines read just before and after it.

    boxes: [n, 4] x0, y0, x1, y1 in pixels, in reading order. Returns [n, N_GEOM] numbers that do
    not depend on scan size.
    """
    n = len(texts)
    clean = [strip_marks(t).strip() for t in texts]
    n_chars = np.array([len(t) for t in clean], np.float32)
    x0, y0, x1, y1 = (boxes[:, i].astype(np.float32) for i in range(4))
    w, h = x1 - x0, y1 - y0
    med_h = max(float(np.median(h)) if n else 1.0, 1.0)
    text_l, text_r = (float(np.percentile(x0, 5)), float(np.percentile(x1, 95))) if n else (0.0, page_w)
    span = max(text_r - text_l, 1.0)
    own = [
        x0 / page_w, x1 / page_w, y0 / page_h, y1 / page_h,   # where on the page
        w / page_w, h / med_h,                                # how wide, and letter size against the page's usual
        (x0 - text_l) / span, (text_r - x1) / span,           # inset from the text's left and right edges
        np.log1p(n_chars) / 5.0, np.minimum(w / np.maximum(n_chars, 1) / med_h, 3.0),  # length, and width per character
        np.full(n, page_w / page_h, np.float32), np.full(n, min(n, 200) / 200.0, np.float32),  # page shape, how full the page is
    ]
    # Against the neighbours in reading order: a paragraph's first line is indented against the next,
    # a heading has air above it, a new column starts far above where the last one ended.
    def against(shift: int) -> list[np.ndarray]:
        j = np.clip(np.arange(n) + shift, 0, max(n - 1, 0))
        there = ((np.arange(n) + shift >= 0) & (np.arange(n) + shift < n)).astype(np.float32)
        gap = (y0 - y1[j]) if shift < 0 else (y0[j] - y1)
        return [there, np.clip(gap / med_h, -4, 4) * there, np.clip((x1 - x1[j]) / span, -1, 1) * there,
                np.clip((x0 - x0[j]) / span, -1, 1) * there]
    marks = [
        np.array([bool(_BULLET.match(t)) for t in clean], np.float32),      # opens with a dash or bullet
        np.array([bool(_NUMBERED.match(t)) for t in clean], np.float32),    # opens with a number or letter marker
        np.array([t.endswith(":") for t in clean], np.float32),
        np.array([t.endswith((".", "؟", "!")) for t in clean], np.float32),
        np.array([bool(t) and sum(c in DIGITS for c in t) > 0.6 * len(t) for t in clean], np.float32),  # mostly digits
        np.array([bool(t) and sum(c.isascii() and c.isalpha() for c in t) > 0.5 * len(t) for t in clean], np.float32),  # mostly Latin
    ]
    return np.stack(own + against(-1) + against(1) + marks, axis=1).astype(np.float32)
