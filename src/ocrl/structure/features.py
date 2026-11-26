"""What the structure model is told about a line: where it sits, how big it is, and a few of its characters.

The same code prepares lines from the Desk database (training) and lines found by OCRL itself
(use), so the two cannot drift apart.
"""

from __future__ import annotations

import numpy as np

from ..text import strip_marks

ROLES = [
    "paragraph", "heading", "list-item", "ref", "footnote", "endnote", "caption", "table-cell", "verse-group", "abstract",
    "title", "title-translated", "byline", "front-matter", "back-matter", "keywords", "quote",
    "pageHeader", "pageNumber", "pageFooter", "noise", "pictureText", "pictureJunk", "ornament", "otherArticle",
    "formulaInk", "showThrough",
]
ROLE_ID = {r: i for i, r in enumerate(ROLES)}
FIRST, LAST = 24, 10  # characters kept from the start and the end of a line
N_GEOM = 12


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


def geometry(boxes: np.ndarray, n_chars: np.ndarray, page_w: float, page_h: float) -> np.ndarray:
    """boxes: [n, 4] x0, y0, x1, y1 in pixels. Returns [n, N_GEOM] numbers that do not depend on scan size."""
    x0, y0, x1, y1 = (boxes[:, i].astype(np.float32) for i in range(4))
    w, h = x1 - x0, y1 - y0
    med_h = float(np.median(h)) if len(h) else 1.0
    text_l, text_r = (float(np.percentile(x0, 5)), float(np.percentile(x1, 95))) if len(h) else (0.0, page_w)
    span = max(text_r - text_l, 1.0)
    return np.stack([
        x0 / page_w, x1 / page_w, y0 / page_h, y1 / page_h,   # where on the page
        w / page_w, h / max(med_h, 1.0),                      # how wide, and letter size against the page's usual
        (x0 - text_l) / span, (text_r - x1) / span,           # inset from the text's left and right edges
        np.log1p(n_chars) / 5.0, np.minimum(w / np.maximum(n_chars, 1) / max(med_h, 1.0), 3.0),  # length, and width per character
        np.full_like(x0, page_w / page_h), np.full_like(x0, min(len(h), 200) / 200.0),          # page shape, how full the page is
    ], axis=1).astype(np.float32)
