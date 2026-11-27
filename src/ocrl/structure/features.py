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


N_INK = 3
INK_WIDTH = 1150  # every page is measured at this width, so the numbers mean the same for any scan


def ink_features(gray: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """What the ink of each line looks like: [n, N_INK] numbers the boxes and the text cannot give.

    gray: the page (dark ink on light paper); boxes [n, 4] in its pixels. For each line:
      how heavy its strokes are against the page's usual line (bold type is heavier),
      how much of its box is ink against the page's usual line,
      how heavy its strokes are for its own height (bold for its size, whatever the size).
    Stroke weight is the mean distance from an ink pixel to the paper, measured on the page scaled to a fixed width.
    """
    import cv2
    n = len(boxes)
    if n == 0:
        return np.zeros((0, N_INK), np.float32)
    s = INK_WIDTH / gray.shape[1]
    g = cv2.resize(gray, (INK_WIDTH, max(int(round(gray.shape[0] * s)), 8)), interpolation=cv2.INTER_AREA)
    _, ink = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    stroke, dens, hh = np.zeros(n, np.float32), np.zeros(n, np.float32), np.ones(n, np.float32)
    for k, (x0, y0, x1, y1) in enumerate(np.asarray(boxes, np.float32) * s):
        a, b, c, d = max(int(y0) + 1, 0), max(int(y1) - 1, int(y0) + 2), max(int(x0) + 1, 0), max(int(x1) - 1, int(x0) + 2)
        m = ink[a:b, c:d]
        hh[k] = max(y1 - y0, 1.0)
        if m.size and m.sum() >= 6:
            stroke[k] = float(dist[a:b, c:d][m > 0].mean()); dens[k] = float(m.mean())
    med_s = float(np.median(stroke[stroke > 0])) if (stroke > 0).any() else 1.0
    med_d = float(np.median(dens[dens > 0])) if (dens > 0).any() else 1.0
    return np.stack([np.clip(stroke / max(med_s, 1e-3), 0, 3), np.clip(dens / max(med_d, 1e-3), 0, 3), np.clip(10 * stroke / hh, 0, 3)], axis=1).astype(np.float32)


CROP_H, CROP_W = 32, 320


def line_crops(gray: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """A small picture of every line, [n, CROP_H, CROP_W] uint8, ink dark on white.

    The page is first scaled to INK_WIDTH, so a letter is the same number of pixels whatever the scan;
    each line is then cut out with a little margin and shrunk to CROP_H high. A line too long for the
    picture is squeezed to fit, which keeps both ends in view (where a list mark or a full stop sits);
    a short one keeps its true width and sits at the right, as Arabic starts there. What the picture
    shows and the box and text cannot: bold or light type, the size of the letters, a bullet, an
    underline, numbers set in a row.
    """
    import cv2
    n = len(boxes)
    out = np.full((n, CROP_H, CROP_W), 255, np.uint8)
    if n == 0:
        return out
    s = INK_WIDTH / gray.shape[1]
    g = cv2.resize(gray, (INK_WIDTH, max(int(round(gray.shape[0] * s)), 8)), interpolation=cv2.INTER_AREA)
    H, W = g.shape
    for k, (x0, y0, x1, y1) in enumerate(np.asarray(boxes, np.float32) * s):
        h = max(y1 - y0, 4.0)
        a, b = max(int(y0 - 0.15 * h), 0), min(int(y1 + 0.15 * h) + 1, H)
        c, d = max(int(x0 - 0.3 * h), 0), min(int(x1 + 0.3 * h) + 1, W)
        if b - a < 3 or d - c < 3:
            continue
        cut = g[a:b, c:d]
        w = int(np.clip(round(cut.shape[1] * CROP_H / cut.shape[0]), 4, CROP_W))
        out[k, :, CROP_W - w :] = cv2.resize(cut, (w, CROP_H), interpolation=cv2.INTER_AREA)
    return out


def geometry(boxes: np.ndarray, texts: list[str], page_w: float, page_h: float, ink: np.ndarray | None = None) -> np.ndarray:
    """What a line looks like, and how it sits against the lines read just before and after it.

    boxes: [n, 4] x0, y0, x1, y1 in pixels, in reading order. Returns [n, N_GEOM] numbers that do
    not depend on scan size. With `ink` (from ink_features) the result is [n, N_GEOM + N_INK].
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
    out = np.stack(own + against(-1) + against(1) + marks, axis=1).astype(np.float32)
    return out if ink is None else np.concatenate([out, np.asarray(ink, np.float32).reshape(n, N_INK)], axis=1)
