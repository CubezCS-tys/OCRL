"""Which way up is the page? A landscape table or chart is often scanned on its side; read as it is, every
line comes out as garbage. Told from the ink alone, before anything is read.

Sideways: lines of text are long runs of letters side by side. Smear the ink along the rows and the letters of a
line run together into long flat bars; smear it along the columns instead and, on an upright page, only the
letters themselves are joined. A page on its side does the opposite.

Upside down (once the lines run across): in Arabic and Latin print alike a line's ink is densest along its
baseline, which sits below the middle of the line (the letters' bodies stand on it, and more strokes rise above
it than hang below). Turned over, the densest row is above the middle.
"""

from __future__ import annotations

import cv2
import numpy as np


def _bars(ink: np.ndarray, tall: float, across: bool) -> tuple[float, list]:
    """Smear along rows (across) or columns: the ink in long thin bars, and the bars (x, y, w, h)."""
    k = max(int(1.2 * tall), 3)
    kernel = np.ones((1, k), np.uint8) if across else np.ones((k, 1), np.uint8)
    sm = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)
    n, _, st, _ = cv2.connectedComponentsWithStats(sm, connectivity=8)
    long_, short_ = (st[1:, 2], st[1:, 3]) if across else (st[1:, 3], st[1:, 2])
    keep = (long_ >= 5 * short_) & (long_ >= 6 * tall) & (short_ >= 0.4 * tall) & (short_ <= 3 * tall)
    return float(st[1:, 4][keep].sum()), [tuple(int(v) for v in s[:4]) for s in st[1:][keep]]


def orientation(gray: np.ndarray) -> tuple[int, float]:
    """(degrees, confidence): turn the page by `degrees` (0, 90, 180 or 270, counter-clockwise as np.rot90
    counts) to stand it upright. Confidence is the share of ink in bars that agrees with the call; a page with
    too little text to tell returns (0, 0.0)."""
    f = 1000 / max(gray.shape)
    small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else gray
    _, ink = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    n, _, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n < 60:
        return 0, 0.0
    hw = np.minimum(st[1:, 2], st[1:, 3])[st[1:, 4] >= 4]
    if not len(hw):
        return 0, 0.0
    tall = max(float(np.percentile(hw, 75)), 2.0)  # a letter's size either way round
    a, bars_a = _bars(ink, tall, True)
    u, bars_u = _bars(ink, tall, False)
    if a + u == 0:
        return 0, 0.0
    sideways = u > a
    conf = max(a, u) / (a + u)
    bars = bars_u if sideways else bars_a
    work = np.rot90(ink) if sideways else ink  # np.rot90: a quarter turn counter-clockwise
    if sideways:  # bars measured on the page as it is: turn them with it
        W = ink.shape[1]
        bars = [(y, W - x - w, h, w) for x, y, w, h in bars]
    below = above = 0.0
    for x, y, w, h in bars:
        prof = work[y : y + h, x : x + w].sum(axis=1).astype(np.float32)
        if prof.sum() == 0 or h < 4:
            continue
        peak = int(np.argmax(prof))
        weight = float(prof.sum())
        if peak > (h - 1) / 2:
            below += weight
        elif peak < (h - 1) / 2:
            above += weight
    flipped = above > 1.5 * below
    deg = (90 if sideways else 0) + (180 if flipped else 0)
    return deg % 360, conf
