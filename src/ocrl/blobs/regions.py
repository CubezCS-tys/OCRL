"""Page regions by whitespace, with no learning: where the columns are, and in what order to read them.

A column gutter is a tall strip of paper that no letter crosses. Find the widest
such strip; if there is none because a full-width heading spans it, cut the page
across at its largest horizontal gap and look again in each part. Regions come
back in reading order: top to bottom, and within a row of columns right to left.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .layout import binarize


@dataclass
class Region:
    x0: int
    y0: int
    x1: int
    y1: int


def find_regions(gray: np.ndarray, gutter: float = 1.3, min_side: int = 12, pad: int = 6) -> list[Region]:
    """`gutter`: narrowest column gap, in letter heights. `min_side`: letters needed on each side of a cut."""
    ink = binarize(gray)
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= 1:
        return []
    boxes = np.stack([stats[1:, 0], stats[1:, 1], stats[1:, 0] + stats[1:, 2], stats[1:, 1] + stats[1:, 3]], axis=1)
    h = stats[1:, 3].astype(np.float64)
    live = stats[1:, 4] >= 3
    tall = float(np.percentile(h[live], 90)) if live.any() else 0.0
    # Letters only: specks are too small to block a gutter, rules too flat, and a frame too big.
    seeds = boxes[live & (h >= 0.45 * tall) & (h <= 4 * tall)]
    if len(seeds) == 0:
        return []

    def widest_gap(b: np.ndarray, axis: int, tolerate: int = 0) -> tuple[float, float]:
        """(width, middle) of the widest interval inside the boxes' span that at most `tolerate` boxes cross."""
        lo, hi = int(b[:, axis].min()), int(b[:, axis + 2].max())
        delta = np.zeros(hi - lo + 2, np.int32)
        np.add.at(delta, b[:, axis] - lo, 1)
        np.add.at(delta, b[:, axis + 2] - lo, -1)
        free = np.concatenate([[False], np.cumsum(delta)[: hi - lo] <= tolerate, [False]])
        edges = np.flatnonzero(free[1:] != free[:-1])  # alternating starts and ends of free runs
        if len(edges) == 0:
            return 0.0, 0.0
        starts, ends = edges[0::2], edges[1::2]
        k = int(np.argmax(ends - starts))
        return float(ends[k] - starts[k]), float(lo + (starts[k] + ends[k]) / 2)

    def columns(b: np.ndarray, tolerate: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Split at the widest gutter, if it is a gutter: wide, tall, with real text on both sides."""
        if b[:, 3].max() - b[:, 1].min() < 5 * tall:  # at least about three lines tall
            return None
        width, mid = widest_gap(b, 0, tolerate)
        if width < gutter * tall:
            return None
        clear = (b[:, 2] <= mid) | (b[:, 0] >= mid)  # whatever crosses a gutter is a stray mark: drop it
        left, right = b[clear & (b[:, 2] <= mid)], b[clear & (b[:, 0] >= mid)]
        if len(left) < min_side or len(right) < min_side:
            return None
        return right, left  # right-to-left reading order

    def cut(b: np.ndarray, depth: int = 0) -> list[np.ndarray]:
        cols = columns(b, 0)
        if cols:
            return cut(cols[0], depth + 1) + cut(cols[1], depth + 1)
        # No clean gutter. Something full-width may be hiding one: try below and above the widest
        # horizontal gap, and keep the cut only if it uncovers columns.
        if depth < 10:
            width, mid = widest_gap(b, 1)
            if width >= 0.6 * tall:
                top, bottom = b[b[:, 3] <= mid], b[b[:, 1] >= mid]
                if len(top) and len(bottom):
                    parts = cut(top, depth + 1) + cut(bottom, depth + 1)
                    if len(parts) > 2:
                        return parts
        # Still nothing. Over a long stretch, a stray mark or a word that sticks out may cross a
        # real gutter once or twice; allow that, but only where there is height enough to be sure.
        height = b[:, 3].max() - b[:, 1].min()
        if height >= 15 * tall:
            cols = columns(b, 1 if height < 40 * tall else 2)
            if cols:
                return cut(cols[0], depth + 1) + cut(cols[1], depth + 1)
        return [b]

    H, W = gray.shape
    out = []
    for b in cut(seeds):
        out.append(Region(max(int(b[:, 0].min()) - pad, 0), max(int(b[:, 1].min()) - pad, 0),
                          min(int(b[:, 2].max()) + pad, W), min(int(b[:, 3].max()) + pad, H)))
    return _merge_stacked(out)


def _stacked(p: Region, r: Region) -> bool:
    """`r` sits under `p` in the same column: same horizontal extent, no overlap in height."""
    wp, wr = p.x1 - p.x0, r.x1 - r.x0
    return r.y0 >= p.y1 - 2 and min(p.x1, r.x1) - max(p.x0, r.x0) > 0.8 * min(wp, wr) and abs(wp - wr) < 0.25 * max(wp, wr)


def _merge_stacked(regions: list[Region]) -> list[Region]:
    """Undo cuts the search made across a column rather than between columns.

    Two pieces of one column, one under the other, become one region. And where the page
    was cut into bands that each hold the same two columns (right, left, right, left),
    the columns are restored so that the whole right column is read before the left.
    """
    join = lambda p, r: Region(min(p.x0, r.x0), p.y0, max(p.x1, r.x1), r.y1)
    out = list(regions)
    changed = True
    while changed:
        changed = False
        for i in range(len(out) - 1):
            if _stacked(out[i], out[i + 1]):
                out[i : i + 2] = [join(out[i], out[i + 1])]
                changed = True
                break
            if i + 3 < len(out) and out[i + 1].x1 <= out[i].x0 and _stacked(out[i], out[i + 2]) and _stacked(out[i + 1], out[i + 3]):
                out[i : i + 4] = [join(out[i], out[i + 2]), join(out[i + 1], out[i + 3])]
                changed = True
                break
    return out
