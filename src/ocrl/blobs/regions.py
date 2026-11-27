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
    # A region joined from pieces (a column and the footer under it) is not a rectangle: its box can reach
    # over a neighbouring column. The pieces say which ink is really its own.
    parts: list[tuple[int, int, int, int]] | None = None

    def crop(self, gray: np.ndarray, pad: int = 16) -> np.ndarray:
        """This region's own ink on white, with a margin: whatever lies in the box but outside the
        region's pieces belongs to another region and is painted out, so no ink is read twice."""
        out = gray[self.y0 : self.y1, self.x0 : self.x1]
        if self.parts and len(self.parts) > 1:
            keep = np.zeros(out.shape, bool)
            for x0, y0, x1, y1 in self.parts:
                keep[max(y0 - self.y0, 0) : y1 - self.y0, max(x0 - self.x0, 0) : x1 - self.x0] = True
            out = np.where(keep, out, 255).astype(gray.dtype)
        return np.pad(out, pad, constant_values=255)


def partition(seeds: np.ndarray, tall: float, gutter: float = 1.3, min_side: int = 12,
              rules: list[tuple[int, int, int, int]] | None = None, prose: bool = False, ltr: bool = False) -> list[np.ndarray]:
    """Cut a set of boxes [n, 4+] (x0, y0, x1, y1, ...) into groups in reading order: top to bottom, and
    within a row of columns right to left (left to right if `ltr`). `tall` is the unit of length: a
    letter's height when the boxes are letters, a line's height when they are lines.

    `prose` (for line boxes): split only where at least one side is running text, whose lines fill their
    column. The columns of a table are short cells with air around them on both sides of every gap; a
    table is left whole, to be read row by row."""
    def gaps(b: np.ndarray, axis: int, tolerate: int = 0) -> list[tuple[float, float]]:
        """(width, middle) of every interval inside the boxes' span that at most `tolerate` boxes cross, widest first."""
        lo, hi = int(b[:, axis].min()), int(b[:, axis + 2].max())
        delta = np.zeros(hi - lo + 2, np.int32)
        np.add.at(delta, b[:, axis] - lo, 1)
        np.add.at(delta, b[:, axis + 2] - lo, -1)
        free = np.concatenate([[False], np.cumsum(delta)[: hi - lo] <= tolerate, [False]])
        edges = np.flatnonzero(free[1:] != free[:-1])  # alternating starts and ends of free runs
        starts, ends = edges[0::2], edges[1::2]
        order = np.argsort(starts - ends, kind="stable")
        return [(float(ends[k] - starts[k]), float(lo + (starts[k] + ends[k]) / 2)) for k in order]

    def widest_gap(b: np.ndarray, axis: int, tolerate: int = 0) -> tuple[float, float]:
        g = gaps(b, axis, tolerate)
        return g[0] if g else (0.0, 0.0)

    def columns(b: np.ndarray, tolerate: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Split at the widest gutter that is one: wide, tall, with real text on both sides. (The widest
        clear strip of all is often the one between the text and a stray mark in the margin.)"""
        if b[:, 3].max() - b[:, 1].min() < 5 * tall:  # at least about three lines tall
            return None
        # With letters, only the widest strip is tried, as before (measured that way). With whole lines
        # every strip is, widest first: the widest of all is often the one beside a stray mark in the margin.
        for width, mid in gaps(b, 0, tolerate)[: None if prose else 1]:
            if width < gutter * tall:
                return None
            clear = (b[:, 2] <= mid) | (b[:, 0] >= mid)  # whatever crosses a gutter is a stray mark: drop it
            left, right = b[clear & (b[:, 2] <= mid)], b[clear & (b[:, 0] >= mid)]
            if len(left) < min_side or len(right) < min_side:
                continue
            if prose:
                filled = lambda g: np.median(g[:, 2] - g[:, 0]) >= 0.6 * (g[:, 2].max() - g[:, 0].min())
                if not (filled(left) or filled(right)):  # one column of short lines beside a full one is still prose
                    continue
            return (left, right) if ltr else (right, left)
        return None

    def cut(b: np.ndarray, depth: int = 0) -> list[np.ndarray]:
        x0, x1, y0, y1 = b[:, 0].min(), b[:, 2].max(), b[:, 1].min(), b[:, 3].max()
        mid_y = (b[:, 1] + b[:, 3]) / 2
        for rx0, ry0, rx1, ry1 in rules or []:
            ym = (ry0 + ry1) / 2
            if y0 < ym < y1 and min(rx1, x1) - max(rx0, x0) >= 0.6 * (x1 - x0) and depth < 40:
                top, bottom = b[mid_y < ym], b[mid_y >= ym]
                if len(top) and len(bottom):
                    return cut(top, depth + 1) + cut(bottom, depth + 1)
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

    return cut(seeds)


def find_regions(gray: np.ndarray, gutter: float = 1.3, min_side: int = 12, pad: int = 6,
                 rules: list[tuple[int, int, int, int]] | None = None) -> list[Region]:
    """`gutter`: narrowest column gap, in letter heights. `min_side`: letters needed on each side of a cut.
    `rules`: ruled lines across the page (x0, y0, x1, y1), as the cleaner reports them. The printer drew
    them to separate things, so a rule that spans a region cuts it there, before any gutter is looked for."""
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

    H, W = gray.shape
    out = []
    for b in partition(seeds, tall, gutter, min_side, rules):
        box = (max(int(b[:, 0].min()) - pad, 0), max(int(b[:, 1].min()) - pad, 0), min(int(b[:, 2].max()) + pad, W), min(int(b[:, 3].max()) + pad, H))
        out.append(Region(*box, parts=[box]))
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
    join = lambda p, r: Region(min(p.x0, r.x0), p.y0, max(p.x1, r.x1), r.y1, (p.parts or []) + (r.parts or []))
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


def is_ltr(gray: np.ndarray) -> bool:
    """Is this page set in a left-to-right script? Told from the ink alone, before anything is read:
    Arabic letters carry dots and marks, about one small blob for every one or two letter-sized ones;
    Latin print has almost none (the dots of i and j, and punctuation)."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(binarize(gray), connectivity=8)
    h, area = stats[1:, 3], stats[1:, 4]
    live = area >= 3
    if live.sum() < 40:
        return False
    tall = float(np.percentile(h[live], 90))
    letters = (live & (h >= 0.45 * tall)).sum()
    return (live & (h < 0.45 * tall)).sum() < 0.3 * max(letters, 1)


def _rows(g: np.ndarray, h: float, ltr: bool) -> np.ndarray:
    """Order within one group: top to bottom, and lines level with each other in reading direction."""
    mid = (g[:, 1] + g[:, 3]) / 2
    todo, out = list(np.argsort(mid, kind="stable")), []
    while todo:
        top = todo[0]
        row = [i for i in todo if abs(mid[i] - mid[top]) < 0.5 * h]
        out += sorted(row, key=lambda i: g[i, 0] if ltr else -g[i, 2])
        todo = [i for i in todo if i not in row]
    return np.array(out, np.int64)


def line_regions(lines: np.ndarray, shape: tuple[int, int], rules: list[tuple[int, int, int, int]] | None = None,
                 ltr: bool = False) -> list[tuple[Region, np.ndarray, np.ndarray]]:
    """The page's regions from its lines (from the trained line finder): [(region, which lines, their rank)].

    The same gutter search as for letters, run on whole lines. A gutter no line crosses is far plainer
    than one judged from letters; columns of running text are told from the columns of a table, which is
    kept whole and read row by row; and a ruled line across a region cuts it. Regions come in reading
    order, and `rank` gives each of a region's lines its place in it."""
    n = len(lines)
    if n == 0:
        return []
    H, W = shape
    h = float(np.median(lines[:, 3] - lines[:, 1]))
    boxes = np.concatenate([np.round(lines).astype(np.int64), np.arange(n)[:, None]], axis=1)
    groups = partition(boxes, h, gutter=0.8, min_side=4, rules=rules, prose=True, ltr=ltr)
    kept = {int(i) for g in groups for i in g[:, 4]}
    stray = [i for i in range(n) if i not in kept]  # a line that crossed a gutter was left out of both columns
    if stray:
        groups.append(boxes[stray])
    bb = [(int(g[:, 0].min()), int(g[:, 1].min()), int(g[:, 2].max()), int(g[:, 3].max())) for g in groups]
    out = []
    for k, g in enumerate(groups):
        x0, y0, x1, y1 = bb[k]
        # The region takes all the page around its lines, out to halfway to each neighbour: the letters and
        # marks past the ends of its lines, and any line the finder missed nearby, are then inside it.
        ex0, ey0, ex1, ey1 = 0, 0, W, H
        for j, (a0, b0, a1, b1) in enumerate(bb):
            if j == k:
                continue
            if min(y1, b1) - max(y0, b0) > 0:  # side by side
                if a1 <= x0: ex0 = max(ex0, (a1 + x0) // 2)
                if a0 >= x1: ex1 = min(ex1, (x1 + a0) // 2)
            if min(x1, a1) - max(x0, a0) > 0:  # one above the other
                if b1 <= y0: ey0 = max(ey0, (b1 + y0) // 2)
                if b0 >= y1: ey1 = min(ey1, (y1 + b0) // 2)
        box = (ex0, ey0, ex1, ey1)
        order = _rows(g, h, ltr)
        rank = np.empty(len(g), np.int64); rank[order] = np.arange(len(g))
        out.append((Region(*box, parts=[box]), g[:, 4].astype(np.int64), rank))
    return out


def plan(gray: np.ndarray, rules: list[tuple[int, int, int, int]] | None = None, lines: np.ndarray | None = None, pad: int = 16,
         from_lines: bool = False):
    """Everything the page is cut into, in reading order: [(region, its image, hints or None, rank or None)].

    The image is the region's own ink with `pad` pixels of margin; hints are line boxes in that image's
    pixels. Regions are found by whitespace between letters, and each takes the finder's lines whose
    centre it holds. This is the layout the measured results were made with.

    `from_lines` (experimental, 2026-11-27): regions from the lines themselves, tiling the page, with tables
    kept whole and read row by row and the direction taken from the script. On 50 documents it raised a
    rough reading-order measure from 94% to 96%, but it lowered the scores on the hand-keyed development
    documents (a table read as columns, a magazine page), so it is off until that is sorted out."""
    local = lambda r, bx: bx - np.array([r.x0 - pad, r.y0 - pad, r.x0 - pad, r.y0 - pad], np.float32)
    if lines is None or len(lines) == 0:
        return [(r, r.crop(gray, pad), None, None) for r in find_regions(gray, rules=rules)]
    if not from_lines:
        cx, cy = (lines[:, 0] + lines[:, 2]) / 2, (lines[:, 1] + lines[:, 3]) / 2
        return [(r, r.crop(gray, pad), local(r, lines[(cx >= r.x0) & (cx <= r.x1) & (cy >= r.y0) & (cy <= r.y1)]), None)
                for r in find_regions(gray, rules=rules)]
    work, out = gray.copy(), []
    for r, idx, rank in line_regions(lines, gray.shape, rules, is_ltr(gray)):
        out.append((r, r.crop(work, pad), local(r, lines[idx]), rank))
        work[r.y0 : r.y1, r.x0 : r.x1] = 255
    return out
