"""Tables, put back together from their cells.

The line finder gives every cell of a table as its own line and the structure model says which lines
are table cells. Here those lines become tables again: which cells belong to one table, how the table
divides into rows and columns, which cells span several rows or columns, and which column is read first.

Where the page has ruled lines, they give the grid exactly: a row or column boundary is a rule, and two
neighbouring places of the grid are one cell where the rule between them is missing (a category label
beside a group of rows). Where it has none, the cells' alignment gives it: columns are the vertical
strips of paper that no cell crosses, rows are cells that sit at the same height, and a line that sits
under another cell in the same column, closer than the rows are to each other and in a row that leaves
other columns empty, is that cell's second line.

Nothing here is trained; every rule is the reading a person would make from the page's shape.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from ..text import DIGITS

Box = tuple[int, int, int, int]
_ARABIC = re.compile("[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_LATIN = re.compile("[A-Za-z]")
_DIGIT = re.compile(f"[{DIGITS}]")
NESTED = True  # a ruled frame with another frame inside it is a box round the table or page, not a table (find_tables)
_NOT_IN = {"running_head", "running_foot", "page_number", "table_caption", "figure_caption", "footnote"}


@dataclass
class Cell:
    lines: list  # TextLine, in reading order
    row: int
    col: int  # in reading order: column 0 is read first (the right-hand column of an Arabic table)
    rowspan: int = 1
    colspan: int = 1
    box: Box = (0, 0, 0, 0)  # the cell's place on the page, pixels

    @property
    def text(self) -> str:
        return " ".join(l.text for l in self.lines if l.text)


@dataclass
class Table:
    page: int
    box: Box
    cells: list[Cell]
    n_rows: int
    n_cols: int
    rtl: bool = True
    ruled: bool = False  # its grid came from ruled lines
    header_rows: int = 0
    lines: list = field(default_factory=list)  # every line of the table, cells in reading order
    members: list = field(default_factory=list)  # indices of its lines in the page's line list (find_tables)

    def grid(self) -> list[list]:
        """Rows of the table, one entry per column in reading order: a Cell where a cell starts, None for an
        empty place, "^" for a place covered by a cell from a row above, "<" for one covered by the cell
        before it in the same row."""
        g: list[list] = [[None] * self.n_cols for _ in range(self.n_rows)]
        for c in self.cells:
            for r in range(c.row, c.row + c.rowspan):
                for k in range(c.col, c.col + c.colspan):
                    g[r][k] = "^" if r > c.row else "<" if k > c.col else c
        for row in g:
            for k, v in enumerate(row):
                if isinstance(v, Cell) and not v.lines:
                    row[k] = None if v.rowspan == v.colspan == 1 else v
        return g


# --- the grid ------------------------------------------------------------------------------------


class _Grid:
    """A grid of boundaries (xs left to right, ys top to bottom) and the cells made of its places."""

    def __init__(self, xs: list[float], ys: list[float]):
        self.xs, self.ys = list(xs), list(ys)
        R, C = len(ys) - 1, len(xs) - 1
        self.parent = list(range(R * C))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, r0: int, c0: int, r1: int, c1: int) -> None:
        C = len(self.xs) - 1
        a, b = self.find(r0 * C + c0), self.find(r1 * C + c1)
        if a != b:
            self.parent[b] = a

    def groups(self) -> dict[int, tuple[int, int, int, int]]:
        """Each cell as (row, col, rows, cols); a merge that is not a rectangle is closed into one."""
        R, C = len(self.ys) - 1, len(self.xs) - 1
        while True:
            rect: dict[int, list[int]] = {}
            for r in range(R):
                for c in range(C):
                    g = self.find(r * C + c)
                    b = rect.setdefault(g, [r, c, r, c])
                    b[0], b[1], b[2], b[3] = min(b[0], r), min(b[1], c), max(b[2], r), max(b[3], c)
            changed = False
            for g, (r0, c0, r1, c1) in rect.items():
                for r in range(r0, r1 + 1):
                    for c in range(c0, c1 + 1):
                        if self.find(r * C + c) != self.find(g):
                            self.union(r0, c0, r, c)
                            changed = True
            if not changed:
                return {g: (r0, c0, r1 - r0 + 1, c1 - c0 + 1) for g, (r0, c0, r1, c1) in rect.items()}

    def place(self, x: float, y: float) -> tuple[int, int]:
        r = int(np.clip(np.searchsorted(self.ys, y) - 1, 0, len(self.ys) - 2))
        c = int(np.clip(np.searchsorted(self.xs, x) - 1, 0, len(self.xs) - 2))
        return r, c


def _cluster(values: list[float], weights: list[float], tol: float) -> list[tuple[float, float]]:
    """Positions closer than `tol` are one; returns (weighted position, total weight)."""
    out: list[list[float]] = []
    for v, w in sorted(zip(values, weights)):
        if out and v - out[-1][2] <= tol:
            o = out[-1]
            o[0] += v * w; o[1] += w; o[2] = v
        else:
            out.append([v * w, w, v])
    return [(s / max(w, 1e-6), w) for s, w, _ in out]


def _cover(spans: list[tuple[float, float]], a: float, b: float) -> float:
    """Share of [a, b] covered by the union of spans."""
    if b <= a:
        return 0.0
    got, end = 0.0, a
    for s, e in sorted(spans):
        s, e = max(s, end), min(e, b)
        if e > s:
            got += e - s
            end = e
    return got / (b - a)


# --- rows and columns by alignment ---------------------------------------------------------------


def _columns(lines: list, x0: float, x1: float, lh: float) -> list[float]:
    """Column boundaries from the strips of paper no cell crosses. Cells much wider than the others
    (a heading over several columns) are left out, and a column gap may be crossed by a few cells."""
    W = int(x1 - x0) + 1
    wide = 0.6 * (x1 - x0)
    narrow = [l for l in lines if l.x1 - l.x0 <= wide]
    use = narrow if len(narrow) >= 3 else lines
    count = np.zeros(W, np.int32)
    for l in use:
        count[max(int(l.x0 - x0), 0) : max(int(l.x1 - x0) + 1, 0)] += 1
    rows = len(_bands(use, lh))
    allow = int(0.1 * rows)
    xs = [x0]
    free = count <= allow
    i = 0
    while i < W:
        if free[i]:
            j = i
            while j < W and free[j]:
                j += 1
            if i > 0 and j < W and j - i >= 0.4 * lh:
                xs.append(x0 + (i + j) / 2)
            i = j
        else:
            i += 1
    xs.append(x1)
    return xs


def _bands(lines: list, lh: float) -> list[list]:
    """Lines that share a height: a line joins a band when it overlaps a line of it by 30% of the shorter.
    A "line" far taller than a line of text (a column of dashes the line finder took as one) does not tie
    bands together; it goes with the band its middle is in."""
    hs = [l.y1 - l.y0 for l in lines]
    tall_h = 2.2 * float(np.median(hs)) if hs else 0
    tall = [l for l in lines if l.y1 - l.y0 > tall_h]
    if tall and len(tall) < len(lines):
        bands = _bands([l for l in lines if l.y1 - l.y0 <= tall_h], lh)
        for l in tall:
            cy = (l.y0 + l.y1) / 2
            best = min(bands, key=lambda b: 0 if min(o.y0 for o in b) <= cy <= max(o.y1 for o in b)
                       else min(abs(cy - min(o.y0 for o in b)), abs(cy - max(o.y1 for o in b))))
            best.append(l)
        return bands
    order = sorted(lines, key=lambda l: (l.y0 + l.y1) / 2)
    parent = list(range(len(order)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(order):
        for j in range(i + 1, len(order)):
            b = order[j]
            if b.y0 > a.y1:
                break
            if min(a.y1, b.y1) - max(a.y0, b.y0) >= 0.3 * min(a.y1 - a.y0, b.y1 - b.y0):
                parent[find(j)] = find(i)
    groups: dict[int, list] = {}
    for i, l in enumerate(order):
        groups.setdefault(find(i), []).append(l)
    return sorted(groups.values(), key=lambda g: min(l.y0 for l in g))


def _span(l, xs: list[float]) -> tuple[int, int]:
    """Columns a line covers (left to right): those it overlaps by 40% of their width or of its own."""
    cols = []
    for c in range(len(xs) - 1):
        ov = min(l.x1, xs[c + 1]) - max(l.x0, xs[c])
        if ov > 0 and (ov >= 0.4 * (xs[c + 1] - xs[c]) or ov >= 0.4 * max(l.x1 - l.x0, 1)):
            cols.append(c)
    if not cols:
        c = int(np.clip(np.searchsorted(xs, (l.x0 + l.x1) / 2) - 1, 0, len(xs) - 2))
        return c, c
    return cols[0], cols[-1]


def _rows(lines: list, xs: list[float], lh: float) -> list[list]:
    """Rows of cells: bands of lines, with a band that only carries on cells of the band above joined to it."""
    bands = _bands(lines, lh)
    if len(bands) <= 1:
        return bands
    info = []
    for b in bands:
        cols = set()
        for l in b:
            c0, c1 = _span(l, xs)
            cols |= set(range(c0, c1 + 1))
        core = [l for l in b if l.y1 - l.y0 <= 2.2 * lh] or b  # a tall stray does not stretch the row
        info.append((b, cols, min(l.y0 for l in core), max(l.y1 for l in core)))
    gaps = [info[i + 1][2] - info[i][3] for i in range(len(info) - 1)]
    row_gap = max(float(np.median(gaps)), 0.0)
    rows = [[info[0]]]
    for cur in info[1:]:
        last = rows[-1][-1]
        above = set().union(*(i[1] for i in rows[-1]))
        gap = cur[2] - last[3]
        # a second line of a cell: tight under the row, in columns that row already has, leaving others empty
        # or, tight under a row, the other cells of it: a label of two lines with the row's values set
        # between its lines (label line, values, label line)
        if (len(above) >= 2 and cur[1] < above and gap < 0.6 * lh and gap <= 1.2 * row_gap + 2) or (
                above and cur[1] and not (cur[1] & above) and gap < 0.6 * lh and len(rows[-1]) == 1):
            rows[-1].append(cur)
        else:
            rows.append([cur])
    return [[l for i in r for l in i[0]] for r in rows]


def _row_bounds(rows: list[list], lh: float) -> list[float]:
    """Boundaries between consecutive rows: halfway across the paper between them."""
    core = lambda r: [l for l in r if l.y1 - l.y0 <= 2.2 * lh] or r
    tops = [min(l.y0 for l in core(r)) for r in rows]
    bots = [max(l.y1 for l in core(r)) for r in rows]
    return [(bots[i] + tops[i + 1]) / 2 for i in range(len(rows) - 1)]


# --- tables on a page ----------------------------------------------------------------------------


def frames(grid: list[Box], uprights: list[Box], lh: float) -> list[Box]:
    """Ruled frames: rules across and uprights that touch one another, with at least two of each."""
    segs = [(b, True) for b in grid] + [(b, False) for b in uprights]
    if not segs:
        return []
    tol = max(6, int(0.3 * lh))
    n = len(segs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    bx = np.array([b for b, _ in segs], np.float32)
    for i in range(n):
        a = bx[i]
        hit = (bx[:, 0] <= a[2] + tol) & (bx[:, 2] >= a[0] - tol) & (bx[:, 1] <= a[3] + tol) & (bx[:, 3] >= a[1] - tol)
        for j in np.flatnonzero(hit):
            if j > i:
                parent[find(int(j))] = find(i)
    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    out = []
    for idx in comps.values():
        across = sum(segs[i][1] for i in idx)
        up = len(idx) - across
        b = bx[idx]
        box = (int(b[:, 0].min()), int(b[:, 1].min()), int(b[:, 2].max()), int(b[:, 3].max()))
        if across >= 2 and up >= 2 and box[2] - box[0] > 3 * lh and box[3] - box[1] > 2 * lh:
            out.append(box)
    return out


def short_rules(gray: np.ndarray, grid: list[Box], uprights: list[Box], lh: float, text: list[Box] = ()) -> list[Box]:
    """Pieces of rule inside ruled frames that the cleaner's long-rule search cannot see: it looks for rules
    seven letters long across and nearly two up, so the rule across a narrow column (of row numbers), or the
    sides of a table drawn as one small box per cell, are not among them. Look again inside each frame for
    thin runs about a line's height long, both ways. A run through the middle of a line of `text` is part of
    a letter (a baseline, an alef), not a rule."""
    import cv2

    tb = np.array(text, np.float32).reshape(-1, 4)
    hh, ww = tb[:, 3] - tb[:, 1], tb[:, 2] - tb[:, 0]
    out: list[Box] = []
    k = max(int(0.8 * lh), 12)
    for x0, y0, x1, y1 in frames(grid, uprights, lh):
        ox, oy = max(x0 - 4, 0), max(y0 - 4, 0)
        crop = gray[oy : y1 + 5, ox : x1 + 5]
        _, ink = cv2.threshold(crop, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        for across in (True, False):
            runs = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, k) if across else (k, 1), np.uint8))
            n, _, st, _ = cv2.connectedComponentsWithStats(runs, connectivity=8)
            for x, y, w, h, a in st[1:]:
                if a / max(w if across else h, 1) > max(0.25 * lh, 4):  # not thin for its length
                    continue
                bx0, by0, bx1, by1 = x + ox, y + oy, x + w + ox, y + h + oy
                if across:
                    cy = (by0 + by1) / 2
                    inside = (tb[:, 0] < bx1) & (tb[:, 2] > bx0) & (tb[:, 1] + 0.15 * hh < cy) & (cy < tb[:, 3] - 0.15 * hh)
                else:
                    cx = (bx0 + bx1) / 2
                    inside = (tb[:, 1] < by1) & (tb[:, 3] > by0) & (tb[:, 0] + 0.05 * ww < cx) & (cx < tb[:, 2] - 0.05 * ww)
                if not inside.any():
                    out.append((int(bx0), int(by0), int(bx1), int(by1)))
    return out


def _ruled_grid(page, box: Box, lines: list, lh: float) -> _Grid | None:
    """The grid a frame's rules draw, with places merged where the rule between them is missing."""
    x0, y0, x1, y1 = box
    tol = max(8, 0.35 * lh)
    hs = [b for b in page.grid if b[0] < x1 - tol and b[2] > x0 + tol and y0 - tol <= (b[1] + b[3]) / 2 <= y1 + tol]
    vs = [b for b in page.uprights if b[1] < y1 - tol and b[3] > y0 + tol and x0 - tol <= (b[0] + b[2]) / 2 <= x1 + tol]
    short = [b for b in getattr(page, "short", []) if b[0] < x1 + tol and b[2] > x0 - tol and b[1] < y1 + tol and b[3] > y0 - tol]
    hs += [b for b in short if b[2] - b[0] > b[3] - b[1] and y0 - tol <= (b[1] + b[3]) / 2 <= y1 + tol]
    vs += [b for b in short if b[3] - b[1] > b[2] - b[0] and x0 - tol <= (b[0] + b[2]) / 2 <= x1 + tol]
    ys = [p for p, w in _cluster([(b[1] + b[3]) / 2 for b in hs], [b[2] - b[0] for b in hs], tol) if w > 0.1 * (x1 - x0)]
    # an upright counts as a column boundary when it runs through a fair part of the table: the strokes of a
    # word set sideways in a cell, or a diagonal, are uprights too but short ones
    xs = [p for p, w in _cluster([(b[0] + b[2]) / 2 for b in vs], [b[3] - b[1] for b in vs], tol) if w > max(1.2 * lh, 0.25 * (y1 - y0))]
    ys = sorted(set([y0] + [y for y in ys if y0 + tol < y < y1 - tol] + [y1]))
    xs = sorted(set([x0] + [x for x in xs if x0 + tol < x < x1 - tol] + [x1]))

    def thin_out(ps: list[float], keep: float) -> list[float]:
        """Two rules closer than a line's height (a double rule) are one boundary."""
        out = [ps[0]]
        for p in ps[1:-1]:
            if p - out[-1] >= keep:
                out.append(p)
        if ps[-1] - out[-1] < keep and len(out) > 1:
            out[-1] = ps[-1]
        else:
            out.append(ps[-1])
        return out

    ys, xs = thin_out(ys, 0.8 * lh), thin_out(xs, 0.8 * lh)
    if len(xs) < 2 or len(ys) < 2:
        return None
    # Rows inside a ruled row: a frame with uprights but few rules across holds rows set by alignment.
    extra: list[float] = []
    heights = np.diff(ys)
    for r in range(len(ys) - 1):
        # A table ruled under every row has its rows already; only a row far taller than the others (a body
        # ruled only at its top and foot) is divided by alignment. Lines side by side in an ordinary ruled
        # row are cells of more than one line, often set at different heights.
        if len(heights) >= 4 and heights[r] < 3 * np.median(heights):
            continue
        inside = [l for l in lines if ys[r] <= (l.y0 + l.y1) / 2 < ys[r + 1]]
        if len(inside) < 4:
            continue
        rows = _rows(inside, xs, lh)
        cols_in = [len({_span(l, xs)[0] for l in row}) for row in rows]
        # three rows at least: two lines side by side in a ruled row are more often cells of two lines
        if len(rows) >= 3 and sum(c >= 2 for c in cols_in) >= 3:
            extra += _row_bounds(rows, lh)
    ys = sorted(ys + extra)
    g = _Grid(xs, ys)
    R, C = len(ys) - 1, len(xs) - 1
    hspan = {}
    for r in range(1, R):
        y = ys[r]
        if y in extra:
            hspan[r] = None  # set by alignment: present everywhere, unless a line crosses it
        else:
            hspan[r] = [(b[0], b[2]) for b in hs if abs((b[1] + b[3]) / 2 - y) <= tol]
    vspan = {c: [(b[1], b[3]) for b in vs if abs((b[0] + b[2]) / 2 - xs[c]) <= tol] for c in range(1, C)}
    m = 0.15
    for r in range(R):
        for c in range(C):
            if r + 1 < R:
                spans = hspan[r + 1]
                a, b = xs[c] + m * (xs[c + 1] - xs[c]), xs[c + 1] - m * (xs[c + 1] - xs[c])
                if spans is not None and _cover(spans, a, b) < 0.5:
                    g.union(r, c, r + 1, c)
            if c + 1 < C:
                a, b = ys[r] + m * (ys[r + 1] - ys[r]), ys[r + 1] - m * (ys[r + 1] - ys[r])
                if _cover(vspan[c + 1], a, b) < 0.5:
                    g.union(r, c, r, c + 1)
    for l in lines:  # a line across a boundary set by alignment holds the two places together
        for r in range(1, R):
            if hspan[r] is None and l.y0 + 0.3 * (l.y1 - l.y0) < ys[r] < l.y1 - 0.3 * (l.y1 - l.y0):
                c = g.place((l.x0 + l.x1) / 2, ys[r])[1]
                g.union(r - 1, c, r, c)
    return g


def _aligned_grid(lines: list, box: Box, lh: float, rules: list[Box]) -> _Grid:
    """The grid of an unruled table, from its cells' alignment, with any rules across it as row boundaries."""
    x0, y0, x1, y1 = box
    xs = _columns(lines, x0, x1, lh)
    # rules across most of the table split it into groups of rows (a rule under the header, say)
    tol = 0.3 * lh
    cuts = sorted((b[1] + b[3]) / 2 for b in rules
                  if min(b[2], x1) - max(b[0], x0) > 0.6 * (x1 - x0) and y0 + tol < (b[1] + b[3]) / 2 < y1 - tol)
    groups: list[list] = [[] for _ in range(len(cuts) + 1)]
    for l in lines:
        groups[int(np.searchsorted(cuts, (l.y0 + l.y1) / 2))].append(l)
    every_row = len(cuts) >= 3  # a rule under every row: each group is one row
    ys = [y0]
    for gi, grp in enumerate(groups):
        # the lines above the first rule, over a body below it, are the header: one row, its cells of several lines
        header = gi == 0 and cuts and any(groups[1:])
        if grp and not every_row and not header:
            ys += _row_bounds(_rows(grp, xs, lh), lh)
        if gi < len(cuts):
            ys.append(cuts[gi])
    ys.append(y1)
    ys = sorted(set(ys))
    g = _Grid(xs, ys)
    for l in lines:
        c0, c1 = _span(l, xs)
        r = g.place((l.x0 + l.x1) / 2, (l.y0 + l.y1) / 2)[0]
        for c in range(c0, c1):
            g.union(r, c, r, c + 1)
    return g


def _script_rtl(lines: list, page_lines: list) -> bool:
    # letters only: Arabic-Indic digits sit in the Arabic block, but a table of them with English headings is English
    text = _DIGIT.sub("", " ".join(l.text for l in lines))
    ar, la = len(_ARABIC.findall(text)), len(_LATIN.findall(text))
    if ar + la < 3:  # numbers only: go by the page
        text = _DIGIT.sub("", " ".join(l.text for l in page_lines))
        ar, la = len(_ARABIC.findall(text)), len(_LATIN.findall(text))
    return ar >= la


def _build(page, lines: list, grid: _Grid, box: Box, ruled: bool, lh: float) -> Table:
    xs, ys = grid.xs, grid.ys
    groups = grid.groups()
    held: dict[int, list] = {}
    for l in lines:
        r, c = grid.place((l.x0 + l.x1) / 2, (l.y0 + l.y1) / 2)
        held.setdefault(grid.find(r * (len(xs) - 1) + c), []).append(l)
    R, C = len(ys) - 1, len(xs) - 1
    # Rows and columns that hold nothing and are thinner than a line are slivers between double rules.
    used_r = np.zeros(R, bool)
    used_c = np.zeros(C, bool)
    for g, (r, c, nr, nc) in groups.items():
        if held.get(g):
            used_r[r : r + nr] = True
            used_c[c : c + nc] = True
    keep_r = [r for r in range(R) if used_r[r] or ys[r + 1] - ys[r] >= 1.2 * lh]
    keep_c = [c for c in range(C) if used_c[c] or xs[c + 1] - xs[c] >= 1.2 * lh]
    rmap = {r: i for i, r in enumerate(keep_r)}
    cmap = {c: i for i, c in enumerate(keep_c)}
    nR, nC = len(keep_r), len(keep_c)
    rtl = _script_rtl(lines, page.lines)
    cells = []
    for g, (r, c, nr, nc) in groups.items():
        rs = [rmap[i] for i in range(r, r + nr) if i in rmap]
        cs = [cmap[i] for i in range(c, c + nc) if i in cmap]
        if not rs or not cs:
            continue
        ls = sorted(held.get(g, []), key=lambda l: ((l.y0 + l.y1) / 2, -l.x1 if rtl else l.x0))
        ls = [l for b in _bands(ls, lh) for l in sorted(b, key=lambda l: -l.x1 if rtl else l.x0)] if ls else []
        col = nC - 1 - cs[-1] if rtl else cs[0]
        cells.append(Cell(ls, rs[0], col, len(rs), len(cs), (int(xs[c]), int(ys[r]), int(xs[c + nc]), int(ys[r + nr]))))
    cells.sort(key=lambda c: (c.row, c.col))
    t = Table(page.number, box, cells, nR, nC, rtl, ruled)
    t.lines = [l for c in cells for l in c.lines]
    t.header_rows = _header(t, page, lh)
    return t


def _digit_share(cells: list[Cell]) -> float:
    text = "".join(c.text for c in cells).replace(" ", "")
    return len(_DIGIT.findall(text)) / max(len(text), 1)


def _header(t: Table, page, lh: float) -> int:
    """The first row is a header when it holds words over a body that holds more numbers, or when a rule
    sets it off; a first row of numbers is data."""
    if t.n_rows < 2:
        return 0
    first = [c for c in t.cells if c.row == 0 and c.lines]
    body = [c for c in t.cells if c.row > 0 and c.lines]
    if not first or not body or _digit_share(first) > 0.3:
        return 0
    # a column of numbers below (row numbers, values) whose first cell is a number too: the first row is data
    for k in range(t.n_cols):
        col = [c for c in body if c.col == k]
        top = [c for c in first if c.col == k]
        if len(col) >= 2 and top and sum(_digit_share([c]) > 0.5 for c in col) >= 0.7 * len(col) and _digit_share(top) > 0.5:
            return 0
    if t.ruled or (t.n_cols >= 2 and len(first) >= t.n_cols - (t.n_cols > 2)):
        return 1
    y = max(c.box[3] for c in t.cells if c.row == 0)
    ruled_under = any(abs((b[1] + b[3]) / 2 - y) < 1.5 * lh and min(b[2], t.box[2]) - max(b[0], t.box[0]) > 0.5 * (t.box[2] - t.box[0])
                      for b in page.grid)
    return int(ruled_under or _digit_share(body) >= _digit_share(first) + 0.2)


def innermost(boxes: list[Box]) -> list[Box]:
    """The frames with no other frame wholly inside them. A frame round another frame is a box drawn round
    the table (or round the whole page, as some journals print every page), not the table: built as one, it
    took in every line on the page, running heads and body text with the table, and carried on as one
    table to the document's end."""
    inside = lambda a, b: a != b and b[0] <= a[0] and b[1] <= a[1] and a[2] <= b[2] and a[3] <= b[3]
    return [b for b in boxes if not any(inside(a, b) for a in boxes)]


def find_tables(page, lines: list, kinds: list[str]) -> list[Table]:
    """Tables on one page. `lines` are the page's lines and `kinds` what the structure model called each.

    A ruled frame of at least two rows and two columns whose lines look like cells is a table whatever the
    model said of its lines. Elsewhere, lines the model called table cells are gathered into tables by
    how close they sit, and a line it called something else joins one when it sits inside the table or
    in a row of short lines along its top or bottom edge.
    """
    if not lines:
        return []
    lh_page = float(np.median([l.y1 - l.y0 for l in lines]))
    taken: set[int] = set()
    tables: list[tuple[Table, list[int]]] = []

    def centre_in(l, b, pad=0.0):
        cx, cy = (l.x0 + l.x1) / 2, (l.y0 + l.y1) / 2
        return b[0] - pad <= cx <= b[2] + pad and b[1] - pad <= cy <= b[3] + pad

    # --- ruled frames
    found = frames(page.grid, page.uprights, lh_page)
    found = innermost(found) if NESTED else found
    for box in sorted(found, key=lambda b: (b[1], -b[2])):
        # inside a frame, whatever it was called (a heading row read as a running head, say), but a page
        # number in the page's top or bottom margin: a frame drawn round the whole page, or one reaching
        # down to the foot, holds it too. A number the model called a page number in the body of the page
        # is a cell (a row number, a total); left out, it went missing from the table.
        margin = lambda l: l.y1 < 0.12 * page.height or l.y0 > 0.88 * page.height
        idx = [i for i, l in enumerate(lines) if i not in taken and centre_in(l, box) and not (kinds[i] == "page_number" and margin(l))]
        if len(idx) < 2:
            continue
        ls = [lines[i] for i in idx]
        lh = float(np.median([l.y1 - l.y0 for l in ls]))
        g = _ruled_grid(page, box, ls, lh)
        if g is None or len(g.xs) < 3 or len(g.ys) < 3:
            continue
        t = _build(page, ls, g, box, True, lh)
        cellish = sum(kinds[i] in ("table", "figure") for i in idx) / len(idx)
        filled = [c for c in t.cells if c.lines]
        per_cell = len(ls) / max(len(filled), 1)
        if t.n_rows < 2 or t.n_cols < 2 or not (cellish >= 0.25 or (len(filled) >= 6 and per_cell <= 2.5)):
            continue  # a box around text, or columns of text with a rule between them
        taken |= set(idx)
        tables.append((t, idx))

    # --- cells the structure model found, outside any frame; then text it called part of a figure, when
    #     that text sits in a full grid of rows and columns (a chart's labels do not: they run along its axes)
    for kind in ("table", "figure"):
        seeds = [i for i, k in enumerate(kinds) if k == kind and i not in taken]
        for cl in _gather(lines, kinds, seeds, taken, lh_page):
            cl = [i for i in cl if i not in taken]  # already taken in by a table found before
            if not cl:
                continue
            members = _grow(lines, kinds, set(cl), taken, tables)
            idx = sorted(members)
            ls = [lines[i] for i in idx]
            lh = float(np.median([l.y1 - l.y0 for l in ls]))
            box = _box(ls)
            t = _build(page, ls, _aligned_grid(ls, box, lh, page.grid), box, False, lh)
            if kind == "figure":
                filled = sum(bool(c.lines) for c in t.cells)
                cols_used = [sum(1 for c in t.cells if c.lines and c.col == k) for k in range(t.n_cols)]
                if t.n_rows < 3 or t.n_cols < 2 or filled < 0.6 * t.n_rows * t.n_cols or sum(n >= 3 for n in cols_used) < 2:
                    continue
            taken |= members
            tables.append((t, idx))
    for t, idx in tables:
        t.members = idx
    return [t for t, _ in sorted(tables, key=lambda x: (x[0].box[1], -x[0].box[2]))]


def _gather(lines, kinds, seeds: list[int], taken: set[int], lh: float) -> list[list[int]]:
    """Seed lines into groups, one per table: a line joins the group above it when it is close below it,
    under it, and no caption, heading or full line of text stands between them."""
    clusters: list[list[int]] = []
    for i in sorted(seeds, key=lambda i: lines[i].y0):
        l = lines[i]
        near = []
        for cl in clusters:
            b = _box([lines[j] for j in cl])
            x_near = min(l.x1, b[2]) - max(l.x0, b[0]) > -0.25 * (b[2] - b[0] + l.x1 - l.x0)
            # or another cell of a row already in the group, however wide the column gap
            same_row = any(min(l.y1, lines[j].y1) - max(l.y0, lines[j].y0) > 0.5 * min(l.y1 - l.y0, lines[j].y1 - lines[j].y0) for j in cl)
            if (l.y0 - b[3] <= 2.5 * lh and x_near and not _blocked(lines, kinds, b, l, seeds, taken)) or same_row:
                near.append(cl)
        # a line that joins two groups (the cells of one row, met first as separate columns) makes them one
        merged = [i] + [j for cl in near for j in cl]
        clusters = [cl for cl in clusters if not any(cl is o for o in near)] + [merged]
    return clusters


def _grow(lines, kinds, members: set[int], taken: set[int], tables) -> set[int]:
    """Take in what the model missed: lines inside the table, and rows of two or more short lines along
    its top or bottom edge (a header row the model called something else)."""
    held = {i for _, idx in tables for i in idx}
    for _ in range(3):
        b = _box([lines[j] for j in members])
        lh = float(np.median([lines[j].y1 - lines[j].y0 for j in members]))
        free = [i for i in range(len(lines)) if i not in members and i not in taken and i not in held and kinds[i] not in _NOT_IN]
        grow = set()
        for i in free:
            l = lines[i]
            cx, cy = (l.x0 + l.x1) / 2, (l.y0 + l.y1) / 2
            # a line of running text inside the box (the box of a few stray cells laid over a column of text)
            # is not taken: what is taken is short, as cells are
            if b[0] - 0.25 * lh <= cx <= b[2] + 0.25 * lh and b[1] - 0.25 * lh <= cy <= b[3] + 0.25 * lh and l.x1 - l.x0 < 0.5 * (b[2] - b[0]):
                grow.add(i)
        for above in (True, False):
            near = [i for i in free if lines[i].x0 >= b[0] - lh and lines[i].x1 <= b[2] + lh and lines[i].x1 - lines[i].x0 < 0.6 * (b[2] - b[0])
                    and (0 <= b[1] - lines[i].y1 < 1.5 * lh if above else 0 <= lines[i].y0 - b[3] < 1.5 * lh)]
            for band in _bands([lines[i] for i in near], lh):
                if len(band) >= 2:
                    grow |= {i for i in near if any(lines[i] is o for o in band)}
        if not grow:
            break
        members |= grow
    return members


def _blocked(lines, kinds, b: Box, l, seeds, taken) -> bool:
    """A full line of running text, a caption or a heading between a table and a cell below it: two tables."""
    for j, o in enumerate(lines):
        if j in seeds or j in taken or kinds[j] in ("running_head", "page_number", "noise"):
            continue
        cy = (o.y0 + o.y1) / 2
        if b[3] <= cy <= l.y0 and min(o.x1, b[2]) - max(o.x0, b[0]) > 0:
            if kinds[j] in ("table_caption", "heading") or o.x1 - o.x0 > 0.6 * (b[2] - b[0]):
                return True
    return False


def _box(ls: list) -> Box:
    return min(l.x0 for l in ls), min(l.y0 for l in ls), max(l.x1 for l in ls), max(l.y1 for l in ls)


# --- a table that carries on over the page ------------------------------------------------------


def _similar(a: str, b: str) -> float:
    from rapidfuzz.distance import Levenshtein
    return 1 - Levenshtein.distance(a, b) / max(len(a), len(b), 1)


def continues(a: Table, b: Table) -> bool:
    """`b`, at the top of the next page, carries on `a`: same number of columns, at about the same places."""
    if a.n_cols != b.n_cols or b.page != a.page + 1:
        return False
    wa, wb = a.box[2] - a.box[0], b.box[2] - b.box[0]
    return abs(wa - wb) < 0.15 * max(wa, wb)


def same_direction(a: Table, b: Table) -> None:
    """Give `b`, which carries `a` on over the page, `a`'s direction. Each page's part decides its own from
    its script, and a part without the header (numbers with a few Latin words, say) can come out the other
    way: its columns then landed mirrored under the first part's header."""
    if b.rtl == a.rtl:
        return
    for c in b.cells:
        c.col = b.n_cols - c.col - c.colspan
    b.cells.sort(key=lambda c: (c.row, c.col))
    b.lines = [l for c in b.cells for l in c.lines]
    b.rtl = a.rtl


def repeated_header(a: Table, b: Table) -> bool:
    """The first row of `b` says again what `a`'s header said."""
    if not a.header_rows or not b.header_rows:
        return False
    ha = " ".join(c.text for c in a.cells if c.row == 0)
    hb = " ".join(c.text for c in b.cells if c.row == 0)
    return _similar(ha, hb) > 0.5
