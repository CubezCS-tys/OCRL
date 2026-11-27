"""From pixels to structure, with no learning: blobs -> lines -> bodies and satellites.

Every ink pixel ends up in exactly one blob, every blob in exactly one cluster
(a body plus the dots and marks that belong to it), every cluster in one line.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

NOISE, BODY, SAT = 0, 1, 2


@dataclass
class Line:
    baseline: int
    scale: float  # height of tall letters above the baseline, in pixels
    bodies: list[int]  # blob ids, right to left


@dataclass
class Layout:
    labels: np.ndarray  # int32 blob id per pixel, 0 = paper
    boxes: np.ndarray  # [n+1, 4] x0, y0, x1, y1 (exclusive)
    area: np.ndarray
    role: np.ndarray  # NOISE | BODY | SAT
    line: np.ndarray  # line index per blob, -1 if none
    host: np.ndarray  # body a blob belongs to (itself for bodies)
    lines: list[Line]
    stroke: float
    tall: float

    def satellites(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for b in np.flatnonzero(self.role == SAT):
            out.setdefault(int(self.host[b]), []).append(int(b))
        return out


def binarize(gray: np.ndarray) -> np.ndarray:
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return ink


def analyze(gray: np.ndarray, min_area: int = 3, aniso: float = 2.5, hints: np.ndarray | None = None,
            hint_rank: np.ndarray | None = None) -> Layout:
    """`hints`: line boxes [n, 4] (x0, y0, x1, y1, this image's pixels) from the trained line finder.
    With them, which line a letter belongs to is taken from the hint it sits in; without them, lines
    are chained from the letters' vertical overlap. `hint_rank`: each hint's place in reading order; with
    it the lines come out in that order, not simply top to bottom (two columns side by side in one image)."""
    ink = binarize(gray)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    boxes = np.stack(
        [stats[:, 0], stats[:, 1], stats[:, 0] + stats[:, 2], stats[:, 1] + stats[:, 3]], axis=1
    ).astype(np.int32)
    area = stats[:, 4].astype(np.int64)
    h = stats[:, 3].astype(np.float64)
    role = np.full(n, SAT, np.int8)
    role[0] = NOISE
    role[area < min_area] = NOISE
    line = np.full(n, -1, np.int32)
    host = np.arange(n, dtype=np.int32)
    live = role != NOISE
    if not live.any():
        return Layout(labels, boxes, area, role, line, host, [], 0.0, 0.0)

    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    stroke = max(1.0, 4.0 * float(dist[ink > 0].mean()) - 2.0)
    tall = float(np.percentile(h[live], 90))

    # 1. Seeds: blobs tall enough to be letters for certain. Chain them into lines
    #    by vertical overlap.
    if hints is not None and len(hints):
        # A blob is a letter of a hinted line if its centre is inside the hint and it is a fair share of the
        # line's height. The hint decides the line; overlap chaining only handles what no hint covers.
        cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
        # The finder marks a line's core, which can stop short of the line's last word. A letter level with
        # a hint and within a line-height and a half of its end still belongs to it; where two hints could
        # claim a letter (table cells in a row), the nearer end wins.
        reach = 1.5 * (hints[:, 3] - hints[:, 1])
        beside = np.maximum(np.maximum(hints[None, :, 0] - cx[:, None], cx[:, None] - hints[None, :, 2]), 0)
        inside = (beside <= reach[None, :]) & (cy[:, None] >= hints[None, :, 1]) & (cy[:, None] <= hints[None, :, 3])
        mid = (hints[:, 1] + hints[:, 3]) / 2
        off = np.where(inside, beside * 1000 + np.abs(cy[:, None] - mid[None, :]), np.inf)
        which = np.where(np.isfinite(off.min(axis=1)), off.argmin(axis=1), -1)
        hh = np.where(which >= 0, (hints[:, 3] - hints[:, 1])[np.maximum(which, 0)], np.inf)
        hinted = live & (which >= 0) & (h >= 0.3 * hh)
        loose = live & ~hinted & (which < 0) & (h >= 0.45 * tall)
        seeds = np.flatnonzero(hinted | loose)
        _, seed_line = np.unique(which[seeds], return_inverse=True)  # hinted seeds: one line per hint in use
        seed_line = seed_line.astype(np.int32)
        if loose.any():
            is_loose = loose[seeds]
            extra = _chain(boxes[seeds[is_loose]], tall)
            first = int(seed_line[~is_loose].max()) + 1 if (~is_loose).any() else 0
            seed_line[is_loose] = first + extra
            _, seed_line = np.unique(seed_line, return_inverse=True)
            seed_line = seed_line.astype(np.int32)
    else:
        seeds = np.zeros(0, np.int64)
    if len(seeds) == 0:  # no hints, or hints that hold nothing letter-sized: chain by overlap
        seeds = np.flatnonzero(live & (h >= 0.45 * tall))
        seed_line = _chain(boxes[seeds], tall)
    n_lines = int(seed_line.max()) + 1 if len(seeds) else 0
    line[seeds] = seed_line
    role[seeds] = BODY

    # 2. Baseline of each line: the row where its seeds put the most ink.
    H = labels.shape[0]
    ys, xs = np.nonzero(labels)
    li = line[labels[ys, xs]]
    keep = li >= 0
    prof = np.bincount(li[keep] * H + ys[keep], minlength=n_lines * H).reshape(n_lines, H).astype(np.float32)
    k = max(1, int(round(stroke)))
    prof = cv2.blur(prof, (k, 1), borderType=cv2.BORDER_REPLICATE)
    baseline = prof.argmax(axis=1)
    scale = np.empty(n_lines)
    for l in range(n_lines):
        tops = boxes[seeds[seed_line == l], 1]
        scale[l] = max(float(baseline[l] - np.percentile(tops, 10)), stroke)
    scale = np.where(scale < 0.5 * np.median(scale), np.median(scale), scale)

    # Line order: top to bottom, or the reading order handed in with the hints. A line no hint covers
    # goes right after the hinted line nearest above it.
    order = np.argsort(baseline)
    if hint_rank is not None and hints is not None and len(hints) and n_lines:
        hint_of = np.full(n_lines, -1)
        hinted_seed = hinted[seeds] if len(seeds) else np.zeros(0, bool)
        hint_of[seed_line[hinted_seed]] = which[seeds[hinted_seed]]
        key = np.where(hint_of >= 0, hint_rank[np.maximum(hint_of, 0)].astype(np.float64), np.nan)
        known = np.flatnonzero(hint_of >= 0)
        for l in np.flatnonzero(hint_of < 0):
            if len(known):
                above = known[baseline[known] <= baseline[l]]
                key[l] = key[above[np.argmax(baseline[above])]] + 0.5 if len(above) else -1.0
            else:
                key[l] = float(baseline[l])
        order = np.lexsort((baseline, key))
    remap = np.empty(n_lines, np.int32)
    remap[order] = np.arange(n_lines)
    line[seeds] = remap[seed_line]
    baseline, scale = baseline[order], scale[order]

    # 3. Small blobs. One sitting on a baseline with nothing above or below it is a
    #    body of its own (a full stop, a small letter); the rest are satellites.
    small = np.flatnonzero(live & (role != BODY))
    tol = 0.25 * stroke
    col_ink = np.zeros((n_lines, labels.shape[1]), bool)
    left, right = np.full(n_lines, np.inf), np.full(n_lines, -np.inf)  # how far each line's letters reach
    for s in seeds:
        col_ink[line[s], boxes[s, 0] : boxes[s, 2]] = True
        left[line[s]] = min(left[line[s]], boxes[s, 0]); right[line[s]] = max(right[line[s]], boxes[s, 2])

    def which_line(x0: int, x1: int, cy: float) -> tuple[int, float]:
        """The line a small mark belongs to, and how far to the side of that line's letters it lies.
        Two lines can share a height (two columns, cells of a table row): then the nearer one sideways."""
        dy = np.abs(baseline - cy)
        gap = np.maximum(np.maximum(left - x1, x0 - right), 0)
        level = dy <= 1.5 * scale
        l = int(np.argmin(np.where(level, gap * 1000 + dy, np.inf))) if level.any() else int(np.argmin(dy))
        return l, float(gap[l])

    far = 4 * tall  # a mark this far from any letter of its line is a speck in the margin, not part of the line
    for b in small:
        x0, y0, x1, y1 = boxes[b]
        l, gap = which_line(x0, x1, (y0 + y1) / 2)
        if gap > far:
            role[b] = NOISE
            continue
        if y0 <= baseline[l] + tol and y1 - 1 >= baseline[l] - tol:
            if h[b] >= 0.33 * tall or col_ink[l, x0:x1].mean() < 0.3:
                role[b], line[b] = BODY, l

    # 4. Attach each satellite. Dots hang off the baseline stroke: one below the
    #    baseline belongs to the ink straight above it, one above the baseline to
    #    the ink straight below it. Only when its own columns are empty do we fall
    #    back to the nearest body ink, counting sideways distance `aniso` times.
    body_lab = np.where(role[labels] == BODY, labels, 0)
    R = int(round(1.2 * tall))
    sats = np.flatnonzero((role == SAT) & (area >= min_area))
    orphans = []
    for b in sats:
        x0, y0, x1, y1 = boxes[b]
        l, _ = which_line(x0, x1, (y0 + y1) / 2)
        D = int(round(0.9 * scale[l]))
        below = (y0 + y1) / 2 > baseline[l]
        found = 0
        for up in (below, not below):
            found = _look(body_lab, x0, x1, y0, y1, D, up)
            if found:
                break
        if not found:
            wx0, wy0 = max(x0 - R, 0), max(y0 - R, 0)
            win = body_lab[wy0 : y1 + R, wx0 : x1 + R]
            py, px = np.nonzero(win)
            if len(py) == 0:
                orphans.append(b)
                continue
            py, px = py + wy0, px + wx0
            dx = np.maximum(0, np.maximum(x0 - px, px - (x1 - 1))) * aniso
            dy = np.maximum(0, np.maximum(y0 - py, py - (y1 - 1)))
            j = int(np.argmin(dx * dx + dy * dy))
            found = body_lab[py[j], px[j]]
        host[b] = found
        line[b] = line[found]
    for b in orphans:  # nothing nearby to belong to: stands alone
        role[b] = BODY
        line[b] = which_line(boxes[b, 0], boxes[b, 2], (boxes[b, 1] + boxes[b, 3]) / 2)[0] if n_lines else -1

    # 5. Reading order: bodies right to left by their right edge.
    lines = []
    for l in range(n_lines):
        ids = np.flatnonzero((role == BODY) & (line == l))
        ids = ids[np.argsort(-boxes[ids, 2], kind="stable")]
        lines.append(Line(int(baseline[l]), float(scale[l]), [int(i) for i in ids]))
    return Layout(labels, boxes, area, role, line, host, lines, stroke, tall)


def _look(body_lab: np.ndarray, x0: int, x1: int, y0: int, y1: int, reach: int, up: bool) -> int:
    """Nearest body ink straight above (or below) the box, within `reach` rows; 0 if none."""
    if up:
        strip = body_lab[max(y0 - reach, 0) : y0, x0:x1][::-1]
    else:
        strip = body_lab[y1 : y1 + reach, x0:x1]
    rows = np.flatnonzero(strip.any(axis=1))
    if len(rows) == 0:
        return 0
    row = strip[rows[0]]
    return int(row[row > 0][0])


def _chain(boxes: np.ndarray, tall: float, min_overlap: float = 0.5, reach: float = 8.0) -> np.ndarray:
    """Group boxes into lines: connected components of 'overlap vertically and are near'."""
    n = len(boxes)
    if n == 0:
        return np.zeros(0, np.int32)
    x0, y0, x1, y1 = (boxes[:, i].astype(np.float32) for i in range(4))
    ov = np.minimum(y1[:, None], y1[None]) - np.maximum(y0[:, None], y0[None])
    hmin = np.minimum((y1 - y0)[:, None], (y1 - y0)[None])
    gap = np.maximum(x0[:, None], x0[None]) - np.minimum(x1[:, None], x1[None])
    adj = (ov >= min_overlap * hmin) & (gap <= reach * tall)
    lab = np.arange(n)
    while True:
        new = np.where(adj, lab[None, :], n).min(axis=1)
        if (new == lab).all():
            break
        lab = new
    _, out = np.unique(lab, return_inverse=True)
    return out.astype(np.int32)
