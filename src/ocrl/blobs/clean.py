"""Make a scan safe for the blob map: even out the paper, drop the scanner's black edges, mask photographs.

Everything after this step assumes dark ink on light paper. A tinted page, a photograph or a
black border breaks that assumption, so they are dealt with here, by rule, before any blob is found.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Cleaned:
    gray: np.ndarray  # paper evened out to white, borders and pictures painted out
    pictures: list[tuple[int, int, int, int]] = field(default_factory=list)  # x0, y0, x1, y1
    rules: list[tuple[int, int, int, int]] = field(default_factory=list)  # ruled lines across the page that were painted out
    notes: list[str] = field(default_factory=list)
    # The rules that were not reported above because they belong to a table or a box: the across rules
    # that meet uprights, and the uprights themselves (often one piece per row). Painted out all the same.
    grid: list[tuple[int, int, int, int]] = field(default_factory=list)
    uprights: list[tuple[int, int, int, int]] = field(default_factory=list)
    # Line drawings (maps, charts, diagrams: _drawings), also listed in `pictures`. Their strokes are painted out;
    # the text among them is not.
    drawings: list[tuple[int, int, int, int]] = field(default_factory=list)


def clean_page(gray: np.ndarray, drop_rules: bool = True) -> Cleaned:
    H, W = gray.shape
    notes: list[str] = []
    # 1. The paper. Its local tone is what a large neighbourhood's brightest pixels show; dividing
    #    by it turns grey or unevenly lit paper white and leaves ink dark.
    small = cv2.resize(gray, (max(W // 8, 1), max(H // 8, 1)), interpolation=cv2.INTER_AREA)
    k = max(9, (min(small.shape) // 12) | 1)
    paper = cv2.dilate(small, np.ones((k, k), np.uint8))
    paper = cv2.GaussianBlur(paper, (0, 0), k / 2)
    paper = cv2.resize(paper, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    if float(np.percentile(paper, 20)) < 215:
        notes.append("tinted-paper")
    out = np.clip(gray.astype(np.float32) / np.maximum(paper, 1) * 255, 0, 255).astype(np.uint8)

    # 2. Ink, for finding what is not text.
    _, ink = cv2.threshold(out, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    # A page scanned in black and white with a dither (show-through and grey shading come out as fields of
    # dots a pixel or two across) has tens of thousands of specks: they must not set the letter height (a
    # letter of nine pixels made every Arabic alef an "upright rule"), and they are painted out below. A
    # speck is 8 pixels or fewer at 300 dpi; the dot of a 6-point Arabic letter is about 16.
    tiny = stats[:, 4] <= 8
    tiny[0] = False
    stippled = STIPPLE and int(tiny.sum()) > 3000 and tiny[1:].mean() > 0.5
    hs = stats[1:, 3][stats[1:, 4] >= (12 if stippled else 3)]
    tall = float(np.percentile(hs, 90)) if len(hs) else 20.0
    tall = min(tall, H / 25)  # a page of photographs must not make "a letter" huge

    # 3. The scanner's edge: long dark bars that touch the border of the image.
    erase = np.zeros(n, bool)
    x, y, w, h = (stats[:, i] for i in range(4))
    edge = (x <= 2) | (y <= 2) | (x + w >= W - 2) | (y + h >= H - 2)
    bars = edge & ((w > 0.25 * W) | (h > 0.25 * H))
    bars[0] = False
    if bars.any():
        erase |= bars
        notes.append("scan-border")

    # 4. Photographs. Print is two-toned: white paper, dark ink, and only a thin rim of grey where they
    #    meet. A photograph is mostly in-between greys, or solid dark. So look at each neighbourhood's
    #    share of in-between and of dark pixels; text never gets near these shares, a picture always does.
    b = int(1.5 * tall) | 1
    mid = cv2.blur(((out > 70) & (out < 205)).astype(np.float32), (b, b))
    dark = cv2.blur((out <= 70).astype(np.float32), (b, b))
    photo = ((mid > 0.4) | (dark > 0.65) | (mid + dark > 0.7)).astype(np.uint8)
    k4 = int(2 * tall) | 1
    photo = cv2.morphologyEx(photo, cv2.MORPH_OPEN, np.ones((k4, k4), np.uint8))
    k8 = int(4 * tall) | 1  # a photograph's light areas (a face, a shirt) leave holes: bridge them
    photo = cv2.morphologyEx(photo, cv2.MORPH_CLOSE, np.ones((k8, k8), np.uint8))
    m, mlab, mstats, _ = cv2.connectedComponentsWithStats(photo, connectivity=8)
    pictures, pic_mask = [], np.zeros((H, W), np.uint8)
    pad = int(0.6 * tall)
    # Panels: a flat tone with text set on it is not a photograph. White letters on a dark band (a running
    # head, a title box) are turned dark on white; dark letters on a grey shading (a table's shaded cell)
    # get the shading lifted. Either way the text is read instead of painted out.
    panels = 0
    for i in range(1, m):
        px, py, pw, ph, area = mstats[i]
        if pw > 4 * tall and ph > 4 * tall and _panel(out, mlab[py : py + ph, px : px + pw] == i, (px, py)):
            panels += 1
            mlab[mlab == i] = 0
    # Bands one line high are too thin to be taken for photographs above; look for the dark ones here.
    # A dark band and a grey one can touch (a running head set in two boxes): each tone is looked for alone.
    for lo, hi in ((0, 110), (110, 200)):
        solid = cv2.blur(((out >= lo) & (out < hi)).astype(np.float32), (int(0.6 * tall) | 1, int(0.6 * tall) | 1)) > 0.6
        solid = cv2.morphologyEx(solid.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((int(0.5 * tall) | 1, int(0.5 * tall) | 1), np.uint8))
        s_n, s_lab, s_stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
        for i in range(1, s_n):
            px, py, pw, ph, area = s_stats[i]
            if pw > 4 * tall and 0.8 * tall <= ph <= 4 * tall and area > 0.85 * pw * ph:
                panels += _panel(out, s_lab[py : py + ph, px : px + pw] == i, (px, py), dark_only=True)
    if panels:
        notes.append("panels")
        _, ink = cv2.threshold(out, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    for i in range(1, m):
        px, py, pw, ph, area = mstats[i]
        if not (mlab[py : py + ph, px : px + pw] == i).any():
            continue
        if pw > 4 * tall and ph > 4 * tall:  # a banner one line high with text on it is not a picture
            if SHADED and _shaded(out, (int(px), int(py), int(px + pw), int(py + ph))):
                panels += 1
                continue
            pictures.append((int(px), int(py), int(px + pw), int(py + ph)))
            if area > 0.8 * pw * ph:  # a plain rectangle: take all of it
                pic_mask[max(py - pad, 0) : py + ph + pad, max(px - pad, 0) : px + pw + pad] = 1
            else:  # a cut-out shape with text set around it: take only the shape
                pic_mask |= cv2.dilate((mlab == i).astype(np.uint8), np.ones((2 * pad + 1, 2 * pad + 1), np.uint8))

    if SHADED and panels:  # shadings lifted above: the ink as it now stands
        _, ink = cv2.threshold(out, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 5. Ruled lines: table frames, boxes, the rule above footnotes, underlines. A frame is one connected
    #    blob that would swallow every cell inside it. A ruled line is a straight run of ink far longer
    #    than any letter stroke, and thin. Across: seven letter-heights clears the longest stretched Arabic
    #    join. Up and down: scans break a table's uprights into one piece per row, so the bar is lower
    #    (1.8 letter-heights) and the run must also be at least ten times as long as it is thick, which
    #    an alef or lam never is, even in a headline.
    rules = np.zeros((H, W), np.uint8)
    across_boxes: list[tuple[int, int, int, int]] = []
    upright = np.zeros((H, W), np.uint8)
    upright_boxes: list[tuple[int, int, int, int]] = []
    for kernel, across in ((np.ones((1, int(7 * tall)), np.uint8), True), (np.ones((int(1.8 * tall), 1), np.uint8), False)):
        runs = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
        r, rlab, rstats, _ = cv2.connectedComponentsWithStats(runs, connectivity=8)
        length = rstats[:, 2] if across else rstats[:, 3]
        thick = rstats[:, 4] / np.maximum(length, 1)  # ink per unit length: a long rule on a slightly skewed scan has a fat box
        thin = (thick <= max(0.4 * tall, 4)) & (across | (thick * 10 <= length))
        thin[0] = False
        rules |= thin[rlab].astype(np.uint8)
        if across:
            across_boxes = [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h, _ in rstats[thin]]
        else:
            upright_boxes = [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h, _ in rstats[thin]]
            upright = cv2.dilate(thin[rlab].astype(np.uint8), np.ones((int(tall) | 1, int(tall) | 1), np.uint8))
    # A rule standing on its own separates two parts of the page, and the region finder cuts there. A rule
    # that meets uprights is one side of a table or a box, and says nothing about the page's parts.
    across_rules = [bx for bx in across_boxes if not upright[bx[1] : bx[3] + 1, bx[0] : bx[2] + 1].any()]
    grid = [bx for bx in across_boxes if upright[bx[1] : bx[3] + 1, bx[0] : bx[2] + 1].any()]
    if rules.any():
        rules = cv2.dilate(rules, np.ones((5, 5), np.uint8))
        notes.append("ruled-lines")
    if pictures:
        notes.append("pictures")
    out[erase[labels]] = 255
    out[pic_mask > 0] = 255
    if stippled:
        notes.append("stippled")
        out[tiny[labels]] = 255
    if drop_rules:
        out[rules > 0] = 255
    drawings: list[tuple[int, int, int, int]] = []
    if DRAWINGS and drop_rules:
        drawings, strokes = _drawings(out, tall)
        if drawings:
            notes.append("drawings")
            out[strokes] = 255
            # A drawing's frame (a map's border, a chart's box) is ruled lines like a table's: it must not be
            # put together into a table. Rules inside a drawing go with it.
            within = lambda b: any(b[0] >= d[0] - 2 * tall and b[2] <= d[2] + 2 * tall and b[1] >= d[1] - 2 * tall and b[3] <= d[3] + 2 * tall for d in drawings)
            across_rules = [b for b in across_rules if not within(b)]
            grid = [b for b in grid if not within(b)]
            upright_boxes = [b for b in upright_boxes if not within(b)]
    return Cleaned(out, pictures + drawings, across_rules if drop_rules else [], notes, grid, upright_boxes, drawings)


SHADED = True  # an area of a few flat tones with dark text on them (a table shaded row by row) is lifted, not masked (_shaded)


def _shaded(out: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    """A table whose rows are shaded in turn (grey, white, grey), or a box of two shadings, is found as one
    area of in-between greys, like a photograph; but where a photograph has every tone, this area has two or
    three flat ones (the paper and the shadings) and dark letters on them. If so, every pixel near one of the
    flat tones becomes paper, the letters stay, `out` is changed in place, and it is not a picture."""
    x0, y0, x1, y1 = box
    reg = out[y0:y1, x0:x1]
    hist = np.bincount(reg.ravel(), minlength=256).astype(np.float64)
    smooth = np.convolve(hist, np.ones(9) / 9, mode="same")
    peaks = [v for v in range(120, 256) if smooth[v] >= smooth[max(v - 12, 0) : v + 13].max() and smooth[v] > 0.008 * reg.size]
    if not peaks or len(peaks) > 3:
        return False
    near = np.zeros(256, bool)
    for v in peaks:
        near[max(v - 25, 0) : v + 26] = True
    darkest = min(peaks)
    dark = hist[: max(darkest - 60, 0)].sum() / reg.size
    if hist[near].sum() / reg.size < 0.8 or not 0.01 <= dark <= 0.4:
        return False
    reg[reg >= darkest - 30] = 255
    return True


STIPPLE = True  # a dithered black-and-white scan's specks are painted out and kept out of the letter height
DRAWINGS = True  # maps, charts and diagrams in line work are found and their strokes painted out (_drawings)


def _drawings(out: np.ndarray, tall: float) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    """Line drawings: maps, charts, diagrams. Their ink is mostly a few long thin strokes that sprawl over an
    area many letters wide and tall (a coastline, a border, a curve), which no letter, word or line of text does:
    a piece of ink more than eight letter-heights long, more than two and a half across both ways, and filling
    under a sixth of its box. (A frame of ruled lines is gone by now; a word joined to the line below fills more
    of its box.) The drawing is the area such strokes cover, with what lies among them, and is returned as a
    box; the strokes themselves are painted out so that they are not read as letters. Text within the area (a
    map's place names, a chart's labels) is left: the reader decides which of it to keep (pipeline.read_page)."""
    _, ink = cv2.threshold(out, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    x, y, w, h, a = (st[:, i] for i in range(5))
    fill = a / np.maximum(w * h, 1)
    big = (np.maximum(w, h) > 8 * tall) & (np.minimum(w, h) > 2.5 * tall) & (fill < 0.15)
    big[0] = False
    if not big.any():
        return [], np.zeros(out.shape, bool)
    strokes = big[lab]
    # the area: the strokes' boxes, and boxes that overlap them, joined
    boxes = [[int(x[i]), int(y[i]), int(x[i] + w[i]), int(y[i] + h[i])] for i in np.flatnonzero(big)]
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                p, q = boxes[i], boxes[j]
                if min(p[2], q[2]) + tall > max(p[0], q[0]) and min(p[3], q[3]) + tall > max(p[1], q[1]):
                    boxes[i] = [min(p[0], q[0]), min(p[1], q[1]), max(p[2], q[2]), max(p[3], q[3])]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return [tuple(b) for b in boxes], strokes


def _panel(out: np.ndarray, mask: np.ndarray, at: tuple[int, int], dark_only: bool = False) -> bool:
    """If the area `mask` (a box's worth, its corner at `at`) of the page is a flat tone with text on it,
    make it black text on white paper in `out`, in place, and say so. A photograph has many tones; a
    panel has one, and its letters are a small share of it, all lighter (or all darker) than it."""
    x, y = at
    h, w = mask.shape
    reg = out[y : y + h, x : x + w]
    # Its letters can leave holes in the area found (light letters on a dark band): the panel is all of
    # what its outline holds.
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = cv2.drawContours(np.zeros((h, w), np.uint8), contours, -1, 1, thickness=cv2.FILLED) > 0
    inner = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    vals = reg[inner].astype(np.float32)
    if vals.size < 100:
        return False
    tone = float(np.median(vals))
    if np.mean(np.abs(vals - tone) <= 25) < 0.6:
        return False  # many tones: a photograph
    lighter, darker = np.mean(vals > tone + 50), np.mean(vals < tone - 50)
    # A panel's letters sit inside it, with its tone all round them: its top and bottom edges are plain.
    # (Not so a line of heavy type, whose letters fill the box they make.)
    ys = np.flatnonzero(inner.any(axis=1))
    k = max(len(ys) // 10, 1)
    edge = np.concatenate([reg[ys[:k]][inner[ys[:k]]], reg[ys[-k:]][inner[ys[-k:]]]]).astype(np.float32)
    if dark_only and (edge.size == 0 or np.mean(edge > tone + 50) > 0.02):
        return False
    v = reg.astype(np.float32)
    if tone < 200 and 0.02 <= lighter <= 0.45 and darker < 0.05:  # (a black frame round the band is darker)
        new = 255 - np.clip((v - tone) / max(255 - tone, 1), 0, 1) * 255  # light letters on dark: turn them round
        ring = (cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) & ~inner
        reg[inner] = new[inner].astype(np.uint8)
        reg[ring] = 255
        return True
    if not dark_only and tone >= 150 and 0.02 <= darker <= 0.45 and lighter < 0.1:  # (paper at its rim is lighter)
        new = np.clip(v / max(tone - 10, 1) * 255, 0, 255)  # dark letters on a shading: lift the shading
        reg[inner] = new[inner].astype(np.uint8)
        return True
    return False
