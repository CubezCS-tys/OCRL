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
    hs = stats[1:, 3][stats[1:, 4] >= 3]
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
    for i in range(1, m):
        px, py, pw, ph, area = mstats[i]
        if pw > 4 * tall and ph > 4 * tall:  # a banner one line high with text on it is not a picture
            pictures.append((int(px), int(py), int(px + pw), int(py + ph)))
            if area > 0.8 * pw * ph:  # a plain rectangle: take all of it
                pic_mask[max(py - pad, 0) : py + ph + pad, max(px - pad, 0) : px + pw + pad] = 1
            else:  # a cut-out shape with text set around it: take only the shape
                pic_mask |= cv2.dilate((mlab == i).astype(np.uint8), np.ones((2 * pad + 1, 2 * pad + 1), np.uint8))

    # 5. Ruled lines: table frames, boxes, the rule above footnotes, underlines. A frame is one connected
    #    blob that would swallow every cell inside it. A ruled line is a straight run of ink far longer
    #    than any letter stroke, and thin. Across: seven letter-heights clears the longest stretched Arabic
    #    join. Up and down: scans break a table's uprights into one piece per row, so the bar is lower
    #    (1.8 letter-heights) and the run must also be at least ten times as long as it is thick, which
    #    an alef or lam never is, even in a headline.
    rules = np.zeros((H, W), np.uint8)
    across_boxes: list[tuple[int, int, int, int]] = []
    upright = np.zeros((H, W), np.uint8)
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
            upright = cv2.dilate(thin[rlab].astype(np.uint8), np.ones((int(tall) | 1, int(tall) | 1), np.uint8))
    # A rule standing on its own separates two parts of the page, and the region finder cuts there. A rule
    # that meets uprights is one side of a table or a box, and says nothing about the page's parts.
    across_rules = [bx for bx in across_boxes if not upright[bx[1] : bx[3] + 1, bx[0] : bx[2] + 1].any()]
    if rules.any():
        rules = cv2.dilate(rules, np.ones((5, 5), np.uint8))
        notes.append("ruled-lines")
    if pictures:
        notes.append("pictures")
    out[erase[labels]] = 255
    out[pic_mask > 0] = 255
    if drop_rules:
        out[rules > 0] = 255
    return Cleaned(out, pictures, across_rules if drop_rules else [], notes)
