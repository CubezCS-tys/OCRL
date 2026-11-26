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
    notes: list[str] = field(default_factory=list)


def clean_page(gray: np.ndarray) -> Cleaned:
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

    # 4. Photographs and other solid areas. Close small gaps so a halftone becomes one mass, then
    #    take every mass that is both large and mostly filled. Lines of text stay thin strips.
    kk = max(3, int(round(0.3 * tall)) | 1)
    mass = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((kk, kk), np.uint8))
    mass = cv2.morphologyEx(mass, cv2.MORPH_OPEN, np.ones((int(1.5 * tall) | 1, int(1.5 * tall) | 1), np.uint8))
    m, mlab, mstats, _ = cv2.connectedComponentsWithStats(mass, connectivity=8)
    pictures = []
    for i in range(1, m):
        px, py, pw, ph, area = mstats[i]
        if pw > 3 * tall and ph > 3 * tall and area > 0.55 * pw * ph:
            pad = int(0.4 * tall)
            pictures.append((max(px - pad, 0), max(py - pad, 0), min(px + pw + pad, W), min(py + ph + pad, H)))
    if pictures:
        notes.append("pictures")
    out[erase[labels]] = 255
    for x0, y0, x1, y1 in pictures:
        out[y0:y1, x0:x1] = 255
    return Cleaned(out, pictures, notes)
