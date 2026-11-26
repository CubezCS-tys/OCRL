"""Cut each cluster out of the page for the recogniser, and say what text its ink is.

A crop has two channels: the cluster's own ink, and everything else nearby. The
second matters: when a neighbour's tail swallows this cluster's dots, the
evidence for them is in the neighbour's ink.
"""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from ..blobs.layout import BODY, Layout
from ..render.page import PageTruth
from ..text import strip_tatweel

HEIGHT = 48
ABOVE, BELOW, SIDE = 1.5, 0.9, 0.4  # crop window around the baseline, in line scales
MAX_WIDTH = 384
_K3 = np.ones((3, 3), np.uint8)


def cluster_ids(lay: Layout) -> dict[int, list[int]]:
    """body id -> all blob ids of its cluster (body first)."""
    out = {int(b): [int(b)] for ln in lay.lines for b in ln.bodies}
    for body, sats in lay.satellites().items():
        if body in out:
            out[body].extend(sats)
    return out


def crop(gray: np.ndarray, lay: Layout, ids: list[int], baseline: int, scale: float, jitter=(1.0, 0.0)) -> np.ndarray:
    """uint8 [2, HEIGHT, W], flipped so that reading order runs left to right."""
    s = scale * jitter[0]
    base = baseline + jitter[1] * scale
    bx = lay.boxes[ids]
    x0 = int(bx[:, 0].min() - SIDE * s)
    x1 = int(bx[:, 2].max() + SIDE * s) + 1
    y0 = int(round(base - ABOVE * s))
    y1 = int(round(base + BELOW * s))
    H, W = gray.shape
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    ink = np.zeros((y1 - y0, x1 - x0), np.uint8)
    mine = np.zeros_like(ink)
    if cy1 > cy0 and cx1 > cx0:
        ink[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0] = 255 - gray[cy0:cy1, cx0:cx1]
        mine[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0] = np.isin(lay.labels[cy0:cy1, cx0:cx1], ids)
    mine = cv2.dilate(mine, _K3)  # keep the soft edge of the strokes
    own = ink * mine
    other = ink * (1 - mine)
    w = int(np.clip(round(ink.shape[1] * HEIGHT / ink.shape[0]), 8, MAX_WIDTH))
    out = np.stack([cv2.resize(c, (w, HEIGHT), interpolation=cv2.INTER_AREA) for c in (own, other)])
    return out[:, :, ::-1].copy()


def cluster_labels(truth: PageTruth, lay: Layout) -> dict[int, str]:
    """Text of each cluster: the units whose body lies in it, in right-to-left reading order.

    The label follows the ink. Two units that touch share one cluster and one
    label; a speck of noise taken for a body gets the empty string.
    """
    m = (lay.labels > 0) & (truth.owner > 0)
    K = len(truth.units) + 1
    keys, counts = np.unique(lay.labels[m].astype(np.int64) * K + truth.owner[m], return_counts=True)
    best: dict[int, tuple[int, int]] = {}  # unit -> (pixels, blob) of its biggest blob
    for key, n in zip(keys, counts):
        blob, unit = int(key // K), int(key % K) - 1
        if lay.role[blob] and n > best.get(unit, (0, 0))[0]:
            best[unit] = (int(n), blob)
    members = defaultdict(list)
    for unit, (_, blob) in best.items():
        members[int(lay.host[blob])].append(unit)
    out = {}
    for ln in lay.lines:
        for b in ln.bodies:
            us = sorted(members.get(b, ()), key=lambda u: truth.units[u].rank)
            out[b] = strip_tatweel("".join(truth.units[u].text for u in us))
    return out
