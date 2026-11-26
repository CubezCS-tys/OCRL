"""Draw a layout over its page: one colour per cluster, satellites tied to their body."""

from __future__ import annotations

import cv2
import numpy as np

from .layout import BODY, SAT, Layout


def overlay(gray: np.ndarray, lay: Layout, scale: int = 2) -> np.ndarray:
    n = len(lay.role)
    rng = np.random.default_rng(7)
    hue = rng.integers(0, 180, n)
    hsv = np.zeros((n, 1, 3), np.uint8)
    hsv[:, 0, 0] = hue[lay.host]  # a satellite takes its body's hue
    hsv[:, 0, 1] = 255
    hsv[:, 0, 2] = np.where(lay.role == SAT, 255, 170)
    colors = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0]
    colors[lay.role == 0] = (160, 160, 160)
    img = np.full((*gray.shape, 3), 255, np.uint8)
    ink = lay.labels > 0
    img[ink] = colors[lay.labels[ink]]
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    for ln in lay.lines:
        y = ln.baseline * scale
        cv2.line(img, (0, y), (img.shape[1], y), (210, 210, 210), 1)
        for k, b in enumerate(ln.bodies):
            x0, y0, x1, y1 = lay.boxes[b] * scale
            cv2.putText(img, str(k), (int(x1) - 8, y + int(ln.scale * scale * 0.9)),
                        cv2.FONT_HERSHEY_PLAIN, 0.8, (120, 120, 120), 1, cv2.LINE_AA)
    for b in np.flatnonzero(lay.role == SAT):
        hb = lay.host[b]
        c0 = ((lay.boxes[b, :2] + lay.boxes[b, 2:]) * scale // 2).astype(int)
        hx0, hy0, hx1, hy1 = lay.boxes[hb] * scale
        c1 = (int(np.clip(c0[0], hx0, hx1)), int(np.clip(c0[1], hy0, hy1)))
        cv2.line(img, tuple(int(v) for v in c0), c1, (0, 0, 0), 1, cv2.LINE_AA)
    return img
