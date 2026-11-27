"""A trained line finder: where the lines of text are, learned from Azure's line boxes.

Which blobs make up one line is a judgement (spacing, columns, vowel marks, tables, damage), and
hand-written rules for it did not survive real documents. A small U-Net marks the *core* of every
line: the middle band of its height, which never touches the line above or below. Each connected
core is one line. Blobs are then attached to the lines by rule, so every character still traces
back to its ink.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn

WIDTH = 1024  # pages are scaled to this width for the model
CORE = (0.30, 0.75)  # the band of a line's height that is marked as its core


def _conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, base: int = 24):
        super().__init__()
        self.config = {"base": base}
        c = [base, base * 2, base * 4, base * 8]
        self.d1, self.d2, self.d3, self.d4 = _conv(1, c[0]), _conv(c[0], c[1]), _conv(c[1], c[2]), _conv(c[2], c[3])
        self.u3, self.u2, self.u1 = _conv(c[3] + c[2], c[2]), _conv(c[2] + c[1], c[1]), _conv(c[1] + c[0], c[0])
        self.out = nn.Conv2d(c[0], 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = lambda t, ref: nn.functional.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        a = self.d1(x); b = self.d2(self.pool(a)); c = self.d3(self.pool(b)); d = self.d4(self.pool(c))
        c = self.u3(torch.cat([up(d, c), c], 1)); b = self.u2(torch.cat([up(c, b), b], 1)); a = self.u1(torch.cat([up(b, a), a], 1))
        return self.out(a)


def scale_page(gray: np.ndarray) -> tuple[np.ndarray, float]:
    f = WIDTH / gray.shape[1]
    small = cv2.resize(gray, (WIDTH, max(int(round(gray.shape[0] * f)), 32)), interpolation=cv2.INTER_AREA)
    return small, f


def core_mask(shape: tuple[int, int], boxes: np.ndarray) -> np.ndarray:
    """boxes [n, 4] x0, y0, x1, y1 in the scaled page's pixels -> uint8 mask of line cores."""
    m = np.zeros(shape, np.uint8)
    for x0, y0, x1, y1 in boxes:
        h = y1 - y0
        a, b = int(round(y0 + CORE[0] * h)), int(round(y0 + CORE[1] * h))
        b = max(b, a + 2)
        m[max(a, 0) : b, max(int(x0) + 1, 0) : max(int(x1) - 1, int(x0) + 2)] = 1
    return m


class LineFinder:
    def __init__(self, path: str, device: str | None = None):
        ck = torch.load(path, map_location="cpu")
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = UNet(ck.get("base", 24)).to(self.dev).eval()
        self.net.load_state_dict(ck["model"])

    @torch.no_grad()
    def prob(self, gray: np.ndarray) -> tuple[np.ndarray, float]:
        """The finder's core probability on the page scaled to WIDTH, and the scale."""
        small, f = scale_page(gray)
        H, W = small.shape
        ph, pw = (-H) % 8, (-W) % 8
        x = torch.from_numpy(np.pad(small, ((0, ph), (0, pw)), constant_values=255)).float()[None, None].to(self.dev) / 255
        return torch.sigmoid(self.net(x))[0, 0, :H, :W].cpu().numpy(), f

    def lines(self, gray: np.ndarray, thresh: float = 0.5, prob: tuple[np.ndarray, float] | None = None,
              below: float | None = None) -> np.ndarray:
        """Line boxes [n, 4] x0, y0, x1, y1 in the page's own pixels, top to bottom. With `below`: only the cores
        that never reach `below` (the lines the finder was unsure of, found at a lower `thresh`)."""
        prob, f = prob if prob is not None else self.prob(gray)
        n, lab, stats, _ = cv2.connectedComponentsWithStats((prob > thresh).astype(np.uint8), connectivity=8)
        peak = np.zeros(n)
        if below is not None:
            np.maximum.at(peak, lab.ravel(), prob.ravel())
        out = []
        for i, (x0, y0, w, h, area) in enumerate(stats[1:], 1):
            if area < 12 or w < 4 or (below is not None and peak[i] > below):
                continue
            full = h / (CORE[1] - CORE[0])  # the whole line's height, from its core's
            top = y0 - CORE[0] * full
            out.append(((x0 - 1) / f, top / f, (x0 + w + 1) / f, (top + full) / f))
        out.sort(key=lambda b: (b[1], -b[2]))
        return np.array(out, np.float32).reshape(-1, 4)
