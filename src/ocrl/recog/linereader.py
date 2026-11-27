"""The owner's own line reader (inkscript experiment 89), rebuilt from its checkpoint.

The checkpoint holds the weights, the character list and the settings, but not the code. The layer
shapes fix most of the network: four convolution stages with two residual blocks, a (3, 1) convolution
that folds a 48-pixel-high line down to one row, a two-layer bidirectional LSTM, and a CTC head. What
the shapes do not fix (how the width is pooled, ink polarity, where the blank sits, text direction) was
settled by reading lines whose text is known: see scripts/fit_linereader.py.
"""

from __future__ import annotations

import re

import cv2
import numpy as np
import torch
import torch.nn as nn

# The reader sees the line mirrored, so that Arabic runs left to right for it. Whatever the page prints
# left to right (Latin words, numbers) then comes out back to front, and is turned round again here.
#   - a line with no Arabic letters at all is left-to-right as a whole: reverse all of it;
#   - in an Arabic line, a stretch between Arabic letters that holds Latin letters is one left-to-right
#     phrase, from its first letter or digit to its last (brackets and spaces inside go with it);
#   - a stretch with numbers only: each number is turned round on its own, their order stays Arabic.
# Applying it twice gives the text back, so the same function makes the training labels.
_RTL = "\u0590-\u05ff\u0621-\u064a\u066e-\u06d3\u06fa-\u06ff\ufb50-\ufdff\ufe70-\ufefc"
_HAS_RTL = re.compile(f"[{_RTL}]")
_SPAN = re.compile(f"[^{_RTL}]+")
_LAT = "A-Za-z\u00c0-\u024f\u0370-\u03ff\u0400-\u04ff"
_DIG = "0-9\u0660-\u0669\u06f0-\u06f9"
_PHRASE = re.compile(f"[{_LAT}{_DIG}].*[{_LAT}{_DIG}]|[{_LAT}{_DIG}]", re.S)
_HAS_LAT = re.compile(f"[{_LAT}]")
_NUMBER = re.compile(f"[{_DIG}](?:[{_DIG}.,:/%\\-]*[{_DIG}])?")


def _span(m: re.Match) -> str:
    t = m.group()
    if _HAS_LAT.search(t):
        return _PHRASE.sub(lambda x: x.group()[::-1], t, count=1)
    return _NUMBER.sub(lambda x: x.group()[::-1], t)


def to_logical(text: str) -> str:
    if not _HAS_RTL.search(text):
        return text[::-1]
    return _SPAN.sub(_span, text)


class _Res(nn.Module):
    def __init__(self, c: int, late_relu: bool):
        super().__init__()
        self.a = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c))
        self.b = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c))
        self.late_relu = late_relu

    def forward(self, x):
        y = self.b(torch.relu(self.a(x)))
        return torch.relu(x + y) if self.late_relu else x + torch.relu(y)


def _conv(cin, cout, k=(3, 3)):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=(0, 0) if k == (3, 1) else 1, bias=False), nn.BatchNorm2d(cout))


class LineNet(nn.Module):
    def __init__(self, n_out: int, hid: int = 384, feat: int = 384, base: int = 32, wpool=(2, 2, 1, 1), late_relu: bool = True):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        P = lambda w: nn.MaxPool2d((2, w))
        self.cnn = nn.Sequential(
            _conv(1, c[0]), P(wpool[0]), _conv(c[0], c[1]), P(wpool[1]), _conv(c[1], c[2]), _Res(c[2], late_relu), P(wpool[2]),
            _conv(c[2], c[3]), _Res(c[3], late_relu), P(wpool[3]), _conv(c[3], feat, (3, 1)))
        self.rnn = nn.LSTM(feat, hid, num_layers=2, bidirectional=True)
        self.head = nn.Linear(2 * hid, n_out)

    def forward(self, x):  # [B, 1, 48, W]
        f = x
        for m in self.cnn:
            f = m(f)
            if isinstance(m, nn.Sequential):
                f = torch.relu(f)
        f = f.squeeze(2).permute(2, 0, 1)  # [T, B, feat]
        f, _ = self.rnn(f)
        return self.head(f).log_softmax(-1)


class LineReader:
    """Reads one line image (grey, dark ink on light paper) into text."""

    def __init__(self, path: str, wpool=(2, 2, 1, 1), invert: bool = True, blank_first: bool = True, late_relu: bool = True,
                 reverse: bool = False, mirror: bool = True, device: str | None = None):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = ck["meta"]
        self.chars, self.H = m["chars"], m["H"]
        self.invert, self.blank_first, self.reverse, self.mirror = invert, blank_first, reverse, mirror
        base = ck["model"]["cnn.0.0.weight"].shape[0]
        self.net = LineNet(len(self.chars) + 1, m["hid"], m["feat"], base, wpool, late_relu)
        self.net.load_state_dict(ck["model"])
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.dev).eval()

    def prepare(self, gray: np.ndarray) -> np.ndarray:
        w = max(int(round(gray.shape[1] * self.H / gray.shape[0])), 8)
        g = cv2.resize(gray, (w, self.H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        if self.mirror:
            g = g[:, ::-1].copy()
        return 1 - g if self.invert else g

    @torch.no_grad()
    def read(self, lines: list[np.ndarray]) -> list[str]:
        out = [""] * len(lines)
        prep = [self.prepare(g) for g in lines]
        order = np.argsort([p.shape[1] for p in prep])
        for i in range(0, len(order), 64):
            idx = order[i : i + 64]
            W = max(prep[j].shape[1] for j in idx)
            x = torch.full((len(idx), 1, self.H, W), 0.0 if self.invert else 1.0)
            for k, j in enumerate(idx):
                x[k, 0, :, : prep[j].shape[1]] = torch.from_numpy(prep[j])
            ids = self.net(x.to(self.dev)).argmax(-1).cpu().numpy()  # [T, B]
            for k, j in enumerate(idx):
                seq = ids[:, k]
                keep = seq[np.insert(seq[1:] != seq[:-1], 0, True)]
                blank = 0 if self.blank_first else len(self.chars)
                text = "".join(self.chars[c - 1 if self.blank_first else c] for c in keep if c != blank)
                out[j] = text[::-1] if self.reverse else to_logical(text) if self.mirror else text
        return out
