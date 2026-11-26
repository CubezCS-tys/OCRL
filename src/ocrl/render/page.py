"""Compose a page of text and record, pixel by pixel, which unit of text owns the ink."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..rules.joining import unit_spans
from ..text import NBSP, join_latin, runs
from .shaper import Font


@dataclass
class UnitTruth:
    """One unit of text (a sub-word, a digit, a punctuation mark) as placed on the page."""

    text: str
    line: int
    word: int  # index of its word within the line
    rank: int  # position in the line when bodies are read right to left


@dataclass
class PageTruth:
    image: np.ndarray  # uint8, 255 = paper
    owner: np.ndarray  # int32, 0 = paper, else 1 + index into `units`
    units: list[UnitTruth]
    lines: list[str]  # logical text of each line
    baselines: list[int]
    font: str
    px: float
    meta: dict = field(default_factory=dict)


def compose_page(
    font: Font,
    text: str,
    width: int = 1600,
    margin: int = 60,
    pitch: float = 1.9,
    word_space: float = 1.0,
    max_lines: int | None = None,
) -> PageTruth:
    """Lay `text` out right to left, wrapping at `width`. `pitch` is line spacing in ems."""
    space = font.space * word_space
    words = []
    for w in join_latin(text.split(" ")):
        rs = []
        for kind, t in runs(w):
            if kind == "lat":
                rs.append((kind, (font.latin or font).shape(t, rtl=False, ligatures=False)))
            else:
                rs.append((kind, font.shape(t, rtl=kind != "num")))
        if rs:
            words.append((w.replace(NBSP, " "), rs, sum(r.width for _, r in rs)))

    # Greedy line breaking.
    lines: list[list[int]] = [[]]
    used = 0.0
    avail = width - 2 * margin
    for i, (_, _, wd) in enumerate(words):
        need = wd if not lines[-1] else used + space + wd
        if lines[-1] and need > avail:
            if max_lines and len(lines) >= max_lines:
                break
            lines.append([])
            need = wd
        lines[-1].append(i)
        used = need
    lines = [ln for ln in lines if ln]

    step = int(round(font.px * pitch))
    top = margin + int(round(font.px * 1.2))
    height = top + step * (len(lines) - 1) + int(round(font.px * 1.0)) + margin
    ink = np.zeros((height, width), np.uint8)
    owner = np.zeros((height, width), np.int32)
    units: list[UnitTruth] = []
    line_texts, baselines = [], []

    for li, idxs in enumerate(lines):
        base = top + li * step
        baselines.append(base)
        line_texts.append(" ".join(words[i][0] for i in idxs))
        cursor = float(width - margin)
        rank = 0
        for wi, i in enumerate(idxs):
            _, rs, _ = words[i]
            for kind, run in rs:
                ox = int(round(cursor - run.width))
                cursor -= run.width
                spans = [(a, b) for a, b in unit_spans(run.text) if run.text[a:b].strip()]
                first = len(units)
                order = range(len(spans)) if kind not in ("num", "lat") else range(len(spans) - 1, -1, -1)
                ranks = {k: rank + n for n, k in enumerate(order)}
                rank += len(spans)
                for k, (a, b) in enumerate(spans):
                    units.append(UnitTruth(run.text[a:b], li, wi, ranks[k]))
                if run.coverage.size:
                    _blit_max(ink, run.coverage, ox + run.x, base + run.y)
                k = 0
                for cl in run.clusters:
                    while cl.start >= spans[k][1]:
                        k += 1
                    _stamp(owner, cl.mask, ox + cl.x, base + cl.y, first + k + 1)
            cursor -= space

    return PageTruth(
        255 - ink, owner, units, line_texts, baselines,
        font.name, font.px, {"pitch": pitch, "word_space": word_space},
    )


def _clip(dst_shape, h, w, x, y):
    y0, x0 = max(y, 0), max(x, 0)
    y1, x1 = min(y + h, dst_shape[0]), min(x + w, dst_shape[1])
    return y0, y1, x0, x1


def _blit_max(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    y0, y1, x0, x1 = _clip(dst.shape, *src.shape, x, y)
    if y1 > y0 and x1 > x0:
        view = dst[y0:y1, x0:x1]
        np.maximum(view, src[y0 - y : y1 - y, x0 - x : x1 - x], out=view)


def _stamp(dst: np.ndarray, mask: np.ndarray, x: int, y: int, value: int) -> None:
    y0, y1, x0, x1 = _clip(dst.shape, *mask.shape, x, y)
    if y1 > y0 and x1 > x0:
        dst[y0:y1, x0:x1][mask[y0 - y : y1 - y, x0 - x : x1 - x]] = value
