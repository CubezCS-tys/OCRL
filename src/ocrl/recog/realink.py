"""Real ink, labelled without hand work: Azure's words give the text, the script rules place it on our blobs.

Azure says what each word is and where its box lies. The rules split the word into
its units (sub-words, digits, Latin letters); on clean ink each unit is one body
blob. So when the clusters we find inside a word's box are exactly as many as the
word's units, each cluster takes one unit, right to left. When the counts differ,
ink has touched or broken, or a stray mark got in, and the word is skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..blobs.clean import clean_page
from ..blobs.layout import analyze
from ..blobs.regions import find_regions
from ..rules.joining import unit_spans
from ..text import CHARSET, TATWEEL, runs
from .crops import cluster_ids, crop

_OK = set(CHARSET)
_PERSIAN_YA, _PERSIAN_KAF = "ی", "ک"
_AR = lambda c: "ء" <= c <= "ي"


def fold(word: str) -> str:
    """Azure's spelling brought to the letters this project reads."""
    w = word.replace(TATWEEL, "").replace(_PERSIAN_KAF, "ك")
    out = []
    for i, ch in enumerate(w):
        if ch == _PERSIAN_YA:
            # Azure writes one ya for both: dotted inside a word, dotless at its end.
            inside = i + 1 < len(w) and (_AR(w[i + 1]) or w[i + 1] in (_PERSIAN_YA,))
            ch = "ي" if inside else "ى"
        out.append(ch)
    return "".join(out)


def visual_units(word: str) -> list[str] | None:
    """The word's units in the order they sit on the page, right to left. None if it has a character we do not read."""
    if not word or any(c not in _OK for c in word):
        return None
    rs = runs(word)
    if not rs:
        return None
    out: list[str] = []
    for kind, text in rs:
        us = [text[a:b] for a, b in unit_spans(text) if text[a:b].strip()]
        out.extend(us if kind in ("ar", "p") else us[::-1])  # numbers and Latin run left to right
    return out


@dataclass
class Sample:
    crop: np.ndarray  # uint8 [2, 48, W], as training cuts it
    label: str
    word: int  # index of the Azure word on its page
    rank: int  # which cluster of the word, counted from the right


def align_page(gray: np.ndarray, words: list[dict], page_w: float, page_h: float, min_conf: float = 0.6,
               clean: bool = True) -> tuple[list[Sample], dict]:
    """`words`: Azure's words for the page (content, polygon in the page's unit, confidence)."""
    stats = {k: 0 for k in ("words", "low_confidence", "not_our_characters", "no_clusters", "too_few_clusters",
                             "too_many_clusters", "aligned", "clusters")}
    if clean:
        gray = clean_page(gray).gray
    sx, sy = gray.shape[1] / page_w, gray.shape[0] / page_h
    wb = np.array([[min(w["polygon"][0::2]) * sx, min(w["polygon"][1::2]) * sy, max(w["polygon"][0::2]) * sx, max(w["polygon"][1::2]) * sy]
                   for w in words], np.float32).reshape(-1, 4)
    held: dict[int, list] = {}
    for r in find_regions(gray):
        img = np.pad(gray[r.y0 : r.y1, r.x0 : r.x1], 16, constant_values=255)
        lay = analyze(img)
        ids = cluster_ids(lay)
        ox, oy = r.x0 - 16, r.y0 - 16
        for ln in lay.lines:
            if not ln.bodies or not len(wb):
                continue
            bx = lay.boxes[ln.bodies].astype(np.float32)
            cx, cy = (bx[:, 0] + bx[:, 2]) / 2 + ox, (bx[:, 1] + bx[:, 3]) / 2 + oy
            inside = (cx[:, None] >= wb[None, :, 0]) & (cx[:, None] <= wb[None, :, 2]) & (cy[:, None] >= wb[None, :, 1]) & (cy[:, None] <= wb[None, :, 3])
            for k, b in enumerate(ln.bodies):
                hit = np.flatnonzero(inside[k])
                if len(hit) == 1:  # a centre inside two word boxes belongs to neither with any certainty
                    held.setdefault(int(hit[0]), []).append((float(bx[k, 2]) + ox, img, lay, ids[b], ln.baseline, ln.scale))
    out: list[Sample] = []
    for wi, w in enumerate(words):
        stats["words"] += 1
        if w.get("confidence", 1.0) < min_conf:
            stats["low_confidence"] += 1
            continue
        units = visual_units(fold(w["content"]))
        if units is None:
            stats["not_our_characters"] += 1
            continue
        got = sorted(held.get(wi, ()), key=lambda t: -t[0])  # right to left by right edge
        if not got:
            stats["no_clusters"] += 1
        elif len(got) < len(units):
            stats["too_few_clusters"] += 1  # units touching each other, or swallowed by a neighbour
        elif len(got) > len(units):
            stats["too_many_clusters"] += 1  # broken strokes, or a stray mark standing as a body
        else:
            stats["aligned"] += 1
            stats["clusters"] += len(units)
            for rank, ((_, img, lay, cid, base, scale), unit) in enumerate(zip(got, units)):
                out.append(Sample(crop(img, lay, cid, base, scale), unit, wi, rank))
    return out, stats


def save_shard(path: str | Path, crops: list[np.ndarray], labels: list[str], prov: list[tuple]) -> None:
    """One file: every crop laid end to end along its width, with the offsets to cut them apart again."""
    offsets = np.concatenate([[0], np.cumsum([c.shape[2] for c in crops])]).astype(np.int64)
    torch.save({"strip": torch.from_numpy(np.concatenate(crops, axis=2)), "offsets": torch.from_numpy(offsets),
                "labels": labels, "prov": prov}, str(path))


def load_shard(path: str | Path) -> list[tuple[np.ndarray, str]]:
    """[(crop uint8 [2, 48, W], label)], the same pairs `ocrl.recog.data.collate` takes."""
    d = torch.load(str(path), map_location="cpu")
    strip, off = d["strip"].numpy(), d["offsets"].tolist()
    return [(strip[:, :, a:b], lab) for a, b, lab in zip(off, off[1:], d["labels"])]
