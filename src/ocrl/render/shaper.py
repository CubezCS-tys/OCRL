"""Shape and rasterise text with HarfBuzz + FreeType, keeping track of whose ink is whose.

Shaping is deterministic, so rendering our own pages gives exact ground truth:
for every character cluster we know precisely which pixels it put on the page.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import freetype
import numpy as np
import uharfbuzz as hb

_FT_FLAGS = freetype.FT_LOAD_RENDER | freetype.FT_LOAD_NO_HINTING | freetype.FT_LOAD_NO_BITMAP


@dataclass
class Cluster:
    """The ink of one character cluster (a letter, or a ligature such as lam-alef)."""

    start: int  # [start, end) characters of the run text
    end: int
    mask: np.ndarray  # bool, tight crop
    x: int  # left edge relative to the run's origin (pen x = 0)
    y: int  # top edge relative to the baseline (negative = above)


@dataclass
class ShapedRun:
    text: str
    width: float  # advance in pixels
    clusters: list[Cluster]  # in logical order
    coverage: np.ndarray  # uint8 antialiased ink, tight crop
    x: int  # placement of `coverage` relative to run origin / baseline
    y: int


_SUB = 4  # sub-pixel positions per pixel; joins between letters need this to close properly
_IDENTITY = freetype.Matrix(0x10000, 0, 0, 0x10000)


class Font:
    def __init__(self, path: str | Path, px: float, weight: float | None = None, latin: "Font | None" = None):
        """`weight` picks a point on a variable font's weight axis (0 = lightest, 1 = heaviest).
        `latin` is another face to draw Latin runs with, as real documents usually do."""
        self.path = str(path)
        self.name = Path(path).stem
        self.px = px
        self.latin = latin
        self._face = freetype.Face(self.path)
        self._face.set_char_size(0, int(round(px * 64)), 72, 72)
        blob = hb.Blob.from_file_path(self.path)
        self._hb_face = hb.Face(blob)
        self._hb = hb.Font(self._hb_face)
        if weight is not None:
            try:
                axes = self._face.get_variation_info().axes
            except Exception:
                axes = []
            if any(a.tag == "wght" for a in axes):
                coords = [a.minimum + weight * (a.maximum - a.minimum) if a.tag == "wght" else a.default for a in axes]
                self._face.set_var_design_coords(coords)
                self._hb.set_variations({a.tag: c for a, c in zip(axes, coords)})
                self.name += f"@{int(coords[[a.tag for a in axes].index('wght')])}"
        self._scale = px / self._hb_face.upem
        self._glyphs: dict[tuple[int, int], tuple[np.ndarray, int, int]] = {}
        self._runs: dict[tuple[str, bool, bool], ShapedRun] = {}
        self.space = self._advance(" ")

    def _advance(self, text: str) -> float:
        buf = hb.Buffer()
        buf.add_str(text)
        buf.direction, buf.script, buf.language = "rtl", "Arab", "ar"
        hb.shape(self._hb, buf)
        return sum(p.x_advance for p in buf.glyph_positions) * self._scale

    def has_glyphs(self, text: str) -> bool:
        return all(self._hb.get_nominal_glyph(ord(c)) for c in text)

    def _glyph(self, gid: int, sub: int) -> tuple[np.ndarray, int, int]:
        g = self._glyphs.get((gid, sub))
        if g is None:
            self._face.set_transform(_IDENTITY, freetype.Vector(sub * 64 // _SUB, 0))
            self._face.load_glyph(gid, _FT_FLAGS)
            bm = self._face.glyph.bitmap
            if bm.rows and bm.width:
                arr = np.array(bm.buffer, dtype=np.uint8).reshape(bm.rows, bm.pitch)[:, : bm.width]
            else:
                arr = np.zeros((0, 0), np.uint8)
            g = (arr, self._face.glyph.bitmap_left, self._face.glyph.bitmap_top)
            self._glyphs[(gid, sub)] = g
        return g

    def shape(self, text: str, rtl: bool = True, ligatures: bool = True) -> ShapedRun:
        key = (text, rtl, ligatures)
        run = self._runs.get(key)
        if run is None:
            if len(self._runs) > 50_000:
                self._runs.clear()
            run = self._runs[key] = self._shape(text, rtl, ligatures)
        return run

    def _shape(self, text: str, rtl: bool, ligatures: bool) -> ShapedRun:
        buf = hb.Buffer()
        buf.add_str(text)
        buf.direction, buf.script, buf.language = ("rtl" if rtl else "ltr"), "Arab", "ar"
        # Latin ligatures (fi, fl) would hide which ink belongs to which letter.
        hb.shape(self._hb, buf, None if ligatures else {"liga": False, "clig": False, "dlig": False})

        placed: dict[int, list[tuple[np.ndarray, int, int]]] = {}
        pen = 0.0
        boxes = []
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            q = int(round((pen + pos.x_offset * self._scale) * _SUB))
            arr, left, top = self._glyph(info.codepoint, q % _SUB)
            if arr.size:
                gx = q // _SUB + left
                gy = -int(round(pos.y_offset * self._scale)) - top
                placed.setdefault(info.cluster, []).append((arr, gx, gy))
                boxes.append((gx, gy, gx + arr.shape[1], gy + arr.shape[0]))
            pen += pos.x_advance * self._scale

        if not boxes:
            return ShapedRun(text, pen, [], np.zeros((0, 0), np.uint8), 0, 0)
        x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        total = np.zeros((y1 - y0, x1 - x0), np.uint16)

        starts = sorted(placed)
        clusters = []
        for k, start in enumerate(starts):
            end = starts[k + 1] if k + 1 < len(starts) else len(text)
            glyphs = placed[start]
            cx0, cy0 = min(g[1] for g in glyphs), min(g[2] for g in glyphs)
            cx1 = max(g[1] + g[0].shape[1] for g in glyphs)
            cy1 = max(g[2] + g[0].shape[0] for g in glyphs)
            canvas = np.zeros((cy1 - cy0, cx1 - cx0), np.uint16)
            for arr, gx, gy in glyphs:
                canvas[gy - cy0 : gy - cy0 + arr.shape[0], gx - cx0 : gx - cx0 + arr.shape[1]] += arr
            # Ink adds where glyphs overlap, as it does in a real rasteriser.
            total[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0] += canvas
            clusters.append(Cluster(start, end, canvas >= 128, cx0, cy0))
        return ShapedRun(text, pen, clusters, np.minimum(total, 255).astype(np.uint8), x0, y0)
