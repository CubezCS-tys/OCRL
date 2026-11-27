"""Page image -> text, with every character traceable to the blobs it was read from.

    blobs -> lines, bodies, satellites      (geometry, no learning)
    cluster -> letters                      (the neural recogniser)
    letters -> checked, spaced, ordered     (rules)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from .blobs.layout import Layout, analyze
from .blobs.regions import Region, find_regions, is_ltr, plan, split_at_uprights
from .recog.crops import cluster_ids, crop
from .recog.decode import greedy, nbest
from .recog.model import load
from .rules.joining import first_violation, forces_word_break
from .rules.signature import satellite_range
from .text import DIGITS, LATIN

_NUMBER = re.compile(f"[{DIGITS}]+(?:[.,][{DIGITS}]+)*")
# A line that holds nothing but the mark opening a list item: a bullet or dash, a number or letter with its
# mark ("١-", "3.", "أ-", "(٢)", "b)").
_LONE_MARK = re.compile(rf"^\s*(?:[-–—−ـ•*·▪■●◦►✓]+|\(?\s?[{DIGITS}]{{1,3}}\s?[-–—ـ.)]|\(\s?(?:[{DIGITS}]{{1,3}}|[أ-ي])\s?\)|[أ-ي]\s?[-–ـ)]|[A-Za-z]\s?[.)])\s*$")


@dataclass
class Cluster:
    body: int  # blob id of the main stroke
    blobs: list[int]  # every blob the text was read from
    text: str
    score: float  # log-probability of the reading
    flags: list[str] = field(default_factory=list)  # doubts: the reading and its ink disagree
    touching: bool = False  # the reading spans more than one unit: bodies have run together here
    first: str = ""  # what the recogniser said before the rules were consulted
    space_after: bool = False


@dataclass
class LineResult:
    text: str
    clusters: list[Cluster]  # right to left, as they sit on the page


@dataclass
class PageResult:
    lines: list[LineResult]
    layout: Layout
    notes: list[str] = field(default_factory=list)  # what the page's ink is doing, e.g. "dots-fusing"

    @property
    def text(self) -> str:
        return "\n".join(ln.text for ln in self.lines)


TOUCHING = "not-one-body"
MISSING = "missing-dots"
EXTRA = "unexplained-ink"


@dataclass
class DocumentPage:
    """A whole page: its regions in reading order, each read on its own."""

    regions: list[tuple[Region, PageResult]]
    # The page as cleaned before reading (ocrl.blobs.clean.Cleaned): its image, and the ruled lines and
    # pictures that were painted out of it. Handed on so that nothing downstream cleans the page again.
    cleaned: object = None

    @property
    def text(self) -> str:
        return "\n".join(res.text for _, res in self.regions if res.text.strip())


def check(text: str, n_satellites: int) -> list[str]:
    """Every way reading `text` off a cluster with this many satellites departs from the rules.

    What a finding *means* is decided by the caller: see `OCR.read`.
    """
    found = []
    if text and first_violation(text) is not None:
        found.append(TOUCHING)  # letters that cannot join: more than one unit in this blob
    lo, hi = satellite_range(text)
    if n_satellites < lo:
        found.append(MISSING)
    elif n_satellites > hi:
        found.append(EXTRA)
    return found


class OCR:
    def __init__(self, checkpoint: str, repair: bool = False, margin: float = 3.0, space_ratio: float = 0.5,
                 fusing_rate: float = 0.02, linefinder: str | None = None, line_reader: str | None = None):
        # `repair` lets the rules swap in a next-best reading when the ink disagrees with the first.
        # It helps a little on clean rendered pages and hurts on the one real scan tested so far
        # (4.9% vs 4.6% character error), so it is off unless asked for. Doubts are flagged either way.
        self.model, self.vocab, _ = load(checkpoint)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.dev).eval()
        # A line reader reads each whole line in one go, in place of the cluster reader. The blobs
        # still say which ink each line is made of.
        self.line_reader = None
        if line_reader:
            from .recog.linereader import LineReader
            self.line_reader = LineReader(line_reader, device=self.dev)
        self.finder = None
        if linefinder:
            from .blobs.linefinder import LineFinder
            self.finder = LineFinder(linefinder)
        self.repair, self.margin, self.space_ratio, self.fusing_rate = repair, margin, space_ratio, fusing_rate

    def read(self, gray: np.ndarray, lay: Layout | None = None, hints: np.ndarray | None = None,
             hint_rank: np.ndarray | None = None) -> PageResult:
        lay = lay or analyze(gray, hints=hints, hint_rank=hint_rank)
        ids = cluster_ids(lay)
        if self.line_reader is not None:
            return self._read_lines(gray, lay, ids)
        items = [(li, b) for li, ln in enumerate(lay.lines) for b in ln.bodies]
        crops = [crop(gray, lay, ids[b], lay.lines[li].baseline, lay.lines[li].scale) for li, b in items]
        logps = self._run(crops)

        # First pass: what the recogniser says, and every place that disagrees with the rules.
        firsts = [greedy(logp, self.vocab) for logp in logps]
        found = [check(t, len(ids[b]) - 1) for t, (_, b) in zip(firsts, items)]

        # A rule only binds where its premise holds, and the page tells us whether it does.
        # "Each dot is its own blob" fails when ink has spread: dots fuse into strokes all
        # over the page and the missing-dots finding stops meaning anything. If it fires
        # on more than a sliver of the page, that is the ink talking, not the recogniser.
        read_n = sum(1 for t in firsts if t)
        missing_n = sum(1 for f in found if MISSING in f)
        notes, doubts = [], {MISSING, EXTRA}
        if missing_n >= 3 and missing_n >= self.fusing_rate * read_n:
            notes.append("dots-fusing")
            doubts.discard(MISSING)

        lines: list[list[Cluster]] = [[] for _ in lay.lines]
        for (li, b), logp, text, f in zip(items, logps, firsts, found):
            n_sat = len(ids[b]) - 1
            flags = [x for x in f if x in doubts]
            first, score = text, 0.0
            if flags and self.repair:
                # The ink and the reading disagree. Ask the recogniser for its other readings
                # and take the best one the ink agrees with, if it is nearly as likely.
                hyps = nbest(logp, vocab=self.vocab)
                score = hyps[0][1] if hyps else 0.0
                for alt, s in hyps:
                    if s < hyps[0][1] - self.margin:
                        break
                    if not any(x in doubts for x in check(alt, n_sat)):
                        text, flags, score = alt, ["repaired"], s
                        break
            # Letters that cannot join in one body are not a doubt about the reading: they
            # say two units have run together, which the reading has correctly spelled out.
            touching = bool(text) and first_violation(text) is not None
            lines[li].append(Cluster(b, ids[b], text, score, flags, touching, first))

        self._space(lay, lines)
        return PageResult([LineResult(_line_text(cl), cl) for cl in lines], lay, notes)

    def _read_lines(self, gray: np.ndarray, lay: Layout, ids: dict[int, list[int]]) -> PageResult:
        """Read each line as one image. A line's image is the box of its ink with a little margin."""
        H, W = gray.shape
        crops, owners = [], []
        for li, ln in enumerate(lay.lines):
            blobs = [b for body in ln.bodies for b in ids[body]]
            if not blobs:
                continue
            bx = lay.boxes[blobs]
            x0, y0, x1, y1 = int(bx[:, 0].min()), int(bx[:, 1].min()), int(bx[:, 2].max()), int(bx[:, 3].max())
            h = y1 - y0
            crops.append(gray[max(int(y0 - 0.18 * h), 0) : min(int(y1 + 0.18 * h) + 1, H), max(int(x0 - 0.3 * h), 0) : min(int(x1 + 0.3 * h) + 1, W)])
            owners.append(li)
        texts = dict(zip(owners, self.line_reader.read(crops))) if crops else {}
        lines = [LineResult(texts.get(li, ""), [Cluster(b, ids[b], "", 0.0) for b in ln.bodies] if li in texts else []) for li, ln in enumerate(lay.lines)]
        return PageResult(lines, lay, ["read-by-line"])

    def read_page(self, gray: np.ndarray) -> DocumentPage:
        """Clean the page, find its columns by whitespace and its lines with the trained line finder
        (if one was given), then read each column in reading order.

        Blob ids in each result are local to its region; add the region's corner to place them on the page.
        """
        from .blobs.clean import clean_page

        cleaned = clean_page(gray)
        gray = cleaned.gray
        lines = self.finder.lines(gray) if self.finder is not None else None
        lines = split_at_uprights(lines, cleaned.uprights, gray)  # one line per table cell, not two cells as one
        if CELL_INK and lines is not None:
            lines = cell_ink(lines, cleaned.uprights, gray)
        ltr = is_ltr(gray)
        ups = np.asarray(cleaned.uprights, np.float32).reshape(-1, 4)
        out = []
        for r, crop, hints, rank in plan(gray, cleaned.rules, lines, pad=16):
            res = self.read(crop, hints=hints, hint_rank=rank)
            if self.line_reader is not None:
                _join_marks(res, ltr, ups - np.array([r.x0 - 16, r.y0 - 16] * 2, np.float32))
            out.append((r, res))
        return DocumentPage(out, cleaned)

    def _run(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        out: list[np.ndarray | None] = [None] * len(crops)
        order = np.argsort([c.shape[2] for c in crops])
        with torch.no_grad():
            for i in range(0, len(order), 256):
                idx = order[i : i + 256]
                W = max(crops[j].shape[2] for j in idx)
                x = torch.zeros(len(idx), 2, crops[idx[0]].shape[1], W)
                for k, j in enumerate(idx):
                    x[k, :, :, : crops[j].shape[2]] = torch.from_numpy(crops[j]).float() / 255
                logp = self.model(x.to(self.dev)).cpu().numpy()
                for k, j in enumerate(idx):
                    out[j] = logp[: crops[j].shape[2] // 2, k]
        return out  # type: ignore[return-value]

    def _space(self, lay: Layout, lines: list[list[Cluster]]) -> None:
        """Word boundaries. Where the facing letters of two separate bodies would have
        joined, there must be a boundary; those certain gaps also tell us how wide a
        word space is on this page, which settles the uncertain ones."""
        pairs = []
        for li, cl in enumerate(lines):
            cl[:] = [c for c in cl if c.text]
            s = lay.lines[li].scale
            for a, b in zip(cl, cl[1:]):
                gap = (lay.boxes[a.body, 0] - lay.boxes[b.body, 2]) / s
                pairs.append((a, b, gap, forces_word_break(a.text, b.text)))
        certain = [g for _, _, g, forced in pairs if forced]
        threshold = self.space_ratio * float(np.median(certain)) if len(certain) >= 5 else 0.22
        for a, b, gap, forced in pairs:
            # Digits of one number are set wider apart than letters; a space between two
            # digits needs a clearly wider gap. (A prior, not a rule of the script.)
            wide = 1.6 if a.text[-1] in DIGITS and b.text[0] in DIGITS else 1.0
            a.space_after = forced or gap > threshold * wide


CELL_INK = True  # ink alone in a ruled table cell that no line holds is a line of its own (cell_ink)


def cell_ink(lines: np.ndarray, uprights: list, gray: np.ndarray) -> np.ndarray:
    """A ruled table's cell holding only a dash or a lone digit: the line finder, trained on lines of words,
    often marks nothing there, and the cell came out empty. Ink that sits between two uprights of a ruled
    table (one to its left and one to its right, both running past it), that no line holds, and that is more
    than a speck (20 pixels of ink, a fifth of a line high or wide) becomes a line of its own, as tight as its
    ink so that it is a full share of its line's height. `uprights` and `gray` as the cleaner left them."""
    ups = np.asarray(uprights, np.float32).reshape(-1, 4)
    if len(ups) < 2 or len(lines) == 0:
        return lines
    lh = float(np.median(lines[:, 3] - lines[:, 1]))
    ink = (gray < 128).astype(np.uint8)
    # one piece per group of marks set close together ("٠٫٠٠", "- ٣"): join across small gaps
    k = max(int(0.3 * lh), 1)
    joined = cv2.dilate(ink, np.ones((max(int(0.15 * lh), 1), k), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    add = []
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        piece = ink[y : y + h, x : x + w]
        rows, cols = np.flatnonzero(piece.any(1)), np.flatnonzero(piece.any(0))
        if not len(rows) or int(piece.sum()) < 20:  # nothing, or a speck of dust
            continue
        x0, y0, x1, y1 = x + cols[0], y + rows[0], x + cols[-1] + 1, y + rows[-1] + 1
        w, h = x1 - x0, y1 - y0
        # a dash, a digit or two; not a whole word (the finder sees those), not a leftover of a ruled line
        if h > 1.5 * lh or w > 4 * lh or max(w, h) < 0.2 * lh or (h < 0.2 * lh and w > 1.2 * lh):
            continue
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if ((lines[:, 0] - 0.2 * lh <= cx) & (cx <= lines[:, 2] + 0.2 * lh) & (lines[:, 1] - 0.2 * lh <= cy) & (cy <= lines[:, 3] + 0.2 * lh)).any():
            continue
        across = (ups[:, 1] < y0 - 0.2 * lh) & (ups[:, 3] > y1 + 0.2 * lh)
        if (across & (ups[:, 2] <= x0)).any() and (across & (ups[:, 0] >= x1)).any():
            add.append((x0, y0, x1, y1))
    if not add:
        return lines
    return np.concatenate([lines, np.asarray(add, np.float32)])


def _join_marks(res: PageResult, ltr: bool, uprights: np.ndarray) -> None:
    """Give each list item back its mark. The line finder often marks an item's opening mark ("١-", "•",
    "أ-") as a line of its own when it hangs out in the margin, a gap away from its text; the mark is then
    read alone and the item's line starts without it, so nothing downstream can tell it is a list item.
    A line read as nothing but such a mark, small, the outermost thing on its row, with a line of text
    level with it a short gap further in (left of it on a right-to-left page) and no ruled upright
    between them, is put back at the head of that line. `uprights` in this region's pixels."""
    lay, lines = res.layout, res.lines
    if len(lines) < 2:
        return
    box = {}
    for li, ln in enumerate(lines):
        if ln.clusters:
            bx = lay.boxes[[b for c in ln.clusters for b in c.blobs]]
            box[li] = (float(bx[:, 0].min()), float(bx[:, 1].min()), float(bx[:, 2].max()), float(bx[:, 3].max()))
    level = lambda a, b: min(a[3], b[3]) - max(a[1], b[1]) >= 0.5 * min(a[3] - a[1], b[3] - b[1])
    gone = {}
    for a, A in box.items():
        if not _LONE_MARK.match(lines[a].text) or a in gone.values():
            continue
        best, best_gap = None, np.inf
        outer = False
        for b, B in box.items():
            if b == a or b in gone or not level(A, B):
                continue
            hb = B[3] - B[1]
            gap = (B[0] - A[2]) if ltr else (A[0] - B[2])  # B further into the line than the mark
            if gap < -0.2 * hb:
                if ((A[0] - B[2]) if ltr else (B[0] - A[2])) > -0.2 * hb:
                    outer = True  # something level on the mark's outer side: a table row, not an item
                continue
            if A[2] - A[0] <= 2.5 * hb and B[2] - B[0] >= 3 * hb and gap <= 3 * hb and gap < best_gap and not _LONE_MARK.match(lines[b].text):
                lo, hi = (A[2], B[0]) if ltr else (B[2], A[0])
                cross = len(uprights) and ((uprights[:, 0] < hi) & (uprights[:, 2] > lo) & (uprights[:, 1] < min(A[3], B[3])) & (uprights[:, 3] > max(A[1], B[1]))).any()
                if not cross:
                    best, best_gap = b, gap
        if best is not None and not outer:
            gone[a] = best
    if not gone:
        return
    for a, b in gone.items():
        lines[b].clusters = lines[a].clusters + lines[b].clusters
        lines[b].text = lines[a].text.strip() + " " + lines[b].text.lstrip()
        lay.line[lay.line == a] = b
    keep = [li for li in range(len(lines)) if li not in gone]
    remap = np.full(len(lines), -1, np.int32)
    remap[keep] = np.arange(len(keep))
    lay.line[:] = np.where(lay.line >= 0, remap[np.maximum(lay.line, 0)], -1)
    lay.lines[:] = [lay.lines[li] for li in keep]
    lines[:] = [lines[li] for li in keep]



def _line_text(clusters: list[Cluster]) -> str:
    """Clusters arrive right to left, as they sit on the page. Arabic is then already in
    reading order; numbers and Latin run the other way and are turned around."""
    words, cur = [], ""
    for c in clusters:
        cur += c.text
        if c.space_after:
            words.append(cur)
            cur = ""
    if cur:
        words.append(cur)
    out: list[str] = []
    run: list[str] = []  # consecutive Latin words: a left-to-right phrase, met from its right end

    def flush() -> None:
        out.extend(w[::-1] for w in reversed(run))
        run.clear()

    for w in words:
        if any(ch in LATIN for ch in w):
            run.append(w)
        else:
            flush()
            out.append(_NUMBER.sub(lambda m: m.group()[::-1], w))
    flush()
    return " ".join(out)
