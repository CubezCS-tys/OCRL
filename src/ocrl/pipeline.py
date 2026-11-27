"""Page image -> text, with every character traceable to the blobs it was read from.

    blobs -> lines, bodies, satellites      (geometry, no learning)
    cluster -> letters                      (the neural recogniser)
    letters -> checked, spaced, ordered     (rules)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import torch

from .blobs.layout import Layout, analyze
from .blobs.regions import Region, find_regions, plan
from .recog.crops import cluster_ids, crop
from .recog.decode import greedy, nbest
from .recog.model import load
from .rules.joining import first_violation, forces_word_break
from .rules.signature import satellite_range
from .text import DIGITS, LATIN

_NUMBER = re.compile(f"[{DIGITS}]+(?:[.,][{DIGITS}]+)*")


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
        out = []
        for r, crop, hints, rank in plan(gray, cleaned.rules, lines, pad=16):
            out.append((r, self.read(crop, hints=hints, hint_rank=rank)))
        return DocumentPage(out)

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
