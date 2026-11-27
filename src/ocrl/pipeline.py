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
from .blobs import regions as regions_mod
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
    warnings: list[str] = field(default_factory=list)  # why this line may not have been read properly (_warn)
    notes: list[str] = field(default_factory=list)  # how the line was found or read when a rule, not the models, did it (N_*)


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
    found: np.ndarray | None = None  # the line finder's boxes on the page, after cutting at uprights (for diagnosis)
    warnings: list[str] = field(default_factory=list)  # why this page may not have been read properly (_warn)
    turned: int = 0  # quarter turns (np.rot90's k) the page was given to stand it upright before reading; boxes are on the turned page
    kept_upright: bool = False  # the ink looked sideways but the reader read it surer as it lies (OCR._upright): not turned

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
        crops, owners, turned_up = [], [], set()
        for li, ln in enumerate(lay.lines):
            blobs = [b for body in ln.bodies for b in ids[body]]
            if not blobs:
                continue
            bx = lay.boxes[blobs]
            x0, y0, x1, y1 = int(bx[:, 0].min()), int(bx[:, 1].min()), int(bx[:, 2].max()), int(bx[:, 3].max())
            if DROP_SPECKS and max(x1 - x0, y1 - y0) < 12:
                continue  # a speck (12 pixels at 300 dpi is under a 6-point letter): read, it becomes invented text
            h = y1 - y0
            w = x1 - x0
            # A line set on its side (a running head printed up the margin, a table's turned header) is a tall narrow
            # box: four times as high as wide and three letters long at least. Its margins go by its width; the
            # reader reads lines lying down, so it is given the line turned both ways and the way it is surer of wins.
            # Not a thin stroke (a rule's leftover, a margin bar): at least half a letter wide, and more than one piece.
            if UPRIGHT_LINES and h >= 4 * w and h >= 3 * lay.tall and w >= 0.5 * lay.tall and len(ln.bodies) >= 2:
                c = gray[max(int(y0 - 0.3 * w), 0) : min(int(y1 + 0.3 * w) + 1, H), max(int(x0 - 0.18 * w), 0) : min(int(x1 + 0.18 * w) + 1, W)]
                a, b = np.ascontiguousarray(np.rot90(c, 1)), np.ascontiguousarray(np.rot90(c, 3))
                crops.append(a if self._surety(a) >= self._surety(b) else b)
                turned_up.add(li)
            elif SHORT_INK and h < 0.6 * lay.tall:
                # Ink lower than any letter (a dash, a row of dots, a lone zero): a crop as high as the ink alone
                # blows a dash up to the reader's full height. It is given the room of a line of the page's letters,
                # centred on the ink, as every line it learned from had.
                m = 0.5 * (0.8 * lay.tall - h)
                crops.append(gray[max(int(y0 - m), 0) : min(int(y1 + m) + 1, H), max(int(x0 - 0.25 * lay.tall), 0) : min(int(x1 + 0.25 * lay.tall) + 1, W)])
            else:
                crops.append(gray[max(int(y0 - 0.18 * h), 0) : min(int(y1 + 0.18 * h) + 1, H), max(int(x0 - 0.3 * h), 0) : min(int(x1 + 0.3 * h) + 1, W)])
            owners.append(li)
        texts = dict(zip(owners, self.line_reader.read(crops))) if crops else {}
        lines = [LineResult(texts.get(li, ""), [Cluster(b, ids[b], "", 0.0) for b in ln.bodies] if li in texts else []) for li, ln in enumerate(lay.lines)]
        for li in turned_up:
            lines[li].notes.append(N_TURNED)
        if DASH:
            for li, ln in enumerate(lines):
                n = _dashes(lay, [b for c in ln.clusters for b in c.blobs]) if ln.clusters else 0
                if n and ln.text.strip() != "-" * n:
                    ln.text, ln.notes = "-" * n, ln.notes + [N_DASH]
        return PageResult(lines, lay, ["read-by-line"])

    def read_page(self, gray: np.ndarray) -> DocumentPage:
        """Clean the page, find its columns by whitespace and its lines with the trained line finder
        (if one was given), then read each column in reading order.

        Blob ids in each result are local to its region; add the region's corner to place them on the page.
        """
        from .blobs.clean import clean_page

        turned, kept = 0, False
        if TURN and self.finder is not None and self.line_reader is not None:
            from .blobs.orient import orientation
            if orientation(gray)[0] in (90, 270):
                turned = self._upright(gray)
                kept = not turned
                if turned:
                    gray = np.ascontiguousarray(np.rot90(gray, turned))
        cleaned = clean_page(gray)
        gray = cleaned.gray
        prob = self.finder.prob(gray) if self.finder is not None else None
        lines = self.finder.lines(gray, prob=prob) if self.finder is not None else None
        weak = self.finder.lines(gray, thresh=WEAK, prob=prob, below=0.5) if self.finder is not None and WEAK else None
        lines = split_at_uprights(lines, cleaned.uprights, gray)  # one line per table cell, not two cells as one
        if regions_mod.GAP_SPLIT and lines is not None:
            lines = regions_mod.split_at_gaps(lines, gray)  # nor two cells of an unruled table
        n_found = len(lines) if lines is not None else 0  # lines past this one were added by rule, not the finder
        if weak is not None and len(weak):
            weak = split_at_uprights(weak, cleaned.uprights, gray)
            if regions_mod.GAP_SPLIT:
                weak = regions_mod.split_at_gaps(weak, gray)
            lines = np.concatenate([lines, weak])
        if CELL_INK and lines is not None:
            lines = cell_ink(lines, cleaned.uprights, gray)
        ltr = is_ltr(gray)
        ups = np.asarray(cleaned.uprights, np.float32).reshape(-1, 4)
        out = []
        for r, crop, hints, rank in plan(gray, cleaned.rules, lines, pad=16, n_finder=n_found):
            res = self.read(crop, hints=hints, hint_rank=rank)
            if self.line_reader is not None:
                _join_marks(res, ltr, ups - np.array([r.x0 - 16, r.y0 - 16] * 2, np.float32))
            out.append((r, res))
        if any(getattr(r, "lone", False) for r, _ in out):
            _drop_twice_read(out)
        if lines is not None and len(lines) > n_found:
            _note_rule_lines(out, lines[:n_found], lines[n_found:])
        doc = DocumentPage(out, cleaned, lines, turned=turned, kept_upright=kept)
        if WARN:
            _warn(doc)
        return doc

    @torch.no_grad()
    def _surety(self, c: np.ndarray) -> float:
        """How sure the line reader is of its letters on one line image (mean top probability over non-blank frames)."""
        x = torch.from_numpy(self.line_reader.prepare(c))[None, None].float().to(self.line_reader.dev)
        p = torch.softmax(self.line_reader.net(x).float(), -1)[:, 0].cpu().numpy()
        top = p.argmax(-1)
        letters = top != (0 if self.line_reader.blank_first else p.shape[1] - 1)
        return float(p.max(-1)[letters].mean()) if letters.any() else 0.0

    @torch.no_grad()
    def _upright(self, gray: np.ndarray) -> int:
        """A page lying on its side: which quarter turn (np.rot90's k, 1 or 3; 0: it is upright after all) stands it up. The ink alone tells
        sideways from upright reliably but the two ways round less so (22 of 27 pages), so both are tried: the
        line finder's longest lines on each are read, and the way the reader is surer of its letters wins."""
        c = self._confs(gray, (0, 1, 3))
        k = 3 if c[3] > c[1] else 1
        # The ink can mistake a page of photographs (columns, trees, people standing) for lines on their side.
        # The reader settles it: a page it reads clearly surer as it lies than turned either way is upright
        # (0 returned). On 31 pages the detector called sideways (2026-11-27: 29 Azure found turned, all still
        # turned, the nearest 0.004 below; 2 upright pages of photographs, kept, 0.088 and 0.96 above).
        return 0 if c[0] > c[k] + UPRIGHT_MARGIN else k

    @torch.no_grad()
    def _confs(self, gray: np.ndarray, ks=(0, 1, 3)) -> dict[int, float]:
        """For each quarter turn k: how sure the reader is of its letters on the line finder's longest lines of
        the page turned by k (mean top probability over non-blank frames; 0 when no line is found)."""
        from .blobs.clean import clean_page
        out = {}
        for k in ks:
            g = clean_page(np.ascontiguousarray(np.rot90(gray, k))).gray
            boxes = self.finder.lines(g)
            if not len(boxes):
                out[k] = 0.0
                continue
            boxes = boxes[np.argsort(boxes[:, 0] - boxes[:, 2])[:12]]
            H, W = g.shape
            confs = []
            for x0, y0, x1, y1 in boxes:
                h = y1 - y0
                c = g[max(int(y0 - 0.1 * h), 0) : min(int(y1 + 0.1 * h) + 1, H), max(int(x0), 0) : min(int(x1) + 1, W)]
                if c.size == 0 or c.shape[0] < 4:
                    continue
                x = torch.from_numpy(self.line_reader.prepare(c))[None, None].float().to(self.line_reader.dev)
                p = torch.softmax(self.line_reader.net(x).float(), -1)[:, 0].cpu().numpy()  # [T, C]
                top = p.argmax(-1)
                blank = 0 if self.line_reader.blank_first else p.shape[1] - 1
                letters = top != blank
                if letters.any():
                    confs.append(float(p.max(-1)[letters].mean()))
            out[k] = float(np.mean(confs)) if confs else 0.0
        return out

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


DROP_SPECKS = True  # a line whose ink is a speck is not read (_read_lines)
WARN = True  # every page and line says when it may not have been read properly (_warn)
# The words of the warnings, as the master file carries them (the viewer shows them as they are).
W_NOTHING = "nothing read"  # a line with ink, read as nothing
W_SHORT = "read far too short for its ink"
W_TALL = "taller than its letters: joined with a speck or the line above"
W_UNREAD = "ink not read"  # page: a share of the letter-sized ink is in no line
W_BLANK = "blank page"
W_DOUBT = "many lines doubtful"
W_SIDEWAYS = "page turned on its side: read as it lies, the text is likely garbage"
W_TURNED = "page lay on its side: turned upright ({} degrees counter-clockwise) and read; boxes are on the turned page"
TURN = True  # a page lying on its side is turned upright before it is read (OCR._upright)
UPRIGHT_MARGIN = 0.03  # reader's surety as the page lies must beat the better turn by this much to keep it (OCR._upright)


def _warn(doc: DocumentPage) -> None:
    """No silent errors: say where the reading cannot be trusted, from the ink and the reading alone.

    A line: read as nothing although it holds ink; read far too short for the length of its ink (Arabic and
    Latin print give about three letters for each line-height of length; under half a letter per line-height is
    not a reading of that ink: measured on 25,000 Azure lines, 99.5% hold more than 1.4); or much taller than its
    own letters and than the page's lines (a letter joined with a streak, a speck or the line above, which the
    reader then cannot read). A page: blank; a share of its letter-sized ink in no line at all (lost lines); many
    doubtful lines."""
    hs, boxes = [], []
    for r, res in doc.regions:
        lay = res.layout
        for ln_geo, ln in zip(lay.lines, res.lines):
            if not ln.clusters:
                continue
            bx = lay.boxes[[b for c in ln.clusters for b in c.blobs]]
            box = (float(bx[:, 0].min()), float(bx[:, 1].min()), float(bx[:, 2].max()), float(bx[:, 3].max()))
            boxes.append((r.x0 - 16 + box[0], r.y0 - 16 + box[1], r.x0 - 16 + box[2], r.y0 - 16 + box[3]))
            hs.append(box[3] - box[1])
    med = float(np.median(hs)) if hs else 0.0
    n_doubt = 0
    for r, res in doc.regions:
        lay = res.layout
        for ln_geo, ln in zip(lay.lines, res.lines):
            if not ln.clusters:
                continue
            bx = lay.boxes[[b for c in ln.clusters for b in c.blobs]]
            w, h = float(bx[:, 2].max() - bx[:, 0].min()), float(bx[:, 3].max() - bx[:, 1].min())
            letters = len(ln.text.replace(" ", ""))
            if not ln.text.strip():
                ln.warnings.append(W_NOTHING)
            elif h > 0 and w / h >= 3 and letters < 0.5 * w / h:
                ln.warnings.append(W_SHORT)
            body_h = lay.boxes[ln_geo.bodies, 3] - lay.boxes[ln_geo.bodies, 1] if ln_geo.bodies else np.zeros(0)
            if len(body_h) and h > 2.5 * float(np.median(body_h)) and h > 1.8 * med:
                ln.warnings.append(W_TALL)
            n_doubt += bool(ln.warnings)
    cleaned = doc.cleaned
    if cleaned is None:
        return
    gray = cleaned.gray
    from .blobs.orient import orientation
    if doc.turned:
        doc.warnings.append(W_TURNED.format(90 * doc.turned))
    elif not doc.kept_upright and orientation(gray)[0] in (90, 270):  # on 27 pages Azure found turned, all 27 called sideways; none of 30 upright pages
        doc.warnings.append(W_SIDEWAYS)
    from .blobs.layout import binarize
    n, _, st, _ = cv2.connectedComponentsWithStats(binarize(gray), connectivity=8)
    st = st[1:]
    live = st[:, 4] >= 3
    tall = float(np.percentile(st[live, 3], 90)) if live.any() else 0.0
    letters = live & (st[:, 3] >= 0.6 * tall) & (st[:, 3] <= 4 * tall) & (st[:, 2] <= 15 * tall)
    if letters.sum() < 5 or not boxes:
        if letters.sum() < 5:
            doc.warnings.append(W_BLANK)
        return
    cx, cy = st[letters, 0] + st[letters, 2] / 2, st[letters, 1] + st[letters, 3] / 2
    area = st[letters, 4].astype(np.float64)
    b = np.asarray(boxes, np.float32)
    pad = 0.3 * med
    inside = ((b[None, :, 0] - pad <= cx[:, None]) & (cx[:, None] <= b[None, :, 2] + pad)
              & (b[None, :, 1] - pad <= cy[:, None]) & (cy[:, None] <= b[None, :, 3] + pad)).any(axis=1)
    lost = float(area[~inside].sum() / max(area.sum(), 1))
    if lost >= UNREAD_SHARE:
        doc.warnings.append(f"{W_UNREAD}: {round(100 * lost)}% of the page's letters are in no line")
    if n_doubt >= 3 and n_doubt >= 0.2 * len(boxes):
        doc.warnings.append(f"{W_DOUBT}: {n_doubt} of {len(boxes)}")


UNREAD_SHARE = 0.12  # one page of heavily vowelled text from an unseen journal had 10% (its marks), with no line lost


WEAK = 0.0  # if > 0: cores the finder marked above this but never above 0.5 are lines too, with a note (N_RULE)
UPRIGHT_LINES = True  # a line standing on its side (tall and narrow) is turned to lie down before it is read
N_TURNED = "a line on its side: turned to be read"
DASH = True  # a line whose ink is nothing but one to three short flat strokes is read as dashes (_dashes)
SHORT_INK = True  # a line whose ink is lower than the page's letters is cropped with a letter's height round it (_read_lines)
N_DASH = "read by rule: short flat strokes alone, taken for dashes"
N_RULE = "found by rule: ink the line finder did not mark"


def _dashes(lay: Layout, blobs: list[int]) -> int:
    """How many dashes a line's ink is, if it is nothing but short flat strokes side by side (0 if it is anything
    else). A dash: at least two and a half times as long as it is high, no higher than a third of the page's
    letters, longer than three strokes' width (not a dot) and not long enough to be a rule (the cleaner takes
    those). The reader, trained on lines of words, reads a lone dash poorly; its shape says what it is."""
    big = [b for b in blobs if lay.area[b] >= max(0.5 * lay.stroke ** 2, 3)]
    if not big or len(big) > 3:
        return 0
    for b in big:
        x0, y0, x1, y1 = lay.boxes[b]
        w, h = x1 - x0, y1 - y0
        if not (w >= 2.5 * h and h <= 0.35 * lay.tall and w >= 3 * lay.stroke and w <= 4 * lay.tall):
            return 0
    return len(big)


def _drop_twice_read(out: list) -> None:
    """A region made for one line no other region held (regions._orphan_regions) takes the page round that line;
    where its ink was also read in a neighbouring region, the line is that region's, and the lone region's reading
    of it (the same ink again, or a piece of it) is dropped: a line of a lone region whose box is half covered by a
    line of another region."""
    def boxes(r, res):
        ox, oy = r.x0 - 16, r.y0 - 16
        got = []
        for li, ln in enumerate(res.lines):
            if ln.clusters:
                bx = res.layout.boxes[[b for c in ln.clusters for b in c.blobs]]
                got.append((li, bx[:, 0].min() + ox, bx[:, 1].min() + oy, bx[:, 2].max() + ox, bx[:, 3].max() + oy))
        return got
    others = np.asarray([b[1:] for r, res in out if not getattr(r, "lone", False) for b in boxes(r, res)], np.float32).reshape(-1, 4)
    if not len(others):
        return
    for r, res in out:
        if not getattr(r, "lone", False):
            continue
        for li, x0, y0, x1, y1 in boxes(r, res):
            iw = np.clip(np.minimum(others[:, 2], x1) - np.maximum(others[:, 0], x0), 0, None)
            ih = np.clip(np.minimum(others[:, 3], y1) - np.maximum(others[:, 1], y0), 0, None)
            if (iw * ih).max() >= 0.5 * max((x1 - x0) * (y1 - y0), 1):
                res.lines[li].clusters, res.lines[li].text = [], ""


def _note_rule_lines(out: list, found: np.ndarray, added: np.ndarray) -> None:
    """Every line read from a box a rule added (cell_ink, ...), with no box of the finder holding its centre,
    says so in its notes."""
    for r, res in out:
        ox, oy = r.x0 - 16, r.y0 - 16
        for ln in res.lines:
            if not ln.clusters:
                continue
            bx = res.layout.boxes[[b for c in ln.clusters for b in c.blobs]]
            cx, cy = (bx[:, 0].min() + bx[:, 2].max()) / 2 + ox, (bx[:, 1].min() + bx[:, 3].max()) / 2 + oy
            inb = lambda B: len(B) and ((B[:, 0] <= cx) & (cx <= B[:, 2]) & (B[:, 1] <= cy) & (cy <= B[:, 3])).any()
            if inb(added) and not inb(found):
                ln.notes.append(N_RULE)


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
