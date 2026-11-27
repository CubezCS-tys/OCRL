"""A whole document, parsed locally: lines -> blocks -> what each block is -> reading order.

Nothing here is trained. Blocks are cut where the geometry says a paragraph ends
(a wider gap, a first-line indent, a short last line, a change of letter size), and a
block's type comes from where it sits and how it starts: a lone number in the margin is
a page number, a line that repeats at the top of other pages is a running head, a block
that opens with a marker is a list item, and so on. Every rule is a guess a person
would also make from the page's shape; `score_structure.py` says how often it is right.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np
from rapidfuzz.distance import Levenshtein

from ..blobs.layout import binarize
from ..pipeline import OCR
from ..text import DIGITS, strip_marks
from .tables import find_tables

_MARKER = re.compile(rf"^\(?\s?[{DIGITS}]{{1,3}}\s?\)?\s?[-.)ـ]|^[-•*]\s|^\(\s?[{DIGITS}]{{1,3}}\s?\)|^[أ-ي]\s?[-)]\s")
_REF_MARK = re.compile(rf"^\(?\s?[{DIGITS}][{DIGITS}\s]{{0,4}}\)|^[{DIGITS}]{{1,3}}\s?[-.]\s")
_REF_HEAD = re.compile("المراجع|مراجع|المصادر|الهوامش|[Rr]eferences|REFERENCES")
_DIGITS_ONLY = re.compile(rf"^[\s{DIGITS}\-–.()]+$")


@dataclass
class TextLine:
    page: int
    region: int
    x0: int
    y0: int
    x1: int
    y1: int
    baseline: int
    scale: float  # letter height, pixels
    text: str
    clusters: int
    order: int = 0  # place in the region's reading order, as the pipeline gave it (rows of a table run sideways)
    ink: tuple = ()  # stroke weight, ink density and boldness for size (structure.features.ink_features)


@dataclass
class Block:
    lines: list[TextLine]
    type: str = "paragraph"
    order: int | None = None
    id: str = ""
    parent: "Block | None" = None  # the block this one carries on from, over a column or page break
    table: object = None  # for a table: its grid (ocrl.doc.tables.Table)
    caption: "Block | None" = None  # for a table: its caption block

    @property
    def text(self) -> str:
        return " ".join(ln.text for ln in self.lines if ln.text)

    @property
    def page(self) -> int:
        return self.lines[0].page

    def box(self, lines=None) -> tuple[int, int, int, int]:
        ls = lines or self.lines
        return min(l.x0 for l in ls), min(l.y0 for l in ls), max(l.x1 for l in ls), max(l.y1 for l in ls)


@dataclass
class PageInfo:
    number: int
    width: int
    height: int
    lines: list[TextLine]
    columns: list[tuple[int, int, int, int]]  # region boxes, in reading order
    rules: list[tuple[int, int, int, int]]  # long horizontal strokes
    frames: list[tuple[int, int, int, int]]  # ruled boxes: tables
    notes: list[str] = field(default_factory=list)
    # Ruled lines as the cleaner found them, for putting tables together: every rule across the page
    # (those of a table's grid and those standing alone) and every upright.
    grid: list[tuple[int, int, int, int]] = field(default_factory=list)
    uprights: list[tuple[int, int, int, int]] = field(default_factory=list)
    short: list[tuple[int, int, int, int]] = field(default_factory=list)  # short rules inside ruled frames (tables.short_rules)


def _strokes(gray: np.ndarray) -> tuple[list, list, float]:
    """Horizontal rules and ruled frames on a page, and the page's letter height."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(binarize(gray), connectivity=8)
    if n <= 1:
        return [], [], 0.0
    x, y, w, h, a = (stats[1:, i] for i in range(5))
    tall = float(np.percentile(h[a >= 3], 90)) if (a >= 3).any() else 0.0
    rules = [(int(x[i]), int(y[i]), int(x[i] + w[i]), int(y[i] + h[i])) for i in np.flatnonzero((w > 4 * tall) & (h < 0.4 * tall))]
    frames = [(int(x[i]), int(y[i]), int(x[i] + w[i]), int(y[i] + h[i])) for i in np.flatnonzero((w > 8 * tall) & (h > 6 * tall))]
    return rules, frames, tall


def read_page(ocr: OCR, gray: np.ndarray, number: int) -> PageInfo:
    doc = ocr.read_page(gray)
    rules, frames, _ = _strokes(gray)
    lines, cols, notes = [], [], []
    for ri, (r, res) in enumerate(doc.regions):
        cols.append((r.x0, r.y0, r.x1, r.y1))
        notes.extend(res.notes)
        ox, oy = r.x0 - 16, r.y0 - 16
        for ln_geo, ln in zip(res.layout.lines, res.lines):
            if not ln.clusters:
                continue
            ids = [b for c in ln.clusters for b in c.blobs]
            bx = res.layout.boxes[ids]
            lines.append(TextLine(number, ri, int(bx[:, 0].min()) + ox, int(bx[:, 1].min()) + oy, int(bx[:, 2].max()) + ox,
                                  int(bx[:, 3].max()) + oy, ln_geo.baseline + oy, ln_geo.scale, ln.text, len(ln.clusters), len(lines)))
    if lines:
        from ..structure.features import ink_features
        for l, f in zip(lines, ink_features(gray, np.array([[l.x0, l.y0, l.x1, l.y1] for l in lines], np.float32))):
            l.ink = tuple(float(v) for v in f)
    # The reader cleaned this page already and hands on what it painted out: the ruled lines, which are
    # the exact grid of a ruled table. (A reader that does not hand it on: clean again, about 0.7 s.)
    cleaned = getattr(doc, "cleaned", None)
    if cleaned is None:
        from ..blobs.clean import clean_page
        cleaned = clean_page(gray)
    from .tables import short_rules
    grid = cleaned.grid + cleaned.rules
    lh = float(np.median([l.y1 - l.y0 for l in lines])) if lines else 30.0
    return PageInfo(number, gray.shape[1], gray.shape[0], lines, cols, rules, frames, notes,
                    grid, cleaned.uprights, short_rules(gray, grid, cleaned.uprights, lh, [(l.x0, l.y0, l.x1, l.y1) for l in lines]))


FOOTNOTE_RULE = True  # running text under a short rule at a column's edge is footnotes (_footnotes_under_rule)
MARGINS = True  # running heads and feet found across pages before typing (_margin_lines)
MARKS = True  # look for item marks the reader gave no letter for (unread_marks)
BULLET = "•"  # what a mark the reader could not read is written as


def _lead(ink: np.ndarray, l: TextLine, ltr: bool) -> dict | None:
    """The first piece of ink on a line, in reading direction (the right-hand end for Arabic): its width,
    the blank gap after it, its height, its picture, and how much ink the rest of the line spans."""
    a = ink[max(l.y0, 0) : l.y1, max(l.x0, 0) : l.x1]
    if not ltr:
        a = a[:, ::-1]
    col = a.sum(0) > 0
    n, i = len(col), 0
    while i < n and not col[i]:
        i += 1
    s = i
    while i < n and col[i]:
        i += 1
    e = i
    while i < n and not col[i]:
        i += 1
    if e <= s or i >= n:
        return None
    rows = np.flatnonzero(a[:, s:e].sum(1) > 0)
    return {"w": e - s, "gap": i - e, "h": int(rows[-1] - rows[0] + 1), "pic": a[rows[0] : rows[-1] + 1, s:e], "rest": n - i}


def _same_shape(a: np.ndarray, b: np.ndarray) -> float:
    """Overlap of two small pictures brought to one size (1 = the same shape)."""
    A = cv2.resize(a.astype(np.uint8) * 255, (24, 24), interpolation=cv2.INTER_AREA) > 64
    B = cv2.resize(b.astype(np.uint8) * 255, (24, 24), interpolation=cv2.INTER_AREA) > 64
    return float((A & B).sum()) / max(int((A | B).sum()), 1)


def unread_marks(page: PageInfo, gray: np.ndarray) -> int:
    """Put a bullet at the head of lines that open with an item mark the reader gave no letter for.

    Ornamental marks (a pointing hand, a four-diamond sign) are not in the reader's alphabet: their ink is
    inside the line but its text starts at the word after them, so nothing downstream knows an item opens
    there. The page still shows it. A line counts when its first piece of ink is small (at most about one
    and a half letters wide, no taller than a letter), set off by a gap, too narrow to hold the first
    word the reader gave (so the reader did not read it as that word), and the same shape stands at the
    same place at the head of another line of the column: one mark repeated down a list. Numbers and
    cells of a table row are left alone: a list mark opens words, and a cell has neighbours level with it.
    Returns how many lines were marked."""
    from ..blobs.regions import is_ltr
    from ..structure.tag import _ITEM

    if not page.lines:
        return 0
    ink = binarize(gray)
    ltr = is_ltr(gray)
    ls = page.lines
    cand = {}
    for k, l in enumerate(ls):
        t = strip_marks(l.text).strip()
        if not t or _ITEM.match(t):
            continue
        w0 = t.split()[0]
        if sum(c.isalpha() for c in w0) < 2:
            continue
        if any(o is not l and o.region == l.region and min(o.y1, l.y1) - max(o.y0, l.y0) > 0.5 * min(o.y1 - o.y0, l.y1 - l.y0) for o in ls):
            continue
        L = _lead(ink, l, ltr)
        sc = l.scale
        if L is None or not (0.25 * sc <= L["w"] <= 1.6 * sc and L["h"] <= 1.4 * sc and L["gap"] >= 0.45 * sc):
            continue
        per_letter = L["rest"] / max(len(t.replace(" ", "")), 1)
        if len(w0) * per_letter < 1.5 * L["w"]:
            continue  # the reader's first word would fit in that ink: it is that word
        cand[k] = L
    hits = set()
    for a, A in cand.items():
        la = ls[a]
        for b, B in cand.items():
            lb = ls[b]
            if b == a or la.region != lb.region:
                continue
            sc = max(la.scale, lb.scale)
            edge = abs((la.x0 - lb.x0) if ltr else (la.x1 - lb.x1))
            if edge < 0.4 * sc and abs(A["w"] - B["w"]) < 0.25 * max(A["w"], B["w"]) and _same_shape(A["pic"], B["pic"]) > 0.6:
                hits.add(a)
                break
    for k in hits:
        ls[k].text = BULLET + " " + ls[k].text.strip()
    return len(hits)


def _cut_blocks(page: PageInfo) -> list[Block]:
    """Blocks of one page, in reading order: each column's lines, cut where a paragraph ends."""
    blocks: list[Block] = []
    for ri, col in enumerate(page.columns):
        ls = sorted((l for l in page.lines if l.region == ri), key=lambda l: l.baseline)
        if not ls:
            continue
        right = float(np.percentile([l.x1 for l in ls], 90))
        left = float(np.percentile([l.x0 for l in ls], 10))
        pitches = np.diff([l.baseline for l in ls])
        pitch = float(np.median(pitches)) if len(pitches) else 0.0
        scale = float(np.median([l.scale for l in ls]))
        cur = [ls[0]]
        column = len(ls) >= 4  # indents and short lines only mean something against a column's edges
        for prev, ln in zip(ls, ls[1:]):
            gap = ln.baseline - prev.baseline
            indent = right - ln.x1  # Arabic lines start at the right
            prev_short = prev.x0 - left  # the line above stopped early
            cut = (
                gap > 1.35 * pitch + 0.15 * scale  # more air than between ordinary lines
                or (column and indent > 0.9 * scale and right - prev.x1 < 0.5 * scale and ln.x0 - left < 2.5 * scale)  # a first-line indent
                # the line above stopped early, and either a fuller line follows or it closed its sentence
                or (column and prev_short > 2.5 * scale and (ln.x0 - left < 0.5 * prev_short or prev.text.rstrip().endswith((".", ":", "؟"))))
                or abs(ln.scale - prev.scale) > 0.3 * max(ln.scale, prev.scale)  # letter size changed
            )
            if cut:
                blocks.append(Block(cur))
                cur = []
            cur.append(ln)
        blocks.append(Block(cur))
    return blocks


_REF_WORDS = ("المراجع", "مراجع", "المصادر", "الهوامش", "references")


def _is_ref_heading(text: str) -> bool:
    """A references heading, allowing for a letter or two misread."""
    words = strip_marks(text).lower().replace(":", " ").split()
    return len(words) <= 4 and any(Levenshtein.distance(w, r) <= 1 for w in words for r in _REF_WORDS)


def _similar(a: str, b: str) -> float:
    a, b = strip_marks(a), strip_marks(b)
    return 1 - Levenshtein.distance(a, b) / max(len(a), len(b), 1)


def parse(ocr: OCR, pages: list[np.ndarray], first: int = 1, typer=None,
          progress: Callable[[int, int], None] | None = None) -> tuple[list[PageInfo], list[Block]]:
    """`first` is the page number of the first image, for a document parsed from the middle.

    With a `typer` (ocrl.structure.tag.Typer), block cuts and types come from the trained structure
    model; without one, from the hand-written rules below.

    `progress`, if given, is called as progress(pages_read, pages_total) after each page is read.
    It only reports; it does not change what is parsed.
    """
    infos = []
    for n, g in enumerate(pages, first):
        infos.append(read_page(ocr, g, n))
        if MARKS:
            unread_marks(infos[-1], g)
        if progress is not None:
            progress(len(infos), len(pages))
    if typer is not None:
        return infos, _typed_blocks(infos, typer)
    by_no = {p.number: p for p in infos}
    per_page = [_cut_blocks(p) for p in infos]

    # --- page furniture -------------------------------------------------------------
    tops = {}
    for p, blocks in zip(infos, per_page):
        for b in blocks:
            x0, y0, x1, y1 = b.box()
            one = len(b.lines) == 1
            body_top = min((l.y0 for l in p.lines if l.clusters >= 12), default=0)
            body_bottom = max((l.y1 for l in p.lines if l.clusters >= 12), default=p.height)
            # A page number is a tiny lone line in the top or bottom margin, clear of the text. Its place
            # says so even when its digits were misread.
            tiny = one and b.lines[0].clusters <= 5 and x1 - x0 < 0.1 * p.width
            if tiny and (y0 > body_bottom or y1 < body_top) and (y0 > 0.8 * p.height or y1 < 0.14 * p.height):
                b.type = "page_number"
            elif not b.text.strip() or (b.lines[0].clusters <= 2 and one and not any(c.isalnum() for c in b.text)):
                b.type = "noise"
        head = next((b for b in sorted(blocks, key=lambda b: b.box()[1]) if b.type == "paragraph"), None)
        if head and len(head.lines) == 1 and head.box()[3] < 0.16 * p.height:
            tops[p.number] = head
    # A line that sits at the top of a page and comes back on other pages is a running head.
    # So is a lone top line with a long rule drawn under it.
    for n, b in tops.items():
        again = any(m != n and _similar(b.text, o.text) > 0.5 for m, o in tops.items())
        y1 = b.box()[3]
        ruled = any(0 <= r[1] - y1 < 4 * b.lines[0].scale and r[2] - r[0] > 0.3 * by_no[n].width for r in by_no[n].rules)
        if again or ruled:
            b.type = "running_head"

    # --- tables: whatever sits inside a ruled frame is one table ---------------------
    for p, blocks in zip(infos, per_page):
        for fx0, fy0, fx1, fy1 in p.frames:
            inside = lambda l: fx0 <= (l.x0 + l.x1) / 2 <= fx1 and fy0 <= (l.y0 + l.y1) / 2 <= fy1
            caught = [l for b in blocks for l in b.lines if inside(l)]
            if len(caught) < 3:
                continue
            first = next(i for i, b in enumerate(blocks) if any(inside(l) for l in b.lines))
            for b in blocks:
                b.lines = [l for l in b.lines if not inside(l)]
            blocks.insert(first, Block(sorted(caught, key=lambda l: (l.baseline, -l.x1)), "table"))
            blocks[:] = [b for b in blocks if b.lines]

    # --- the first page: title, author, affiliation, abstract ------------------------
    if per_page:
        p, blocks = infos[0], per_page[0]
        body = [b for b in blocks if b.type == "paragraph"]
        if body:
            med = float(np.median([l.scale for b in body for l in b.lines]))
            text_w = max(l.x1 for l in p.lines) - min(l.x0 for l in p.lines)
            # The title is the first text on the first page, set larger than the body or centred on its own.
            title = body[0] if body[0].box()[1] < 0.4 * p.height and len(body[0].lines) <= 4 else None
            if title and body[0].box()[2] - body[0].box()[0] < 0.7 * text_w:
                title.type = "title"
                rest = body[body.index(title) + 1 :]
                narrow = lambda b: b.box()[2] - b.box()[0] < 0.5 * text_w and len(b.lines) <= 3
                if rest and narrow(rest[0]):
                    rest[0].type = "author"
                    if len(rest[0].lines) > 1:  # the name, with post and faculty set under it as one group
                        under = Block(rest[0].lines[1:], "affiliation")
                        rest[0].lines = rest[0].lines[:1]
                        blocks.insert(blocks.index(rest[0]) + 1, under)
                    else:
                        for b in rest[1:4]:
                            if not narrow(b):
                                break
                            b.type = "affiliation"
                # Full-width text before the page breaks into columns is the abstract.
                for b in rest:
                    if b.type != "paragraph":
                        continue
                    if b.box()[2] - b.box()[0] > 0.75 * text_w and len(b.lines) >= 2:
                        b.type = "abstract"
                    elif any(o.type == "abstract" for o in rest):
                        break

    # --- headings, list items, footnotes, references ---------------------------------
    in_refs = False
    last_ref: Block | None = None
    for pi, (p, blocks) in enumerate(zip(infos, per_page)):
        short_rules = [r for r in p.rules if r[2] - r[0] < 0.3 * p.width and r[1] > 0.5 * p.height]
        rebuilt: list[Block] = []
        for b in blocks:
            if b.type != "paragraph":
                rebuilt.append(b)
                continue
            x0, y0, x1, y1 = b.box()
            cx0, _, cx1, _ = p.columns[b.lines[0].region]
            if sum(l.region == b.lines[0].region for l in p.lines) < 4 and p.lines:
                wide = [l for l in p.lines if l.clusters >= 8] or p.lines
                cx0, cx1 = min(l.x0 for l in wide), max(l.x1 for l in wide)
            text = b.text.strip()
            one = len(b.lines) == 1
            lone = one and x1 - x0 < 0.72 * (cx1 - cx0)
            centred = cx1 - x1 > 1.5 * b.lines[0].scale and x0 - cx0 > 1.5 * b.lines[0].scale
            if lone and centred and _is_ref_heading(text):
                b.type, in_refs, last_ref = "heading", True, None  # each references heading opens a new list
                rebuilt.append(b)
            elif in_refs:
                # A reference opens with its number; the lines after it, down to the next number, are its rest.
                for ln in b.lines:
                    if _REF_MARK.match(ln.text.strip()) or last_ref is None:
                        last_ref = Block([ln], "reference")
                        rebuilt.append(last_ref)
                    elif (last_ref.lines[-1].page, last_ref.lines[-1].region) == (ln.page, ln.region):
                        last_ref.lines.append(ln)
                    else:  # the reference carries on in the next column or on the next page
                        nxt = Block([ln], "reference", parent=last_ref.parent or last_ref)
                        rebuilt.append(nxt)
                        last_ref = nxt
            else:
                if any(r[3] <= y0 and min(r[2], cx1) - max(r[0], cx0) > 0 for r in short_rules):
                    b.type = "footnote"  # under a short rule near the foot of its column
                elif lone and not text.endswith(".") and (text.endswith(":") or centred) and not _MARKER.match(text):
                    b.type = "heading"  # a short line on its own: ends in a colon, or is centred
                elif _MARKER.match(text):
                    b.type = "list_item"
                rebuilt.append(b)
        per_page[pi] = rebuilt

    # --- one text, one order -----------------------------------------------------------
    out: list[Block] = []
    for blocks in per_page:
        out.extend(blocks)
    # A paragraph that reaches the foot of a column with a full last line, followed by a column
    # that opens without an indent, is one paragraph in two parts.
    flow = [b for b in out if b.type == "paragraph"]
    for a, b in zip(flow, flow[1:]):
        la, lb = a.lines[-1], b.lines[0]
        if a.type != b.type or (la.page, la.region) == (lb.page, lb.region):
            continue
        ca, cb = by_no[la.page].columns[la.region], by_no[lb.page].columns[lb.region]
        full = la.x0 - ca[0] < 1.5 * la.scale + 22  # the last line runs to the column's edge
        flush = cb[2] - lb.x1 < 0.9 * lb.scale + 22  # the next column opens without an indent
        same_width = abs((ca[2] - ca[0]) - (cb[2] - cb[0])) < 0.25 * max(ca[2] - ca[0], cb[2] - cb[0])
        if a.type == "paragraph" and same_width and full and flush and not a.text.rstrip().endswith((".", ":", "؟")):
            b.parent = a.parent or a
    order = 0
    for i, b in enumerate(out):
        b.id = f"b{i}"
        if b.type in ("running_head", "page_number", "noise"):
            b.order = None
        elif b.parent is not None:
            b.order = b.parent.order
        else:
            order += 1
            b.order = order
    return infos, out


def _margin_words(text: str) -> str:
    """A line's words without its digits (a running head carries the page number, or a year that changes)."""
    return " ".join("".join(c for c in strip_marks(text) if not c.isdigit()).split())


def _margin_lines(infos: list[PageInfo]) -> list[dict[int, str]]:
    """Per page, the lines (by index in page.lines) in the top or bottom margin whose words stand in the same
    margin of another page: a running head or foot, even when it was set on two lines and the second sits
    close above the text. Only with two pages or more."""
    out: list[dict[int, str]] = [{} for _ in infos]
    if len(infos) < 2:
        return out
    seen = []
    for k, p in enumerate(infos):
        for i, l in enumerate(p.lines):
            side = "top" if l.y1 < 0.12 * p.height else "foot" if l.y0 > 0.88 * p.height else ""
            w = _margin_words(l.text)
            if side and len(w) >= 6 and not _REF_MARK.match(l.text.strip()):  # a numbered note ("ibid.") comes back too
                seen.append((k, i, side, w, (l.y0 + l.y1) / 2, l.y1 - l.y0))
    # the same words at the same height on the page: a note that repeats sits wherever its page's notes end
    for k, i, side, w, y, h in seen:
        if any(k2 != k and s2 == side and abs(y2 - y) < 1.5 * max(h, h2) and _similar(w, w2) > 0.8 for k2, _, s2, w2, y2, h2 in seen):
            out[k][i] = "running_head" if side == "top" else "running_foot"
    return out


def _typed_blocks(infos: list[PageInfo], typer) -> list[Block]:
    """Blocks from the structure model: it names each line and marks where blocks begin; we only group."""
    from ..structure.tag import ASIDE, TO_TYPE

    out: list[Block] = []
    last_flow: Block | None = None  # the last block in the running text, for one that carries on
    margins = _margin_lines(infos) if MARGINS else [{} for _ in infos]
    for p, margin in zip(infos, margins):
        ls = sorted(p.lines, key=lambda l: (l.region, l.order))
        if not ls:
            continue
        ink = np.array([l.ink for l in ls], np.float32) if all(len(l.ink) == 3 for l in ls) else None
        at = {id(l): i for i, l in enumerate(ls)}
        margin = {at[id(p.lines[i])]: kind for i, kind in margin.items()}
        roles, starts, _ = typer.tag(np.array([[l.x0, l.y0, l.x1, l.y1] for l in ls]), [l.text for l in ls], p.width, p.height, ink,
                                     **({"margin": margin} if margin else {}))
        kinds = [TO_TYPE.get(r, "other") for r in roles]
        # One block per table, put together from its cells (tables.py), standing where its first line is read.
        found = find_tables(p, ls, kinds)
        owner = {i: t for t in found for i in t.members}
        placed: set[int] = set()
        cur: Block | None = None
        for i, (ln, kind, start) in enumerate(zip(ls, kinds, starts)):
            if i in owner:
                t = owner[i]
                if id(t) not in placed:
                    placed.add(id(t))
                    lh = float(np.median([l.y1 - l.y0 for l in t.lines]))
                    if len(t.lines) == 1 and not t.ruled and len(t.lines[0].text.strip()) < 4:
                        cur = Block(t.lines, "noise")  # a speck or a letter on its own is not a table
                    elif t.n_rows == 1 and not t.ruled and any(
                            o is not t and o.n_rows > 1 and min(abs(o.box[1] - t.box[3]), abs(t.box[1] - o.box[3])) < 3 * lh for o in found):
                        cur = Block(t.lines, "table_caption")  # one row of words over or under a table: its title
                    else:
                        cur = Block(t.lines, "table", table=t)
                    out.append(cur)
                continue
            same_place = cur is not None and (cur.lines[-1].page, cur.lines[-1].region) == (ln.page, ln.region)
            if cur is not None and cur.type == kind and same_place and start < 0.5 and kind not in ASIDE:
                cur.lines.append(ln)
                continue
            cur = Block([ln], kind)
            # Not marked as a start, yet first in its column: it carries on a block from the column or page before.
            if start < 0.5 and kind not in ASIDE and last_flow is not None and last_flow.type == kind:
                cur.parent = last_flow.parent or last_flow
            out.append(cur)
            if kind not in ASIDE and kind not in ("table", "footnote", "figure"):
                last_flow = cur
    if FOOTNOTE_RULE:
        _footnotes_under_rule(infos, out)
    _references_go_on(infos, out)
    if len(infos) > 1:
        _page_furniture(infos, out)
    _join_tables(out)
    order = 0
    for i, b in enumerate(out):
        b.id = f"b{i}"
        if b.type in ASIDE:
            b.order = None
        elif b.parent is not None:
            b.order = b.parent.order
        else:
            order += 1
            b.order = order
    return out


_UNDER_RULE = {"paragraph", "list_item", "other", "standfirst", "pull_quote", "verse", "abstract"}  # what a footnote gets called


def _footnotes_under_rule(infos: list[PageInfo], blocks: list[Block]) -> None:
    """Footnotes stand under a short rule: a stroke a fifth to a half of the column wide, starting at the
    column's edge (the right for Arabic, the left for Latin), in the lower part of the page, with the body text
    above it and clear air above and below it (an underline has its words sitting on it). Running text below such a rule, in that column, is footnotes, whatever its first line looks like
    (a numbered note reads like a list item, a note's second paragraph like a paragraph)."""
    by_no = {p.number: p for p in infos}
    cuts: dict[int, list[tuple[int, int, int]]] = {}
    for p in infos:
        if not p.lines:
            continue
        lh = float(np.median([l.y1 - l.y0 for l in p.lines]))
        for r in p.rules:
            if r[1] < 0.4 * p.height:
                continue
            span = [l for l in p.lines if l.x1 > r[0] and l.x0 < r[2]]
            if not span:
                continue
            cx0, cx1 = min(l.x0 for l in span), max(l.x1 for l in span)
            w = (r[2] - r[0]) / max(cx1 - cx0, 1)
            flush = abs(r[2] - cx1) < 1.5 * lh or abs(r[0] - cx0) < 1.5 * lh
            # clear of text above and below: an underline has its words sitting on it
            clear = not any(r[1] - 0.6 * lh < l.y1 and l.y0 < r[3] + 0.3 * lh for l in span)
            if 0.2 <= w <= 0.5 and flush and clear and sum(l.y1 <= r[1] for l in span) >= 3:
                cuts.setdefault(p.number, []).append((r[1], cx0, cx1))
    if not cuts:
        return
    for b in blocks:
        if b.type not in _UNDER_RULE or b.table is not None or b.page not in cuts:
            continue
        x0, y0, x1, _ = b.box()
        if any(y0 >= ry and min(x1, cx1) - max(x0, cx0) > 0.5 * (x1 - x0) for ry, cx0, cx1 in cuts[b.page]):
            b.type = "footnote"


def _references_go_on(infos: list[PageInfo], blocks: list[Block]) -> None:
    """A list of references does not turn into footnotes halfway down. Numbered entries under a references
    heading, or straight after references with nothing between them, are references too, though each looks
    like a footnote (a number in brackets, small type, a source cited): a footnote stands under a rule at the
    foot of the text it annotates, so one with a rule above it, or with body text between, keeps its name."""
    by_no = {p.number: p for p in infos}
    prev: Block | None = None  # the last block in the reading order (page furniture aside)
    for b in blocks:
        if b.type in ("running_head", "running_foot", "page_number", "noise"):
            continue
        if b.type == "footnote" and prev is not None and b.page == prev.page and (
                prev.type == "reference" or (prev.type == "heading" and _is_ref_heading(prev.text))):
            top, above = b.box()[1], prev.box()[3]
            if not any(above <= r[1] <= top for r in by_no[b.page].rules + by_no[b.page].grid):
                b.type = "reference"
        prev = b


_FOLIO = re.compile(rf"^[\s\-–—()|.]*([{DIGITS}]{{1,4}})[\s\-–—()|.]*$")
# what the page's place and text can overrule: not a title, an author or a table, which a page's top line can also be
_OVERRULE = {"paragraph", "heading", "other", "noise", "figure", "figure_caption", "equation", "section_label", "standfirst",
             "pull_quote", "running_head", "running_foot", "page_number", "list_item", "footnote"}


def _page_furniture(infos: list[PageInfo], blocks: list[Block]) -> None:
    """What only the whole document shows, decided after every page is typed: a number in a page's top or
    bottom margin that steps by one from page to page is the page number, and a line at the top (or foot)
    of a page whose words come back at the top (or foot) of another page is a running head (or foot).
    Only lone lines in the margins are looked at, and the model's own call stands unless the pattern holds
    on at least two pages."""
    by_no = {p.number: p for p in infos}
    carried = {id(b.parent) for b in blocks if b.parent is not None}
    lone = [b for b in blocks if len(b.lines) == 1 and b.table is None and b.parent is None and id(b) not in carried and b.type in _OVERRULE]

    def band(b) -> str:
        p = by_no[b.page]
        _, y0, _, y1 = b.box()
        return "top" if y1 < 0.12 * p.height else "foot" if y0 > 0.88 * p.height else ""

    # page numbers: the same offset between the printed number and the page's place, on two pages or more
    folios = []
    for b in lone:
        m = _FOLIO.match(strip_marks(b.text))
        if m and band(b):
            folios.append((b, int(m.group(1)) - b.page))
    offsets = {}
    for b, off in folios:
        offsets.setdefault(off, set()).add(b.page)
    for b, off in folios:
        if len(offsets[off]) >= 2:
            b.type = "page_number"
    # running heads and feet: words that come back in the same margin of another page (numbers aside)
    words = lambda b: " ".join("".join(c for c in strip_marks(b.text) if not c.isdigit()).split())
    for b in lone:
        side = band(b)
        if not side or b.type == "page_number" or len(words(b)) < 6:
            continue
        again = any(o.page != b.page and band(o) == side and _similar(words(b), words(o)) > 0.8 for o in lone)
        if again:
            b.type = "running_head" if side == "top" else "running_foot"


def _join_tables(blocks: list[Block]) -> None:
    """A table that ends its page and one that opens the next, with the same columns, are one table;
    and a caption directly above or below a table is that table's caption."""
    from ..structure.tag import ASIDE
    from .tables import continues

    flow = [b for b in blocks if b.type not in ASIDE]
    for a, b in zip(flow, flow[1:]):
        if a.type == b.type == "table" and a.table is not None and b.table is not None and continues(a.table, b.table):
            b.parent = a.parent or a
    for i, b in enumerate(blocks):
        if b.type != "table" or b.table is None:
            continue
        t = b.table
        lh = float(np.median([l.y1 - l.y0 for l in b.lines]))
        near = [o for o in blocks[max(i - 3, 0) : i + 4] if o.type == "table_caption" and o.page == b.page]
        for o in sorted(near, key=lambda o: abs(blocks.index(o) - i)):
            x0, y0, x1, y1 = o.box()
            gap = t.box[1] - y1 if y1 <= t.box[1] + lh else y0 - t.box[3]
            if -lh <= gap < 4 * lh and min(x1, t.box[2]) - max(x0, t.box[0]) > 0:
                b.caption = o
                break


def _table_json(group: list[Block], dpi: int) -> dict:
    """A table's grid for the master file: a header row and rows of cells, each row one entry per column in
    reading order. A cell is {"text", "lines": its line boxes, "box": its place, "page"} with "rowspan" or
    "colspan" when it covers more than one place; None is an empty place, "^" a place covered by the cell
    above, "<" a place covered by the cell before it in the row. A table that carries on over the page is one
    table: the next page's repeat of the header is not a row, its lines are listed in repeated_header_lines."""
    from .tables import Cell, repeated_header

    inch = lambda b: [round(v / dpi, 3) for v in b]
    first = group[0].table

    def cell(c, t):
        if not isinstance(c, Cell):
            return c
        out = {"text": c.text, "lines": [inch((l.x0, l.y0, l.x1, l.y1)) for l in c.lines], "page": t.page, "box": inch(c.box)}
        if c.rowspan > 1:
            out["rowspan"] = c.rowspan
        if c.colspan > 1:
            out["colspan"] = c.colspan
        return out

    header, rows, repeated = [], [], []
    for k, g in enumerate(group):
        t = g.table
        grid = t.grid()
        start = 0
        if k == 0 and t.header_rows:
            header, start = [cell(c, t) for c in grid[0]], 1
        elif k > 0 and repeated_header(first, t):
            repeated += [inch((l.x0, l.y0, l.x1, l.y1)) for c in grid[0] if isinstance(c, Cell) for l in c.lines]
            start = 1
        for j, r in enumerate(grid[start:]):
            row = [cell(c, t) for c in r]
            if k > 0 and j == 0:
                row = [None if c == "^" else c for c in row]  # nothing above it on this page
                for col, top in enumerate(row):
                    _carry_over(rows, col, top, row)
            rows.append(row)
    out = {"header": header, "rows": rows, "columns": first.n_cols, "direction": "rtl" if first.rtl else "ltr",
           "ruled": first.ruled}
    if repeated:
        out["repeated_header_lines"] = repeated
    cap = next((g.caption for g in group if g.caption is not None), None)
    if cap is not None:
        out["caption"] = cap.id
    return out


def _carry_over(rows: list[list], col: int, top, row: list) -> None:
    """A cell that spans rows and is cut by the page break: an empty spanning cell at the foot of one page
    and a filled cell at the head of the next, in the same column, are one cell (a category label beside a
    group of rows that starts on one page and ends on the next). The label is printed on one side only."""
    if not isinstance(top, dict) or not rows:
        return
    i = len(rows) - 1
    while i > 0 and rows[i][col] == "^":
        i -= 1
    origin = rows[i][col]
    if not isinstance(origin, dict) or origin["lines"] or not (origin.get("rowspan", 1) > 1 or top.get("rowspan", 1) > 1):
        return
    span = len(rows) - i + top.get("rowspan", 1)
    origin.update({k: top[k] for k in ("text", "lines", "page", "box")})
    origin["rowspan"] = span
    row[col] = "^"


def _table_text(table: dict) -> str:
    """Plain text of a table: a row to a line, cells separated by a bar."""
    rows = ([table["header"]] if table["header"] else []) + table["rows"]
    return "\n".join(" | ".join(c["text"] if isinstance(c, dict) else "" for c in r) for r in rows)


def to_master(doc_id: str, infos: list[PageInfo], blocks: list[Block], dpi: int = 300, reader: str = "", structure: str = "") -> dict:
    """The same shape as the inkscript master file, as far as this parser goes."""
    out = []
    for b in blocks:
        if b.parent is not None:
            continue
        group = [b] + [o for o in blocks if o.parent is b]
        parts = []
        for g in group:
            regions: dict = {}
            for l in g.lines:
                regions.setdefault((l.page, l.region), []).append(l)
            for (page, _), ls in regions.items():
                box = [round(v / dpi, 3) for v in g.box(ls)]
                lines = [[round(v / dpi, 3) for v in (l.x0, l.y0, l.x1, l.y1)] for l in ls]
                if parts and parts[-1]["page"] == page:
                    parts[-1]["regions"].append(box); parts[-1]["line_boxes"] += lines
                else:
                    parts.append({"page": page, "regions": [box], "line_boxes": lines})
        entry = {"id": b.id, "type": b.type, "text": " ".join(g.text for g in group), "parts": parts}
        if b.type == "table" and b.table is not None:
            # one region per page: the table's own box, not the boxes of the columns its cells fell into
            parts = [{"page": g.table.page, "regions": [[round(v / dpi, 3) for v in g.table.box]],
                      "line_boxes": [[round(v / dpi, 3) for v in (l.x0, l.y0, l.x1, l.y1)] for l in g.table.lines]} for g in group]
            entry["parts"] = parts
            entry["table"] = _table_json(group, dpi)
            entry["text"] = _table_text(entry["table"])
        elif b.type == "table_caption":
            of = next((o for o in blocks if o.caption is b), None)
            if of is not None:
                entry["caption_of"] = of.id
        if b.order is not None:
            entry["order"] = b.order
        else:
            entry["aside"] = True
        out.append(entry)
    return {
        "id": doc_id,
        "made_by": {"text": f"OCRL reader ({reader}), local, no outside service", "structure": structure or "OCRL geometry rules, local, no model. Not checked by a person."},
        "pages": [{"number": p.number, "width": round(p.width / dpi, 4), "height": round(p.height / dpi, 4), "unit": "inch"} for p in infos],
        "blocks": out,
    }
