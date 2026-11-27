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
from dataclasses import dataclass, field

import cv2
import numpy as np
from rapidfuzz.distance import Levenshtein

from ..blobs.layout import binarize
from ..pipeline import OCR
from ..text import DIGITS, strip_marks

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


@dataclass
class Block:
    lines: list[TextLine]
    type: str = "paragraph"
    order: int | None = None
    id: str = ""
    parent: "Block | None" = None  # the block this one carries on from, over a column or page break

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
    return PageInfo(number, gray.shape[1], gray.shape[0], lines, cols, rules, frames, notes)


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


def parse(ocr: OCR, pages: list[np.ndarray], first: int = 1, typer=None) -> tuple[list[PageInfo], list[Block]]:
    """`first` is the page number of the first image, for a document parsed from the middle.

    With a `typer` (ocrl.structure.tag.Typer), block cuts and types come from the trained structure
    model; without one, from the hand-written rules below.
    """
    infos = [read_page(ocr, g, n) for n, g in enumerate(pages, first)]
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


def _typed_blocks(infos: list[PageInfo], typer) -> list[Block]:
    """Blocks from the structure model: it names each line and marks where blocks begin; we only group."""
    from ..structure.tag import ASIDE, TO_TYPE

    out: list[Block] = []
    last_flow: Block | None = None  # the last block in the running text, for one that carries on
    for p in infos:
        ls = sorted(p.lines, key=lambda l: (l.region, l.order))
        if not ls:
            continue
        roles, starts, _ = typer.tag(np.array([[l.x0, l.y0, l.x1, l.y1] for l in ls]), [l.text for l in ls], p.width, p.height)
        cur: Block | None = None
        for ln, role, start in zip(ls, roles, starts):
            kind = TO_TYPE.get(role, "other")
            if kind == "table":  # one block per table: its cells are lines of the same block
                if cur is not None and cur.type == "table" and cur.lines[-1].page == ln.page:
                    cur.lines.append(ln)
                    continue
                cur = Block([ln], "table"); out.append(cur)
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
