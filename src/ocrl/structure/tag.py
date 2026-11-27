"""Use the structure model: lines of a page in, what each one is and where blocks begin out."""

from __future__ import annotations

import re

import numpy as np
import torch

from .features import N_GEOM, N_INK, ROLES, geometry, text_ids
from .model import load
from ..text import strip_marks

# The Desk database's roles, in inkscript_v2's fixed list of block types.
TO_TYPE = {
    "paragraph": "paragraph", "heading": "heading", "list-item": "list_item", "ref": "reference", "footnote": "footnote",
    "endnote": "footnote", "caption": "table_caption", "table-cell": "table", "verse-group": "verse", "abstract": "abstract",
    "title": "title", "title-translated": "title", "byline": "author", "front-matter": "affiliation", "back-matter": "other",
    "keywords": "keywords", "quote": "pull_quote", "pageHeader": "running_head", "pageNumber": "page_number",
    "pageFooter": "running_foot", "noise": "noise", "pictureText": "figure", "pictureJunk": "noise", "ornament": "noise",
    "otherArticle": "other", "formulaInk": "equation", "showThrough": "noise",
}
# A model retrained on block types names them directly: those names pass through unchanged.
TYPES = ["paragraph", "heading", "list_item", "table", "table_caption", "reference", "footnote", "running_head", "running_foot", "page_number", "noise",
         "title", "section_label", "author", "supervisor", "affiliation", "degree_statement", "standfirst", "abstract", "keywords", "figure",
         "figure_caption", "pull_quote", "equation", "verse", "handwriting", "other"]
TO_TYPE.update({t: t for t in TYPES if t not in TO_TYPE})
ASIDE = {"running_head", "running_foot", "page_number", "noise"}  # outside the reading order
MAXL = 320

# A list is written into the page: every item opens with a dash, a bullet, or a number or letter with its
# mark, and the next item opens the same way. The model learned its labels from pages where such items
# were more often called paragraphs, so this one is decided by rule.
_D = "0-9\u0660-\u0669\u06f0-\u06f9"
_ITEM = re.compile(rf"^\s*(?:[-–—−ـ•*·▪■●◦]+\s*\S|\(\s?[{_D}]{{1,3}}\s?\)\s*\S|[{_D}]{{1,3}}\s?[-–—ـ.)]\s*[^{_D}\s]|[أ-ي]\s?(?:[-–)]\s*|ـ\s+)[^\s{_D}])")


_STOP = re.compile(r"[.؟?!:]\s*$")
_NUM_MARK = re.compile(rf"^\s*(\(?)\s?([{_D}]{{1,3}})\s?([-–—ـ.)])")
_LETTER_MARK = re.compile(r"^\s*(\(?)\s?([أ-ي])\s?([-–)])")
_ORDERS = ("أبجدهوزحطي", "أبتثجحخدذرزسشصضطظعغفقكلمنهوي")
SEQUENCE = True


def _mark_key(t: str):
    """The mark that opens a line, as (kind, value, punctuation), when it is a number or a letter."""
    if not _ITEM.match(t):
        return None
    m = _NUM_MARK.match(t)
    if m:
        return ("n", int(m.group(2)), m.group(1) + m.group(3))
    m = _LETTER_MARK.match(t)
    if m:
        return ("a", m.group(2), m.group(1) + m.group(3))
    return None


def _steps(a, b) -> bool:
    """Mark b follows mark a in a list: the next number, or the next letter in either order of the alphabet."""
    if a[0] != b[0] or a[2] != b[2]:
        return False
    if a[0] == "n":
        return b[1] == a[1] + 1
    return any(a[1] in o and b[1] in o and o.index(b[1]) == o.index(a[1]) + 1 for o in _ORDERS)


def list_items(roles: list[str], starts: np.ndarray, texts: list[str], boxes: np.ndarray, open_item: bool = False,
               single: bool = False) -> tuple[list[str], np.ndarray, bool]:
    """Where two or more blocks on a page open with an item mark, they are list items, each mark opening a block.

    A mark counts only where a block could begin: the model thinks one does, or the line before stopped
    short, or there is a gap above. How far the item then runs is read off the page, not asked of the
    model: line after line until the next mark, a blank gap, or a line that stops short of the column's
    far edge (Arabic lines run right to left, so a paragraph's last line ends short on the left). An item
    cut by the foot of a column goes on at the head of the next one, and `open_item` says the page before
    ended inside an item. With `single`, one mark is enough when the model also calls that line an item.
    Returns the roles, the block starts, and whether this page ends inside an item."""
    n = len(texts)
    flow = ("paragraph", "list-item", "list_item", "otherArticle", "other")
    x0, y0, x1, y1 = (np.asarray(boxes, np.float32)[:, k] for k in range(4))
    h = np.maximum(y1 - y0, 1)

    def column(i: int) -> tuple[float, float]:
        """Left and right edge of the column line i sits in, from the lines around it."""
        near = [k for k in range(max(i - 8, 0), min(i + 9, n)) if min(x1[i], x1[k]) - max(x0[i], x0[k]) > 0.5 * min(x1[i] - x0[i], x1[k] - x0[k])]
        return min(x0[k] for k in near), max(x1[k] for k in near)

    def short(j: int) -> bool:
        """Line j is the last of its paragraph: it stops well short of the column's edge, or a little
        short with a full stop."""
        far_edge, near_edge = column(j)
        lack = x0[j] - far_edge
        return lack > 0.25 * (near_edge - far_edge) or (lack > 0.5 * h[j] and bool(_STOP.search(texts[j])))

    marked = []
    for i, t in enumerate(texts):
        m = bool(_ITEM.match(strip_marks(t))) and roles[i] in flow
        if m and i > 0 and starts[i] < 0.3:
            m = short(i - 1) or y0[i] - y1[i - 1] > 0.6 * h[i] or y0[i] < y0[i - 1] or marked[i - 1]
        marked.append(m)
    if SEQUENCE:
        # A number or letter that steps by one from an item's own (1. then 2., أ- then ب-) is an item too,
        # though nothing in the page's shape says a block begins there.
        keys = [_mark_key(strip_marks(t)) if roles[i] in flow else None for i, t in enumerate(texts)]
        cand = [i for i in range(n) if keys[i] is not None]
        changed = True
        while changed:
            changed = False
            for a, b in zip(cand, cand[1:]):
                if marked[a] != marked[b] and _steps(keys[a], keys[b]):
                    marked[a] = marked[b] = True
                    changed = True
    if sum(marked) + open_item < 2 and not (single and any(m and roles[i] in ("list-item", "list_item") for i, m in enumerate(marked))):
        return roles, starts, False
    roles, starts = list(roles), starts.copy()

    def run(j: int, hh: float, top: bool = False) -> tuple[int, bool]:
        """Carry an item on from line j. Returns the first line past it, and whether it was cut off unfinished.
        With `top`, line j is the page's first line of text and goes on the item whatever stands above it."""
        j0 = j
        while j < n and not marked[j] and roles[j] in flow:
            if top and j == j0:
                pass
            elif j > 0 and y0[j] < y0[j - 1] - hh:  # head of the next column: the item goes on only if it was not finished
                if short(j - 1):
                    return j, False
            elif j > 0 and (y0[j] - y1[j - 1] > hh or short(j - 1)):
                return j, False
            roles[j] = "list-item"; starts[j] = 0.0
            j += 1
        return j, j == n and n > 0 and not short(n - 1)

    still_open = False
    # The page's running head, page number and specks stand outside the text: an item carried over from the
    # page before goes on at the first line of text, and one that runs to the last line of text before the
    # page number is still open at the foot.
    aside = ASIDE | {"pageHeader", "pageNumber", "pageFooter", "pictureJunk", "showThrough"}
    first = next((k for k in range(n) if roles[k] not in aside), n)
    last = max((k for k in range(n) if roles[k] not in aside), default=-1)
    if open_item and first < n and not marked[first] and roles[first] in flow:
        j, still_open = run(first, h[first], top=True)
        if first < j < n and not marked[j]:
            starts[j] = max(starts[j], 0.6)
        if j < n and j > last and first <= last:
            still_open = not short(last)
    for i in range(n):
        if not marked[i]:
            continue
        starts[i] = 1.0
        roles[i] = "list-item"
        j, cut = run(i + 1, h[i]) if i + 1 < n else (n, not short(i))
        still_open = cut if j == n else (j > last and not short(j - 1))  # only page furniture after it
        if j < n and not marked[j]:
            starts[j] = max(starts[j], 0.6)  # whatever follows an item begins its own block
    return roles, starts, still_open


ENDS = True  # a short line that closes its sentence ends its paragraph (_paragraph_ends)


def _paragraph_ends(starts: np.ndarray, texts: list[str], boxes: np.ndarray, ltr: bool = False) -> np.ndarray:
    """Set in justified type, every line of a paragraph but its last runs the column's full width. A line that
    stops short by more than a letter or two and ends its sentence (a full stop, a question or exclamation
    mark, a colon) closes its paragraph, so the next line in the same column opens a new block, indent or no
    indent. Only where the column is plain running text: the line before and the line after are about as
    wide as the column, and the next line sits directly below."""
    n = len(texts)
    x0, y0, x1, y1 = (np.asarray(boxes, np.float32)[:, k] for k in range(4))
    h = np.maximum(y1 - y0, 1)
    out = starts.copy()
    for i in range(1, n):
        a = i - 1
        if not _STOP.search(texts[a]) or not (0 < y0[i] - y1[a] < 1.5 * h[i]):
            continue
        near = [k for k in range(max(a - 8, 0), min(a + 9, n)) if min(x1[a], x1[k]) - max(x0[a], x0[k]) > 0.5 * min(x1[a] - x0[a], x1[k] - x0[k])]
        lo, hi = float(np.percentile(x0[near], 10)), float(np.percentile(x1[near], 90))
        width = hi - lo
        if len(near) < 4 or width < 8 * h[a]:
            continue
        lack = (hi - x1[a]) if ltr else (x0[a] - lo)  # the end of the line falls short of the column's far edge
        head = (x0[a] - lo) if ltr else (hi - x1[a])  # and it starts at the near edge (or an indent)
        nxt = x1[i] - x0[i]
        if lack > 1.5 * h[a] and head < 3 * h[a] and nxt > 0.6 * width and min(x1[i], hi) - max(x0[i], lo) > 0.6 * width:
            out[i] = max(out[i], 0.6)
    return out


@torch.no_grad()
def _run(model, vocab, boxes, texts, page_w, page_h, ink=None, crops=None) -> tuple[np.ndarray, np.ndarray]:
    """One model on one page: P(label) per line [n, its labels] and P(line starts a block) [n]."""
    n = len(texts)
    if model.config["n_geom"] > N_GEOM:
        ink = np.zeros((n, N_INK), np.float32) if ink is None else ink
    else:
        ink = None
    geom = torch.from_numpy(geometry(np.asarray(boxes, np.float32), texts, page_w, page_h, ink))
    text = torch.from_numpy(np.stack([text_ids(t, vocab) for t in texts])).long()
    probs, starts = [], []
    for a in range(0, n, MAXL):  # a page with more lines than the model ever saw at once is read in pieces
        g, t = geom[a : a + MAXL][None], text[a : a + MAXL][None]
        pics = torch.from_numpy(crops[a : a + MAXL])[None] if crops is not None and model.config.get("visual") else None
        lr, ls = model(g, t, torch.zeros(1, g.shape[1], dtype=torch.bool), pics)
        probs.append(lr[0].softmax(-1).numpy())
        starts.append(torch.sigmoid(ls[0]).numpy())
    return np.concatenate(probs), np.concatenate(starts).astype(np.float64)


def _as_types(p: np.ndarray, labels: list[str]) -> np.ndarray:
    """A model's probabilities over its own labels, summed into the block types [n, len(TYPES)]."""
    m = np.zeros((len(labels), len(TYPES)))
    for k, lab in enumerate(labels):
        m[k, TYPES.index(TO_TYPE.get(lab, "other"))] = 1
    return p @ m


FLOW = {"paragraph", "list_item", "reference", "footnote", "abstract", "pull_quote", "standfirst", "other", "verse"}  # running text


def _one_type_per_block(q: np.ndarray, starts: np.ndarray, marked: list[bool], itemish: np.ndarray) -> list[str]:
    """Lines between two block starts are one block, so in running text they share one type: the one with
    the most probability over all of them. Two exceptions for list items. A block that opens with an item
    mark that a model also reads as an item is that item. A block named an item that opens without a mark
    is something else (its next best type): the namer calls quotations and numbered text items too, and
    the list rule finds the real ones by their marks. Only a block carried on from the page before keeps
    its item without a mark. Blocks with a line outside running text (a heading, a table cell) keep their
    lines' own types."""
    roles = [TYPES[i] for i in q.argmax(1)]
    n = len(roles)
    LI = TYPES.index("list_item")
    cuts = [0] + [i for i in range(1, n) if starts[i] >= 0.5] + [n]
    for a, b in zip(cuts, cuts[1:]):
        if not all(r in FLOW for r in roles[a:b]):
            continue
        tot = q[a:b].sum(0)
        t = TYPES[int(tot.argmax())]
        if marked[a] and itemish[a]:
            t = "list_item"
        elif t == "list_item" and not marked[a] and (a > 0 or starts[0] >= 0.5):
            tot[LI] = -1
            t = TYPES[int(tot.argmax())]
        if t in FLOW:
            roles[a:b] = [t] * (b - a)
    return roles


# what a line the whole document shows to be page furniture may have been called before
_MARGIN_OVER = {"paragraph", "heading", "other", "noise", "list_item", "footnote", "standfirst", "pull_quote", "section_label",
                "figure", "figure_caption", "running_head", "running_foot"}

DEFAULT = "out/structure_v2.pt+out/structure_v3.pt"  # what the upload page and parse_doc.py use


class Typer:
    """`path`: one checkpoint, or two joined by '+': the combined typer (see `tag`).

    Combined, the first model (v2) cuts the page into blocks and both name the lines: a line's type is the
    best of 0.3 x the first model's probabilities + 0.7 x the last's (v3), over the 27 block types. Within
    each block of running text all lines then take one type (`_one_type_per_block`), and the list rule runs
    last, as for a single model.

    A spec "qwen:URL[?options]+fallback spec" gives the vision-language model's typer instead (structure/qwen.py)."""

    def __new__(cls, path: str = "", lists: bool = True):
        if cls is Typer and isinstance(path, str) and path.startswith("qwen:"):
            from .qwen import QwenTyper
            return QwenTyper(path, lists)
        return super().__new__(cls)

    def __init__(self, path: str, lists: bool = True):
        self.paths = path.split("+")
        self.models = [load(p) for p in self.paths]
        self.model, self.vocab = self.models[-1]
        self.lists = lists  # decide list items by their marks (see list_items)
        self.open_item = False
        self.weights = (0.3, 0.7) if len(self.models) == 2 else [1.0 / len(self.models)] * len(self.models)

    def tag(self, boxes: np.ndarray, texts: list[str], page_w: float, page_h: float, ink: np.ndarray | None = None,
            crops: np.ndarray | None = None, margin: dict[int, str] | None = None):
        """boxes [n, 4] pixels; ink: features.ink_features for the same lines, for a model trained with them;
        crops: features.line_crops for the same lines, for a model that looks at them; margin: lines the
        whole document shows to be a running head or foot ({line: "running_head" | "running_foot"}, see
        ocrl.doc.parse._margin_lines), taken as such unless the models call them a table, title or the like.
        Returns (role per line, P(line starts a block), P(role))."""
        n = len(texts)
        if n == 0:
            self.open_item = False
            return [], np.zeros(0), np.zeros(0)
        outs = [_run(m, v, boxes, texts, page_w, page_h, ink, crops) for m, v in self.models]
        alone = len(outs) == 1
        if alone:
            p, starts = outs[0]
            roles = [self.model.labels[i] for i in p.argmax(1)]
            conf = p.max(1)
        else:
            qs = [_as_types(p, m.labels) for (p, _), (m, _) in zip(outs, self.models)]
            q = sum(w * qk for w, qk in zip(self.weights, qs))
            conf = q.max(1)
            starts = outs[0][1].copy()
            if ENDS:
                latin = sum(c.isascii() and c.isalpha() for t in texts for c in t)
                starts = _paragraph_ends(starts, texts, boxes, ltr=latin > 0.5 * sum(c.isalpha() for t in texts for c in t))
            marked = [bool(_ITEM.match(strip_marks(t))) for t in texts]
            itemish = np.any([qk.argmax(1) == TYPES.index("list_item") for qk in qs], axis=0)
            roles = _one_type_per_block(q, starts, marked, itemish)
        for i, kind in (margin or {}).items():
            if TO_TYPE.get(roles[i], "other") in _MARGIN_OVER:
                roles[i] = kind if not alone else {"running_head": "pageHeader", "running_foot": "pageFooter"}[kind]
                starts[i] = 1.0
                if i + 1 < n:
                    starts[i + 1] = max(starts[i + 1], 0.6)
        if self.lists:  # `open_item` carries an unfinished item over the page break; reset it for a new document
            # combined, one mark is enough when the models also call that line an item
            roles, starts, self.open_item = list_items(roles, starts, texts, boxes, self.open_item, single=not alone)
        return roles, starts, np.asarray(conf)
