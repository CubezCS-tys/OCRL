"""Use the structure model: lines of a page in, what each one is and where blocks begin out."""

from __future__ import annotations

import re

import numpy as np
import torch

from .features import ROLES, geometry, text_ids
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
ASIDE = {"running_head", "running_foot", "page_number", "noise"}  # outside the reading order
MAXL = 320

# A list is written into the page: every item opens with a dash, a bullet, or a number or letter with its
# mark, and the next item opens the same way. The model learned its labels from pages where such items
# were more often called paragraphs, so this one is decided by rule.
_D = "0-9\u0660-\u0669\u06f0-\u06f9"
_ITEM = re.compile(rf"^\s*(?:[-–—−ـ•*·▪■●◦]+\s*\S|\(\s?[{_D}]{{1,3}}\s?\)\s*\S|[{_D}]{{1,3}}\s?[-–—ـ.)]\s*[^{_D}\s]|[أ-ي]\s?[-–ـ)]\s+\S)")


_STOP = re.compile(r"[.؟?!:]\s*$")


def list_items(roles: list[str], starts: np.ndarray, texts: list[str], boxes: np.ndarray, open_item: bool = False) -> tuple[list[str], np.ndarray, bool]:
    """Where two or more blocks on a page open with an item mark, they are list items, each mark opening a block.

    A mark counts only where a block could begin: the model thinks one does, or the line before stopped
    short, or there is a gap above. How far the item then runs is read off the page, not asked of the
    model: line after line until the next mark, a blank gap, or a line that stops short of the column's
    far edge (Arabic lines run right to left, so a paragraph's last line ends short on the left). An item
    cut by the foot of a column goes on at the head of the next one, and `open_item` says the page before
    ended inside an item. Returns the roles, the block starts, and whether this page ends inside an item."""
    n = len(texts)
    flow = ("paragraph", "list-item", "otherArticle")
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
    if sum(marked) + open_item < 2:
        return roles, starts, False
    roles, starts = list(roles), starts.copy()

    def run(j: int, hh: float) -> tuple[int, bool]:
        """Carry an item on from line j. Returns the first line past it, and whether it was cut off unfinished."""
        while j < n and not marked[j] and roles[j] in flow:
            if j > 0 and y0[j] < y0[j - 1] - hh:  # head of the next column: the item goes on only if it was not finished
                if short(j - 1):
                    return j, False
            elif j > 0 and (y0[j] - y1[j - 1] > hh or short(j - 1)):
                return j, False
            roles[j] = "list-item"; starts[j] = 0.0
            j += 1
        return j, j == n and n > 0 and not short(n - 1)

    still_open = False
    if open_item and n and not marked[0] and roles[0] in flow:
        j, still_open = run(0, h[0])
        if 0 < j < n and not marked[j]:
            starts[j] = max(starts[j], 0.6)
    for i in range(n):
        if not marked[i]:
            continue
        starts[i] = 1.0
        roles[i] = "list-item"
        j, cut = run(i + 1, h[i]) if i + 1 < n else (n, not short(i))
        still_open = cut if j == n else False
        if j < n and not marked[j]:
            starts[j] = max(starts[j], 0.6)  # whatever follows an item begins its own block
    return roles, starts, still_open


class Typer:
    def __init__(self, path: str, lists: bool = True):
        self.model, self.vocab = load(path)
        self.lists = lists  # decide list items by their marks (see list_items)
        self.open_item = False

    @torch.no_grad()
    def tag(self, boxes: np.ndarray, texts: list[str], page_w: float, page_h: float):
        """boxes [n, 4] pixels. Returns (role per line, P(line starts a block), P(role))."""
        n = len(texts)
        if n == 0:
            self.open_item = False
            return [], np.zeros(0), np.zeros(0)
        geom = torch.from_numpy(geometry(np.asarray(boxes, np.float32), texts, page_w, page_h))
        text = torch.from_numpy(np.stack([text_ids(t, self.vocab) for t in texts])).long()
        roles, starts, conf = [], [], []
        for a in range(0, n, MAXL):  # a page with more lines than the model ever saw at once is read in pieces
            g, t = geom[a : a + MAXL][None], text[a : a + MAXL][None]
            lr, ls = self.model(g, t, torch.zeros(1, g.shape[1], dtype=torch.bool))
            p = lr[0].softmax(-1)
            roles += [ROLES[i] for i in p.argmax(-1).tolist()]
            conf += p.max(-1).values.tolist()
            starts += torch.sigmoid(ls[0]).tolist()
        starts = np.array(starts)
        if self.lists:  # `open_item` carries an unfinished item over the page break; reset it for a new document
            roles, starts, self.open_item = list_items(roles, starts, texts, boxes, self.open_item)
        return roles, starts, np.array(conf)
