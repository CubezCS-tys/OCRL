"""Use the structure model: lines of a page in, what each one is and where blocks begin out."""

from __future__ import annotations

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


class Typer:
    def __init__(self, path: str):
        self.model, self.vocab = load(path)

    @torch.no_grad()
    def tag(self, boxes: np.ndarray, texts: list[str], page_w: float, page_h: float):
        """boxes [n, 4] pixels. Returns (role per line, P(line starts a block), P(role))."""
        n = len(texts)
        if n == 0:
            return [], np.zeros(0), np.zeros(0)
        n_chars = np.array([len(strip_marks(t).strip()) for t in texts], np.float32)
        geom = torch.from_numpy(geometry(np.asarray(boxes, np.float32), n_chars, page_w, page_h))
        text = torch.from_numpy(np.stack([text_ids(t, self.vocab) for t in texts])).long()
        roles, starts, conf = [], [], []
        for a in range(0, n, MAXL):  # a page with more lines than the model ever saw at once is read in pieces
            g, t = geom[a : a + MAXL][None], text[a : a + MAXL][None]
            lr, ls = self.model(g, t, torch.zeros(1, g.shape[1], dtype=torch.bool))
            p = lr[0].softmax(-1)
            roles += [ROLES[i] for i in p.argmax(-1).tolist()]
            conf += p.max(-1).values.tolist()
            starts += torch.sigmoid(ls[0]).tolist()
        return roles, np.array(starts), np.array(conf)
