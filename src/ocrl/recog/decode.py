"""CTC decoding: greedy, and an n-best beam search for the rules to choose from."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .model import VOCAB


def greedy(logp: np.ndarray, vocab: list[str] = VOCAB) -> str:
    """logp: [T, vocab] log-probabilities for one sample."""
    ids = logp.argmax(-1)
    keep = ids[np.insert(ids[1:] != ids[:-1], 0, True)]
    return "".join(vocab[i] for i in keep if i)


def nbest(logp: np.ndarray, beam: int = 8, top: int = 6, prune: float = -7.0, vocab: list[str] = VOCAB) -> list[tuple[str, float]]:
    """Prefix beam search. Returns up to `top` (text, log-probability), best first."""
    NEG = -math.inf
    beams = {(): (0.0, NEG)}  # prefix -> (log p ending in blank, log p ending in non-blank)
    for t in range(logp.shape[0]):
        row = logp[t]
        cand = np.flatnonzero(row > prune)
        nxt = defaultdict(lambda: [NEG, NEG])
        for prefix, (pb, pnb) in beams.items():
            tot = np.logaddexp(pb, pnb)
            for c in cand:
                p = float(row[c])
                if c == 0:
                    e = nxt[prefix]
                    e[0] = np.logaddexp(e[0], tot + p)
                    continue
                e = nxt[prefix + (int(c),)]
                if prefix and prefix[-1] == c:
                    e[1] = np.logaddexp(e[1], pb + p)  # a repeat needs a blank between
                    s = nxt[prefix]
                    s[1] = np.logaddexp(s[1], pnb + p)
                else:
                    e[1] = np.logaddexp(e[1], tot + p)
        beams = dict(sorted(nxt.items(), key=lambda kv: -np.logaddexp(*kv[1]))[:beam])
    ranked = sorted(((np.logaddexp(*v), k) for k, v in beams.items()), reverse=True)[:top]
    return [("".join(vocab[i] for i in k), float(s)) for s, k in ranked]
