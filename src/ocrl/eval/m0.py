"""Milestone 0: does the blob map hold, and does the no-ML layout stage recover it?

Rendered pages come with exact ink ownership, so both questions are counted, not judged:
  * ink facts  -- properties of the font itself, checked against the rules: how often
                  two units touch, and whether each unit shows as many blobs as its
                  text says it should (one body plus its dots and marks);
  * layout     -- how many units the pipeline isolates exactly (all of the unit's ink
                  and nothing else in one cluster), and why it misses the others.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from ..blobs.layout import BODY, SAT, Layout
from ..render.page import PageTruth
from ..rules.signature import satellite_range


def blob_units(truth: PageTruth, lay: Layout) -> dict[int, set[int]]:
    """For every blob, the units of text whose ink it holds."""
    m = (lay.labels > 0) & (truth.owner > 0)
    K = len(truth.units) + 1
    pairs = np.unique(lay.labels[m].astype(np.int64) * K + truth.owner[m])
    out: dict[int, set[int]] = defaultdict(set)
    for blob, owner in zip(pairs // K, pairs % K):
        out[int(blob)].add(int(owner) - 1)
    return out


def score_page(truth: PageTruth, lay: Layout, examples: dict | None = None) -> Counter:
    c = Counter()
    bu = blob_units(truth, lay)
    U = truth.units
    c["units"] = len(U)

    def note(kind: str, text: str) -> None:
        c[kind] += 1
        if examples is not None:
            examples.setdefault(kind, Counter())[text] += 1

    unit_blobs: dict[int, set[int]] = defaultdict(set)
    for blob, us in bu.items():
        for u in us:
            unit_blobs[u].add(blob)
    touching = set()
    for us in bu.values():
        if len(us) > 1:
            touching |= us
            note("ink_touch_event", "+".join(U[u].text for u in sorted(us)))
    cluster_units: dict[int, set[int]] = defaultdict(set)
    for blob, us in bu.items():
        cluster_units[int(lay.host[blob])] |= us

    main = {}  # the largest blob of each unit: its body
    for u, blobs in unit_blobs.items():
        main[u] = max(blobs, key=lambda b: lay.area[b])

    for u in range(len(U)):
        blobs = unit_blobs.get(u)
        text = U[u].text
        if not blobs:
            note("no_ink", text)
            continue
        # --- ink facts: does the font do what the rules say? ---
        if u in touching:
            c["ink_touching"] += 1
        else:
            lo, hi = satellite_range(text)
            n = len(blobs) - 1
            if n < lo:
                note("ink_fewer_blobs_than_rules", text)
            elif n > hi:
                note("ink_more_blobs_than_rules", text)
            else:
                c["ink_as_rules_predict"] += 1
        # --- layout: did the pipeline isolate it? ---
        hosts = {int(lay.host[b]) for b in blobs}
        if len(hosts) == 1 and cluster_units[next(iter(hosts))] == {u}:
            c["exact"] += 1
        elif u in touching:
            c["miss_touching"] += 1
        elif lay.role[main[u]] != BODY:
            note("miss_body_seen_as_satellite", text)
        elif any(lay.role[b] == BODY for b in blobs if b != main[u]):
            note("miss_piece_seen_as_body", text)
        elif len(hosts) > 1:
            lost = {b for b in blobs if lay.host[b] != main[u]}
            to = {U[x].text for b in lost for x in cluster_units[int(lay.host[b])] if x != u}
            note("miss_lost_a_satellite", f"{text}->{'+'.join(sorted(to))}")
        else:
            c["miss_gained_foreign_ink"] += 1

    c["lines"] = len(truth.lines)
    c["lines_found"] = len(lay.lines)
    if len(lay.lines) == len(truth.lines):
        for li, ln in enumerate(lay.lines):
            ranks = []
            for b in ln.bodies:
                us = bu.get(b, set())
                if len(us) == 1 and main[(u := next(iter(us)))] == b:
                    c["bodies_checked"] += 1
                    c["bodies_right_line"] += U[u].line == li
                    ranks.append(U[u].rank)
            c["lines_in_order"] += all(a < b for a, b in zip(ranks, ranks[1:]))
            c["order_inversions"] += sum(a > b for a, b in zip(ranks, ranks[1:]))
            expected = sum(1 for u in U if u.line == li)
            c["lines_body_count_matches_rules"] += len(ln.bodies) == expected
    return c
