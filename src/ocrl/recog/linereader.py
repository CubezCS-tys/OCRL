"""The owner's own line reader (inkscript experiment 89), rebuilt from its checkpoint.

The checkpoint holds the weights, the character list and the settings, but not the code. The layer
shapes fix most of the network: four convolution stages with two residual blocks, a (3, 1) convolution
that folds a 48-pixel-high line down to one row, a two-layer bidirectional LSTM, and a CTC head. What
the shapes do not fix (how the width is pooled, ink polarity, where the blank sits, text direction) was
settled by reading lines whose text is known: see scripts/fit_linereader.py.
"""

from __future__ import annotations

import re
import unicodedata

import cv2
import numpy as np
import torch
import torch.nn as nn

# The reader sees the line mirrored, so that Arabic runs left to right for it. Whatever the page prints
# left to right (Latin words, numbers) then comes out back to front, and has to be turned round again.
# The OLD hand rule for that (to_logical below; the owner's v7 now uses reader_to_logical further down):
#   - a line with no Arabic letters at all is left-to-right as a whole: reverse all of it;
#   - in an Arabic line, a stretch between Arabic letters that holds Latin letters is one left-to-right
#     phrase, from its first letter or digit to its last (brackets and spaces inside go with it);
#   - a stretch with numbers only: each number is turned round on its own, their order stays Arabic.
# Applying it twice gives the text back, so the same function makes the training labels.
_RTL = "\u0590-\u05ff\u0621-\u064a\u066e-\u06d3\u06fa-\u06ff\ufb50-\ufdff\ufe70-\ufefc"
_HAS_RTL = re.compile(f"[{_RTL}]")
_SPAN = re.compile(f"[^{_RTL}]+")
_LAT = "A-Za-z\u00c0-\u024f\u0370-\u03ff\u0400-\u04ff"
_DIG = "0-9\u0660-\u0669\u06f0-\u06f9"
_PHRASE = re.compile(f"[{_LAT}{_DIG}].*[{_LAT}{_DIG}]|[{_LAT}{_DIG}]", re.S)
_HAS_LAT = re.compile(f"[{_LAT}]")
_NUMBER = re.compile(f"[{_DIG}](?:[{_DIG}.,:/%\\-]*[{_DIG}])?")


def _span(m: re.Match) -> str:
    t = m.group()
    if _HAS_LAT.search(t):
        return _PHRASE.sub(lambda x: x.group()[::-1], t, count=1)
    return _NUMBER.sub(lambda x: x.group()[::-1], t)


def to_logical(text: str) -> str:
    """The old hand rule (see above). Kept under this name, unchanged, because our own readers
    (out/line_reader_ocrl.pt, out/line_reader_v2*.pt) were trained on labels made with it, and
    scripts/train_line_reader.py still makes labels and decodes with it. New code that turns a reader's
    output into reading order should go through `LineReader.to_reading_order`, not call this.
    Measured: on 24,061 Azure lines mixing Arabic with digits or Latin, laid out by the bidi algorithm,
    it restores 91.8% exactly; dates and ranges 66%, Latin with digits 34%."""
    if not _HAS_RTL.search(text):
        return text[::-1]
    return _SPAN.sub(_span, text)


old_rule_to_logical = to_logical


# ---------------------------------------------------------------------------------------------------
# Reading order by the Unicode bidirectional algorithm (UAX #9), run backwards for one line.
#
# A page is typeset from text in reading order (logical order) by the bidi algorithm: each character
# gets a level (1 for right-to-left, 2 for a left-to-right stretch inside a right-to-left line), the
# left-to-right stretches are laid out left to right, and brackets in right-to-left stretches are drawn
# mirrored. A reader that reads the mirrored line image from start to end therefore gives the logical
# text with every level-2 stretch turned round, and the brackets in those stretches mirrored (the image
# flip mirrors every glyph; the typesetter had already mirrored the right-to-left ones, so those come
# back the right way round). `bidi_levels` is the algorithm for one line with no embedding controls;
# `logical_to_reader` is what a perfect reader would emit; `reader_to_logical` undoes it. The last is
# not a plain formula, because the layout is not one-to-one ("Smith 2006" and "2006 Smith" inside Arabic
# look the same on paper): it searches for a logical text that lays out to exactly what was read,
# trying the likelier groupings first, and checks each by running the forward algorithm.
# Which separators keep a number in one piece is the algorithm's: e.g. after an Arabic letter European
# digits count as Arabic numbers (rule W2), so "2018-02-23" in an Arabic sentence is three runs and is
# printed "23-02-2018", while "2018/02/23" stays one run; "45%" in Arabic text is printed "%45".
# The character classes come from Python's unicodedata (Unicode 15).

_PAIRS = {"(": ")", "[": "]", "{": "}", "《": "》", "〈": "〉", "⟨": "⟩"}
_CLOSE = {v: k for k, v in _PAIRS.items()}
_MIRROR = {**_PAIRS, **_CLOSE, "<": ">", ">": "<", "«": "»", "»": "«", "‹": "›",
           "›": "‹", "≤": "≥", "≥": "≤", "﴾": "﴿", "﴿": "﴾"}
# U+FD3E/FD3F (ornate parentheses) are not mirrored by the standard, but they are drawn as a pair
# around Quran text and the reader sees them flipped like any bracket; the pair is kept so they swap
# back with the others in a left-to-right stretch (where they practically never occur).
_STRONG_R = {"R", "AL"}


def _mirror(c: str) -> str:
    return _MIRROR.get(c, c)


def bidi_levels(text: str, base: int = 1) -> list[int]:
    """Embedding level of each character of one line in reading order (UAX #9 rules W1-W7, N0-N2,
    I1-I2, L1), paragraph level `base` (1 = right to left). Explicit embedding controls are not
    handled: they are not in any reader's character list."""
    n = len(text)
    if not n:
        return []
    orig = [unicodedata.bidirectional(c) or "L" for c in text]
    t = ["ON" if c == "BN" else c for c in orig]  # BN never comes from a reader; treat as neutral
    e = "L" if base % 2 == 0 else "R"
    sos = eos = e
    # W1: a combining mark takes the type of what it sits on
    prev = sos
    for i in range(n):
        if t[i] == "NSM":
            t[i] = prev
        prev = t[i]
    # W2: European digits after an Arabic letter are Arabic numbers. W3: AL is R.
    last = sos
    for i in range(n):
        if t[i] in ("L", "R", "AL"):
            last = t[i]
        elif t[i] == "EN" and last == "AL":
            t[i] = "AN"
    t = ["R" if x == "AL" else x for x in t]
    # W4: one separator between two numbers of the same kind joins them
    for i in range(1, n - 1):
        if t[i] == "ES" and t[i - 1] == "EN" and t[i + 1] == "EN":
            t[i] = "EN"
        elif t[i] == "CS" and t[i - 1] == t[i + 1] and t[i - 1] in ("EN", "AN"):
            t[i] = t[i - 1]
    # W5: terminators (%, $, degree...) next to European digits join them
    i = 0
    while i < n:
        if t[i] == "ET":
            j = i
            while j < n and t[j] == "ET":
                j += 1
            if (i > 0 and t[i - 1] == "EN") or (j < n and t[j] == "EN"):
                t[i:j] = ["EN"] * (j - i)
            i = j
        else:
            i += 1
    # W6: other separators and terminators are neutral
    t = ["ON" if x in ("ES", "ET", "CS") else x for x in t]
    # W7: European digits after a Latin letter are Latin
    last = sos
    for i in range(n):
        if t[i] in ("L", "R"):
            last = t[i]
        elif t[i] == "EN" and last == "L":
            t[i] = "L"
    # N0: bracket pairs take the direction of what they hold (BD16 pairing, at most 63 open)
    strong = lambda x: "L" if x == "L" else "R" if x in ("R", "EN", "AN") else None
    stack, pairs = [], []
    for i, c in enumerate(text):
        if t[i] != "ON":
            continue
        if c in _PAIRS:
            if len(stack) == 63:
                break
            stack.append((c, i))
        elif c in _CLOSE:
            for k in range(len(stack) - 1, -1, -1):
                if stack[k][0] == _CLOSE[c]:
                    pairs.append((stack[k][1], i))
                    del stack[k:]
                    break
    for o, cl in sorted(pairs):
        inside = {strong(t[k]) for k in range(o + 1, cl)} - {None}
        if e in inside:
            d = e
        elif inside:
            ctx = sos
            for k in range(o - 1, -1, -1):
                if strong(t[k]):
                    ctx = strong(t[k])
                    break
            d = ctx if ctx != e else e
        else:
            continue
        t[o] = t[cl] = d
        for k in (o, cl):  # marks on a bracket follow it
            j = k + 1
            while j < n and orig[j] == "NSM":
                t[j] = d
                j += 1
    # N1/N2: a stretch of neutrals between two of the same direction takes it, otherwise the line's
    i = 0
    while i < n:
        if t[i] in ("ON", "WS", "S", "B"):
            j = i
            while j < n and t[j] in ("ON", "WS", "S", "B"):
                j += 1
            a = strong(t[i - 1]) if i > 0 else sos
            b = strong(t[j]) if j < n else eos
            t[i:j] = [a if a == b else e] * (j - i)
            i = j
        else:
            i += 1
    # I1/I2
    lv = []
    for x in t:
        if base % 2 == 0:
            lv.append(base + (1 if x == "R" else 2 if x in ("EN", "AN") else 0))
        else:
            lv.append(base + (1 if x in ("L", "EN", "AN") else 0))
    # L1: segment separators and trailing white space go back to the line's level
    tail = True
    for i in range(n - 1, -1, -1):
        if orig[i] in ("S", "B"):
            lv[i] = base
            tail = True
        elif orig[i] in ("WS", "BN") and tail:
            lv[i] = base
        else:
            tail = False
    return lv


def _visual(text: str, lv: list[int]) -> list[int]:
    """Rule L2: the positions of `text` in left-to-right display order."""
    idx = list(range(len(text)))
    if not lv:
        return idx
    lo = min(x for x in lv if x % 2) if any(x % 2 for x in lv) else max(lv) + 1
    for level in range(max(lv), lo - 1, -1):
        i = 0
        while i < len(idx):
            if lv[idx[i]] >= level:
                j = i
                while j < len(idx) and lv[idx[j]] >= level:
                    j += 1
                idx[i:j] = idx[i:j][::-1]
                i = j
            else:
                i += 1
    return idx


def logical_to_reader(text: str, base: int = 1, mirror: bool = True) -> str:
    """What a perfect reader of the mirrored line image emits for a line typeset from `text`: the
    characters from the right edge of the line to the left, each as its mirrored glyph reads (so
    brackets in left-to-right stretches come out the wrong way round, those in right-to-left ones not)."""
    lv = bidi_levels(text, base)
    order = _visual(text, lv)[::-1]
    return "".join(_mirror(text[i]) if mirror and lv[i] % 2 == 0 else text[i] for i in order)


def _line_base(text: str) -> int:
    """A line with no Arabic or Hebrew letter at all is taken as left to right (an English abstract, a
    reference, and also a line of numbers and signs only, such as a table cell: on 463 such real lines
    read by v7, turning the whole line round differed from Azure on 10% of characters, laying it out
    right to left on 21%)."""
    return 1 if any(unicodedata.bidirectional(c) in _STRONG_R for c in text) else 0


def _runs(lv: list[int], base: int) -> list[tuple[int, int]]:
    out, i = [], 0
    while i < len(lv):
        if lv[i] > base:
            j = i
            while j < len(lv) and lv[j] > base:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _flip(text: str, runs, mirror: bool) -> str:
    s = list(text)
    for a, b in runs:
        s[a:b] = [_mirror(c) if mirror else c for c in reversed(text[a:b])]
    return "".join(s)


def _levels_in_paragraph(text: str) -> list[int]:
    """Levels of a line that sits inside a right-to-left paragraph whose text before it ends in an
    Arabic letter: European digits at the start of the line then count as Arabic numbers (rule W2), as
    the typesetter saw them. Checked on real pages: "19.6%" opening a line is printed "%19.6", as in
    the middle of a line, not "19.6%" as it would be if the line stood alone."""
    return bidi_levels("\u0627" + text, 1)[1:]


_RANGE = re.compile(r"(?<![0-9.,/:\-\u2013])([0-9]+)([-\u2013])([0-9]+)(?![.,/:\-\u2013]?[0-9])")


_DATE = re.compile(r"(?<![0-9.,/:\-\u2013])([0-9]{3,4})([-\u2013])([0-9]{1,2})([-\u2013])([0-9]{1,2})(?![.,/:\-\u2013]?[0-9])")


def _low_high(m: re.Match) -> str:
    a, b = m.group(1), m.group(3)
    return f"{b}{m.group(2)}{a}" if int(a) > int(b) else m.group()


def _year_last(m: re.Match) -> str:
    return f"{m.group(5)}{m.group(4)}{m.group(3)}{m.group(2)}{m.group(1)}"


def reader_to_logical(text: str, mirror: bool = False, prefer: str = "runs", in_paragraph: bool = True,
                      ranges_low_high: bool = True) -> str:
    """Inverse of `logical_to_reader` for one line: the reading-order text whose layout is what the
    reader emitted. Left-to-right lines are turned round whole. In a right-to-left line each stretch
    between right-to-left letters is solved on its own: a few likely groupings of its left-to-right
    pieces are tried, each improved until it is consistent (the runs the algorithm finds in the
    candidate are the ones that were turned round to make it), and the first consistent one is kept.
    With prefer="phrase" the first grouping tried is the whole stretch as one run (a Latin phrase with
    its numbers reads "Smith, 2006"); with prefer="runs" it is the runs the algorithm finds in the
    reader's text itself, which keeps each left-to-right piece separate ("2006 ,Smith"; this is the
    order Azure gives most often). Both lay out identically on paper. Numbers and dates between Arabic
    words come out the same either way. If no grouping is consistent, the reader's own runs are used.

    The defaults are what the owner's v7 reader needs, measured on 1,710 real lines (Azure boxes on 110
    pages) read by v7 and compared with Azure: prefer="runs" (2.09% of characters differ from Azure,
    "phrase" 2.37%, the old rule 2.41%); mirror=False, because v7 writes brackets in a left-to-right
    stretch as they stand in reading order, not as their flipped image (mirroring made Latin lines go
    from 1.06% to 2.11%).
    ranges_low_high: a range of two European numbers ("10-15") is printed one of two ways in practice:
    split in two runs as the standard says ("15-10" on paper), or kept as one left-to-right run, as Word
    often does. The reader's text cannot tell them apart, so such a range is written low to high,
    which is how ranges are written (on the real lines: dates and ranges 4.02% -> 3.88%). For the same
    reason a hyphenated date of European digits is written day first, year last ("23-02-2018", the usual
    order in Arabic text; a group of three or four digits taken for the year, so a misread
    "23-02-208" is caught too: 5 of the 7 such dates in Azure's 24,061 mixed lines; and on 2258-004-004-007
    page 1 the reader's text shows the page kept "23-02-2018" as one left-to-right run, Word-style).
    Slashed dates are one run by the standard too and are left as read; Arabic-Indic digits keep the
    standard reading."""
    base = _line_base(text)
    if base == 0:
        return "".join(_mirror(c) if mirror else c for c in reversed(text))
    levels = _levels_in_paragraph if in_paragraph else (lambda t: bidi_levels(t, 1))
    cls = [unicodedata.bidirectional(c) for c in text]
    n = len(text)
    # stretches between right-to-left letters
    spans, i = [], 0
    while i < n:
        if cls[i] in _STRONG_R or cls[i] == "NSM":
            i += 1
            continue
        j = i
        while j < n and cls[j] not in _STRONG_R:
            j += 1
        core = [k for k in range(i, j) if cls[k] in ("L", "EN", "AN")]
        if core:
            spans.append((i, j, core[0], core[-1] + 1))
        i = j
    if not spans:
        return text
    own = _runs(levels(text), 1)
    choice = {s: [r for r in own if s[0] <= r[0] and r[1] <= s[1]] for s in spans}
    for s in spans:
        a, b, ca, cb = s
        wide_a, wide_b = ca, cb  # the phrase with the signs and brackets that stick to it
        while wide_a > a and not text[wide_a - 1].isspace():
            wide_a -= 1
        while wide_b < b and not text[wide_b].isspace():
            wide_b += 1
        phrase = [[(wide_a, wide_b)], [(ca, cb)]]
        seeds = phrase + [choice[s]] if prefer == "phrase" else [choice[s]] + phrase
        found = None
        for seed in seeds:
            cur = seed
            for _ in range(4):
                choice[s] = cur
                cand = _flip(text, [r for v in choice.values() for r in v], mirror)
                got = [r for r in _runs(levels(cand), 1) if a <= r[0] and r[1] <= b]
                if got == cur:
                    found = cur
                    break
                cur = got
            if found is not None:
                break
        choice[s] = found if found is not None else [r for r in own if a <= r[0] and r[1] <= b]
    out = _flip(text, [r for v in choice.values() for r in v], mirror)
    return _DATE.sub(_year_last, _RANGE.sub(_low_high, out)) if ranges_low_high else out


class _Res(nn.Module):
    def __init__(self, c: int, late_relu: bool):
        super().__init__()
        self.a = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c))
        self.b = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c))
        self.late_relu = late_relu

    def forward(self, x):
        y = self.b(torch.relu(self.a(x)))
        return torch.relu(x + y) if self.late_relu else x + torch.relu(y)


def _conv(cin, cout, k=(3, 3)):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=(0, 0) if k == (3, 1) else 1, bias=False), nn.BatchNorm2d(cout))


class LineNet(nn.Module):
    def __init__(self, n_out: int, hid: int = 384, feat: int = 384, base: int = 32, wpool=(2, 2, 1, 1), late_relu: bool = True):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        P = lambda w: nn.MaxPool2d((2, w))
        self.cnn = nn.Sequential(
            _conv(1, c[0]), P(wpool[0]), _conv(c[0], c[1]), P(wpool[1]), _conv(c[1], c[2]), _Res(c[2], late_relu), P(wpool[2]),
            _conv(c[2], c[3]), _Res(c[3], late_relu), P(wpool[3]), _conv(c[3], feat, (3, 1)))
        self.rnn = nn.LSTM(feat, hid, num_layers=2, bidirectional=True)
        self.head = nn.Linear(2 * hid, n_out)

    def forward(self, x):  # [B, 1, 48, W]
        f = x
        for m in self.cnn:
            f = m(f)
            if isinstance(m, nn.Sequential):
                f = torch.relu(f)
        f = f.squeeze(2).permute(2, 0, 1)  # [T, B, feat]
        f, _ = self.rnn(f)
        return self.head(f).log_softmax(-1)


class LineReader:
    """Reads one line image (grey, dark ink on light paper) into text."""

    def __init__(self, path: str, wpool=(2, 2, 1, 1), invert: bool = True, blank_first: bool = True, late_relu: bool = True,
                 reverse: bool = False, mirror: bool = True, device: str | None = None, order: str | None = None):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = ck["meta"]
        self.chars, self.H = m["chars"], m["H"]
        self.invert, self.blank_first, self.reverse, self.mirror = invert, blank_first, reverse, mirror
        # How the reader's output is put into reading order depends on how its training labels were made:
        #   "bidi": the reader emits what is on the paper from right to left (the owner's v7 and the other
        #           inkscript readers were trained that way); undo the page's bidi layout (reader_to_logical).
        #   "old":  our own readers (meta "made_by" OCRL scripts/train_line_reader.py) learned labels made by
        #           the old hand rule `to_logical`, which is its own inverse, so the same rule decodes them.
        #           Changing their decoding would need retraining with labels made by logical_to_reader.
        #   "none": raw output, for experiments.
        # A checkpoint may name its rule in meta["order"]; otherwise it is told by who made it.
        self.order = order or m.get("order") or ("old" if "OCRL scripts/train_line_reader.py" in str(m.get("made_by", "")) else "bidi")
        if self.order not in ("bidi", "old", "none"):
            raise ValueError(f"unknown reading-order rule {self.order!r}")
        base = ck["model"]["cnn.0.0.weight"].shape[0]
        self.net = LineNet(len(self.chars) + 1, m["hid"], m["feat"], base, wpool, late_relu)
        self.net.load_state_dict(ck["model"])
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.dev).eval()

    def prepare(self, gray: np.ndarray) -> np.ndarray:
        w = max(int(round(gray.shape[1] * self.H / gray.shape[0])), 8)
        g = cv2.resize(gray, (w, self.H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        if self.mirror:
            g = g[:, ::-1].copy()
        return 1 - g if self.invert else g

    @torch.no_grad()
    def read(self, lines: list[np.ndarray]) -> list[str]:
        out = [""] * len(lines)
        prep = [self.prepare(g) for g in lines]
        order = np.argsort([p.shape[1] for p in prep])
        for i in range(0, len(order), 64):
            idx = order[i : i + 64]
            W = max(prep[j].shape[1] for j in idx)
            x = torch.full((len(idx), 1, self.H, W), 0.0 if self.invert else 1.0)
            for k, j in enumerate(idx):
                x[k, 0, :, : prep[j].shape[1]] = torch.from_numpy(prep[j])
            ids = self.net(x.to(self.dev)).argmax(-1).cpu().numpy()  # [T, B]
            for k, j in enumerate(idx):
                seq = ids[:, k]
                keep = seq[np.insert(seq[1:] != seq[:-1], 0, True)]
                blank = 0 if self.blank_first else len(self.chars)
                text = "".join(self.chars[c - 1 if self.blank_first else c] for c in keep if c != blank)
                out[j] = text[::-1] if self.reverse else self.to_reading_order(text) if self.mirror else text
        return out

    def to_reading_order(self, text: str) -> str:
        """The reader's output for one mirrored line, put into reading order by this reader's rule."""
        if self.order == "bidi":
            return reader_to_logical(text)
        if self.order == "old":
            return to_logical(text)
        return text

