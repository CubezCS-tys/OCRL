"""Endless training data: render a page, degrade it, run the real layout stage, crop its clusters."""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..blobs.layout import analyze
from ..render.degrade import degrade
from ..render.page import compose_page
from ..render.shaper import Font
from ..rules.joining import joins
from ..text import DIGITS, LATIN, LETTERS, MARKS, NBSP, PUNCT, TATWEEL, load_corpus
from .crops import cluster_ids, cluster_labels, crop
from .model import encode

# Styles whose ink breaks the print assumptions by design (ruq'ah, archaic kufic).
OUT_OF_SCOPE = {"ArefRuqaa-Regular", "Qahiri-Regular"}
# Never trained on: the test of whether the recogniser learned letters or fonts.
HELD_OUT = {"NotoNaskhArabic", "IBMPlexSansArabic-Regular", "Mirza-Regular"}
SIZES = (26, 30, 34, 38, 44, 50, 58)
WEIGHTS = (0.15, 0.4, 0.65, 0.9)  # points on a variable font's weight axis

TANWIN_FATH, SHADDA = "\u064b", "\u0651"
_VOWELS = "\u064e\u064e\u064e\u0650\u0650\u064f\u064f\u0652\u064b\u064c\u064d"  # weighted towards the short vowels


def font_paths(split: str, root: str = "data/fonts") -> list[str]:
    names = sorted(glob.glob(f"{root}/*.ttf"))
    keep = lambda p: Path(p).stem not in OUT_OF_SCOPE and ((Path(p).stem in HELD_OUT) == (split == "held_out"))
    return [p for p in names if keep(p)]


class TextSource:
    """Corpus sentences, salted with what real pages carry and the corpus lacks.

    Made-up letter strings and numbers, so the model cannot lean on vocabulary;
    vowel marks, which the model must read (they are ink like any other);
    stretched joins (tatweel), which it must read *through*; and Latin words.
    """

    def __init__(self, corpus: list[str], rng: np.random.Generator, plain: bool = False):
        self.corpus, self.rng, self.plain = corpus, rng, plain
        self.vowels = self.stretch = 0.0

    def new_page(self) -> None:
        """Pages differ: most are bare, some lightly vowelled, a few heavily; some are justified."""
        r = self.rng.random()
        self.vowels = 0.0 if r < 0.5 or self.plain else 0.03 if r < 0.85 else 0.14
        self.stretch = 0.0 if self.plain else 0.03 if self.rng.random() < 0.7 else 0.2

    def _latin(self) -> str:
        rng = self.rng
        words = []
        for _ in range(int(rng.integers(1, 4))):
            w = "".join(LATIN[i] for i in rng.integers(0, 26, int(rng.integers(2, 10))))
            r = rng.random()
            words.append(w.capitalize() if r < 0.45 else w.upper() if r < 0.55 else w)
        return NBSP.join(words)

    def _dress(self, word: str) -> str:
        """Add vowel marks and stretches to one Arabic word."""
        rng = self.rng
        if self.plain or not word or any(c not in LETTERS for c in word):
            return word
        out = []
        for i, ch in enumerate(word):
            out.append(ch)
            last = i == len(word) - 1
            if self.vowels and ch != "ا" and rng.random() < self.vowels:
                if rng.random() < 0.12:
                    out.append(SHADDA)
                    if rng.random() < 0.6:
                        out.append(_VOWELS[int(rng.integers(8))])
                else:
                    out.append(_VOWELS[int(rng.integers(8 if not last else len(_VOWELS)))])
            elif not last and self.stretch and joins(ch, word[i + 1]) and rng.random() < self.stretch:
                out.append(TATWEEL * int(rng.integers(1, 5)))
        # The one vowel mark that unvowelled print keeps: tanwin on a final alef (أيضاً).
        if len(word) > 2 and word[-1] == "ا" and rng.random() < 0.15:
            if rng.random() < 0.5:
                out.append(TANWIN_FATH)
            else:
                out.insert(len(out) - 1, TANWIN_FATH)
        return "".join(out)

    def words(self, n: int) -> str:
        rng, out = self.rng, []
        while len(out) < n:
            r = rng.random()
            if r < 0.70 or self.plain:
                out.extend(self.corpus[int(rng.integers(len(self.corpus)))].split())
            elif r < 0.86:  # a made-up word: any letter sequence is shapeable
                k = int(rng.integers(1, 8))
                out.append("".join(LETTERS[i] for i in rng.integers(0, len(LETTERS), k)))
            elif r < 0.93:
                out.append(self._latin())
            else:
                digits = DIGITS[:10] if rng.random() < 0.5 else DIGITS[10:]
                num = "".join(digits[i] for i in rng.integers(0, 10, int(rng.integers(1, 6))))
                out.append(num + (PUNCT[int(rng.integers(len(PUNCT)))] if rng.random() < 0.3 else ""))
        return " ".join(self._dress(w) for w in out[:n])


def latin_paths(root: str = "data/fonts_latin") -> list[str]:
    return sorted(glob.glob(f"{root}/*.ttf"))


def make_page(fonts: list[str], text_src: TextSource, rng: np.random.Generator, cache: dict, lines: int = 6,
              strength: float = 1.0, latin_fonts: list[str] | None = None):
    path = fonts[int(rng.integers(len(fonts)))]
    px = SIZES[int(rng.integers(len(SIZES)))]
    weight = WEIGHTS[int(rng.integers(len(WEIGHTS)))] if strength > 0 else None
    # Real documents set Latin words in a Latin face, not in the Arabic font's own.
    lat = latin_fonts[int(rng.integers(len(latin_fonts)))] if latin_fonts and rng.random() < 0.7 else None
    key = (path, px, weight, lat)
    font = cache.get(key)
    if font is None:
        if len(cache) >= 32:
            cache.pop(next(iter(cache)))
        font = cache[key] = Font(path, px, weight, Font(lat, px) if lat else None)
    text_src.new_page()
    page = compose_page(
        font, text_src.words(lines * 14), width=int(px * rng.uniform(24, 34)), max_lines=lines,
        pitch=rng.uniform(1.6, 2.3), word_space=rng.uniform(0.8, 1.5),
    )
    return degrade(page, rng, strength)


def page_samples(page, rng: np.random.Generator | None = None):
    """(crop, label) for every cluster the layout stage finds on the page."""
    lay = analyze(page.image)
    labels = cluster_labels(page, lay)
    ids = cluster_ids(lay)
    out = []
    for ln in lay.lines:
        for b in ln.bodies:
            jitter = (rng.uniform(0.88, 1.14), rng.uniform(-0.06, 0.06)) if rng is not None else (1.0, 0.0)
            out.append((crop(page.image, lay, ids[b], ln.baseline, ln.scale, jitter), labels[b]))
    return out


def collate(samples):
    widths = torch.tensor([s[0].shape[2] for s in samples])
    x = torch.zeros(len(samples), 2, samples[0][0].shape[1], int(widths.max()), dtype=torch.uint8)
    for i, (img, _) in enumerate(samples):
        x[i, :, :, : img.shape[2]] = torch.from_numpy(img)
    targets = [encode(s[1]) for s in samples]
    lengths = torch.tensor([len(t) for t in targets])
    flat = torch.tensor([i for t in targets for i in t], dtype=torch.long)
    return x, widths, flat, lengths



class ClusterStream(IterableDataset):
    def __init__(self, corpus_path: str, split: str = "train", batch: int = 128, pages: int = 10, seed: int = 0):
        self.corpus_path, self.split, self.batch, self.pages, self.seed = corpus_path, split, batch, pages, seed

    def __iter__(self):
        info = get_worker_info()
        rng = np.random.default_rng([self.seed, info.id if info else 0])
        corpus = load_corpus(self.corpus_path)
        corpus = corpus[: int(len(corpus) * 0.95)]  # the tail is kept for testing
        src, fonts, latin, cache = TextSource(corpus, rng), font_paths(self.split), latin_paths(), {}
        torch.set_num_threads(1)
        while True:
            pool = []
            for _ in range(self.pages):
                pool.extend(page_samples(make_page(fonts, src, rng, cache, latin_fonts=latin), rng))
            # Batches of similar width waste little padding; batch order is shuffled.
            pool.sort(key=lambda s: s[0].shape[2] + rng.uniform(0, 6))
            batches = [pool[i : i + self.batch] for i in range(0, len(pool), self.batch)]
            for i in rng.permutation(len(batches)):
                yield collate(batches[i])
