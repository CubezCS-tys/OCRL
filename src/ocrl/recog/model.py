"""The recogniser: a small CNN + BiGRU read out with CTC. It reads one cluster at a time."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..text import CHARSET, LEGACY_CHARSET

VOCAB = ["<blank>"] + list(CHARSET)
LEGACY_VOCAB = ["<blank>"] + list(LEGACY_CHARSET)
CHAR_TO_ID = {c: i for i, c in enumerate(VOCAB)}


def encode(text: str) -> list[int]:
    return [CHAR_TO_ID[c] for c in text]


def _block(cin: int, cout: int, pool: tuple[int, int] | None) -> list[nn.Module]:
    layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(pool))
    return layers


class Recognizer(nn.Module):
    """Input [B, 2, 48, W] in 0..1; output log-probs [W/2, B, vocab]."""

    def __init__(self, width: int = 64, hidden: int = 192, n_out: int = len(VOCAB)):
        super().__init__()
        self.config = {"width": width, "hidden": hidden, "n_out": n_out}
        w = width
        self.cnn = nn.Sequential(
            *_block(2, w, (2, 2)),  # 24 x W/2
            *_block(w, 2 * w, (2, 1)),  # 12
            *_block(2 * w, 3 * w, None),
            *_block(3 * w, 3 * w, (2, 1)),  # 6
            *_block(3 * w, 4 * w, (2, 1)),  # 3
        )
        self.proj = nn.Linear(4 * w * 3, hidden)
        self.rnn = nn.GRU(hidden, hidden, num_layers=2, bidirectional=True, dropout=0.1)
        self.out = nn.Linear(2 * hidden, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x)  # [B, C, 3, T]
        f = f.permute(3, 0, 1, 2).flatten(2)  # [T, B, C*3]
        f = torch.relu(self.proj(f))
        f, _ = self.rnn(f)
        return self.out(f).log_softmax(-1)

    @staticmethod
    def frames(width: torch.Tensor) -> torch.Tensor:
        return width // 2


def save(model: Recognizer, path: str, step: int) -> None:
    torch.save({"model": model.state_dict(), "step": step, "vocab": VOCAB, **model.config}, path)


def load(path: str, device: str = "cpu") -> tuple[Recognizer, list[str], int]:
    """Returns (model, its vocabulary, training step). Older checkpoints carry neither size nor vocabulary."""
    ck = torch.load(path, map_location=device)
    vocab = ck.get("vocab", LEGACY_VOCAB)
    model = Recognizer(ck.get("width", 48), ck.get("hidden", 160), len(vocab))
    model.load_state_dict(ck["model"])
    return model.to(device), vocab, ck.get("step", 0)
