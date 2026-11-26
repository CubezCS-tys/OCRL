"""The structure model: reads all the lines of a page at once and says what each one is.

Each line becomes one token built from its geometry and its first and last characters. A small
transformer lets every line see every other line on the page, because what a line is depends on
its neighbours: a short line is a heading only if a paragraph follows it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .features import FIRST, N_GEOM, ROLES


class LineTagger(nn.Module):
    def __init__(self, n_vocab: int, d: int = 192, layers: int = 4, heads: int = 4, n_roles: int = len(ROLES)):
        super().__init__()
        self.config = {"n_vocab": n_vocab, "d": d, "layers": layers, "heads": heads, "n_roles": n_roles}
        self.char = nn.Embedding(n_vocab + 2, 32, padding_idx=0)
        self.conv = nn.Conv1d(32, 96, 3, padding=1)
        self.geom = nn.Sequential(nn.Linear(N_GEOM, 96), nn.ReLU(), nn.Linear(96, 96))
        self.inp = nn.Linear(96 * 3, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.role = nn.Linear(d, n_roles)
        self.start = nn.Linear(d, 1)

    def forward(self, geom: torch.Tensor, text: torch.Tensor, pad: torch.Tensor):
        """geom [B, L, N_GEOM], text [B, L, chars], pad [B, L] True where there is no line."""
        B, L, C = text.shape
        e = self.char(text.long()).view(B * L, C, -1).transpose(1, 2)
        c = torch.relu(self.conv(e))
        first, last = c[:, :, :FIRST].amax(2), c[:, :, FIRST:].amax(2)
        x = self.inp(torch.cat([first.view(B, L, -1), last.view(B, L, -1), self.geom(geom)], -1))
        x = self.enc(x, src_key_padding_mask=pad)
        return self.role(x), self.start(x).squeeze(-1)


def save(model: LineTagger, vocab: dict, path: str) -> None:
    torch.save({"model": model.state_dict(), "vocab": vocab, "roles": ROLES, **model.config}, path)


def load(path: str) -> tuple[LineTagger, dict]:
    ck = torch.load(path, map_location="cpu")
    m = LineTagger(ck["n_vocab"], ck["d"], ck["layers"], ck["heads"], ck["n_roles"])
    m.load_state_dict(ck["model"])
    return m.eval(), ck["vocab"]
