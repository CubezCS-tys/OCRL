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
    def __init__(self, n_vocab: int, d: int = 192, layers: int = 4, heads: int = 4, n_roles: int = len(ROLES), n_geom: int = N_GEOM,
                 visual: bool = False):
        super().__init__()
        self.config = {"n_vocab": n_vocab, "d": d, "layers": layers, "heads": heads, "n_roles": n_roles, "n_geom": n_geom, "visual": visual}
        self.char = nn.Embedding(n_vocab + 2, 32, padding_idx=0)
        self.conv = nn.Conv1d(32, 96, 3, padding=1)
        self.geom = nn.Sequential(nn.Linear(n_geom, 96), nn.ReLU(), nn.Linear(96, 96))
        # With `visual`, each line also brings a small picture of itself (features.line_crops): a few
        # convolutions turn it into 128 numbers, added to what the box and the text already say.
        self.eye = None
        if visual:
            c = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU())
            self.eye = nn.Sequential(c(1, 24), nn.MaxPool2d(2), c(24, 48), nn.MaxPool2d(2), c(48, 96), nn.MaxPool2d(2), c(96, 128), nn.AdaptiveMaxPool2d((1, 8)))
            self.eye_out = nn.Linear(128 * 8, 128)
        self.inp = nn.Linear(96 * 3 + (128 if visual else 0), d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.role = nn.Linear(d, n_roles)
        self.start = nn.Linear(d, 1)

    def forward(self, geom: torch.Tensor, text: torch.Tensor, pad: torch.Tensor, crops: torch.Tensor | None = None):
        """geom [B, L, N_GEOM], text [B, L, chars], pad [B, L] True where there is no line,
        crops [B, L, H, W] uint8 or float pictures of the lines (only for a model made with `visual`)."""
        B, L, C = text.shape
        e = self.char(text.long()).view(B * L, C, -1).transpose(1, 2)
        c = torch.relu(self.conv(e))
        first, last = c[:, :, :FIRST].amax(2), c[:, :, FIRST:].amax(2)
        parts = [first.view(B, L, -1), last.view(B, L, -1), self.geom(geom)]
        if self.eye is not None:
            if crops is None:
                parts.append(torch.zeros(B, L, 128, device=geom.device))
            else:
                real = ~pad.reshape(-1)  # only the lines that exist go through the convolutions
                v = torch.zeros(B * L, 128, device=geom.device, dtype=geom.dtype)
                pic = 1.0 - crops.reshape(B * L, 1, *crops.shape[-2:])[real].float() / 255.0  # ink as 1
                v[real] = self.eye_out(self.eye(pic).flatten(1)).to(v.dtype)
                parts.append(v.view(B, L, 128))
        x = self.inp(torch.cat(parts, -1))
        x = self.enc(x, src_key_padding_mask=pad)
        return self.role(x), self.start(x).squeeze(-1)


def save(model: LineTagger, vocab: dict, path: str, labels: list[str] | None = None) -> None:
    """`labels`: the names of the model's outputs, when they are not the Desk roles (a model retrained on block types)."""
    torch.save({"model": model.state_dict(), "vocab": vocab, "roles": labels or ROLES, **model.config}, path)


def load(path: str) -> tuple[LineTagger, dict]:
    ck = torch.load(path, map_location="cpu")
    m = LineTagger(ck["n_vocab"], ck["d"], ck["layers"], ck["heads"], ck["n_roles"], ck.get("n_geom", N_GEOM), ck.get("visual", False))
    m.load_state_dict(ck["model"])
    m.labels = list(ck.get("roles") or ROLES)
    return m.eval(), ck["vocab"]
