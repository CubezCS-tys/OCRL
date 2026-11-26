"""One HTML file showing a parsed document: the scan with every block boxed and numbered, and the text beside it.

Laid out and coloured like the inkscript viewer, so the two can be compared side by side.
"""

from __future__ import annotations

import base64
import html
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

COLORS = {
    "running_head": "#8a7f66", "title": "#b4232a", "author": "#1f7a46", "affiliation": "#2563a8", "abstract": "#7048a8",
    "heading": "#a3341f", "paragraph": "#56606e", "page_number": "#7b8190", "list_item": "#c2590a", "noise": "#9aa0ab",
    "footnote": "#607d8b", "table_caption": "#4c5fd5", "table": "#0b7285", "reference": "#5a6f1e", "other": "#6b6b6b",
}
NAMES = {"running_head": "Running head", "page_number": "Page number", "list_item": "List item", "noise": "Scan marks",
         "table_caption": "Table caption"}

_CSS = """
:root{--bg:#f4f1ea;--panel:#fffdf8;--ink:#1d1b16;--muted:#6f695c;--line:#ddd6c6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 "Segoe UI",system-ui,sans-serif}
header{padding:14px 20px 10px;border-bottom:1px solid var(--line)}
h1{font-size:17px;margin:0 0 4px;font-weight:600}h1 span{font-weight:400;color:var(--muted)}
.by{color:var(--muted);font-size:12.5px;max-width:110ch}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;margin-top:8px;font-size:12.5px;color:var(--muted)}
.legend i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px;vertical-align:-1px}
main{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);height:calc(100vh - 118px)}
.pane{overflow-y:auto;padding:14px 20px}
.cap{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 8px}
.scan{position:relative;margin:0 0 22px;background:#fff;box-shadow:0 1px 4px #0002}
.scan img{display:block;width:100%;height:auto}
.box{position:absolute;border:1.5px solid var(--c);background:color-mix(in srgb,var(--c) 10%,transparent);border-radius:3px;cursor:pointer}
.box.aside{border-style:dashed;background:none}
.box b{position:absolute;top:-9px;right:-9px;min-width:19px;height:19px;padding:0 4px;border-radius:10px;background:var(--c);color:#fff;font:600 11px/19px "Segoe UI",sans-serif;text-align:center}
.box.on,.blk.on{outline:3px solid var(--c);outline-offset:1px}
.pg{font-size:12px;color:var(--muted);margin:0 0 4px}
.doc{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:14px 18px}
.blk{display:grid;grid-template-columns:118px minmax(0,1fr);gap:10px;padding:7px 4px;border-radius:4px;cursor:pointer}
.tag{font-size:12px;color:var(--c);white-space:nowrap}
.tag b{display:inline-block;min-width:18px;height:18px;border-radius:9px;background:var(--c);color:#fff;font:600 10.5px/18px "Segoe UI",sans-serif;text-align:center;margin-right:6px;padding:0 4px}
.txt{direction:rtl;text-align:right;font:500 18px/1.9 "Noto Naskh Arabic","Amiri","Traditional Arabic",serif;overflow-wrap:anywhere}
.blk.title .txt{font-size:25px;text-align:center}.blk.author .txt,.blk.affiliation .txt{text-align:center}
.blk.heading .txt{font-weight:700}.blk.aside .txt{color:var(--muted);font-size:15px}
.blk.table .txt{font-size:14px;color:var(--muted)}
@media(max-width:900px){main{grid-template-columns:1fr;height:auto}.pane{overflow:visible}}
"""

_JS = """
const on=(id,v)=>document.querySelectorAll('[data-b="'+id+'"]').forEach(e=>e.classList.toggle('on',v));
document.querySelectorAll('[data-b]').forEach(e=>{
  e.addEventListener('mouseenter',()=>on(e.dataset.b,true));
  e.addEventListener('mouseleave',()=>on(e.dataset.b,false));
  e.addEventListener('click',()=>{const o=[...document.querySelectorAll('[data-b="'+e.dataset.b+'"]')].find(x=>x!==e&&x.classList.contains(e.classList.contains('box')?'blk':'box'));if(o)o.scrollIntoView({behavior:'smooth',block:'center'});});
});
"""


def _png(gray: np.ndarray, width: int = 900) -> str:
    f = width / gray.shape[1]
    small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", small, [cv2.IMWRITE_PNG_COMPRESSION, 9])[1].tobytes()).decode()


def write_viewer(master: dict, images: list[np.ndarray], path: Path, dpi: int = 300, seconds: float = 0.0) -> None:
    name = lambda t: NAMES.get(t, t.replace("_", " ").capitalize())
    used = [t for t in COLORS if any(b["type"] == t for b in master["blocks"])]
    counts = Counter(b["type"] for b in master["blocks"])
    boxes: dict[int, list[str]] = {}
    for b in master["blocks"]:
        aside = "aside" in b
        for part in b["parts"]:
            pg = next(p for p in master["pages"] if p["number"] == part["page"])
            for x0, y0, x1, y1 in part["regions"]:
                style = (f"left:{100 * x0 / pg['width']:.2f}%;top:{100 * y0 / pg['height']:.2f}%;"
                         f"width:{100 * (x1 - x0) / pg['width']:.2f}%;height:{100 * (y1 - y0) / pg['height']:.2f}%;--c:{COLORS.get(b['type'], '#6b6b6b')}")
                badge = "" if aside else f"<b>{b['order']}</b>"
                boxes.setdefault(part["page"], []).append(
                    f'<div class="box{" aside" if aside else ""}" data-b="{b["id"]}" style="{style}" title="{name(b["type"])}">{badge}</div>')
    scans = "".join(
        f'<p class="pg">scan page {i}</p><div class="scan"><img src="{_png(img)}" alt="Scan of page {i}">{"".join(boxes.get(i, []))}</div>'
        for i, img in enumerate(images, master["pages"][0]["number"]))
    doc = "".join(
        f'<div class="blk {b["type"]}{" aside" if "aside" in b else ""}" data-b="{b["id"]}" style="--c:{COLORS.get(b["type"], "#6b6b6b")}">'
        f'<div class="tag">{"" if "aside" in b else "<b>" + str(b["order"]) + "</b>"}{name(b["type"])}</div>'
        f'<div class="txt" lang="ar">{html.escape(b["text"]) or "&nbsp;"}</div></div>'
        for b in master["blocks"] if b["type"] != "noise")
    legend = "".join(f'<span><i style="background:{COLORS[t]}"></i>{name(t)} · {counts[t]}</span>' for t in used)
    speed = f" Parsed in {seconds:.0f} seconds ({seconds / max(len(images), 1):.1f} a page) on this machine's CPU." if seconds else ""
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{master['id']} · parsed locally</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@500;700&display=swap"><style>{_CSS}</style></head><body>
<header><h1>{master['id']} <span>· whole document, {len(images)} pages · parsed locally</span></h1>
<div class="by">Text: {html.escape(master['made_by']['text'])}. Structure: {html.escape(master['made_by']['structure'])}{speed}</div>
<div class="legend">{legend}</div></header>
<main><div class="pane"><p class="cap">The scan, with every block boxed and numbered in reading order</p>{scans}</div>
<div class="pane"><p class="cap">Parsed document</p><div class="doc">{doc}</div></div></main>
<script>{_JS}</script></body></html>"""
    Path(path).write_text(page, encoding="utf-8")
