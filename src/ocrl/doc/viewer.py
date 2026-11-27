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
    # the structure model's other types (structure/tag.py TYPES): each gets its own colour and a place in the legend
    "running_foot": "#8a7f66", "section_label": "#9c4f96", "supervisor": "#1f7a46", "degree_statement": "#2563a8",
    "standfirst": "#8b5e34", "keywords": "#7048a8", "figure": "#3f8f8a", "figure_caption": "#4c5fd5", "pull_quote": "#a0567a",
    "equation": "#5b6abf", "verse": "#b0306a", "handwriting": "#7a5c00",
}
NAMES = {"running_head": "Running head", "page_number": "Page number", "list_item": "List item", "noise": "Scan marks",
         "table_caption": "Table caption", "running_foot": "Running foot", "section_label": "Section label",
         "degree_statement": "Degree statement", "figure_caption": "Figure caption", "pull_quote": "Pull quote"}

_CSS = """
:root{--bg:#f4f1ea;--panel:#fffdf8;--ink:#1d1b16;--muted:#6f695c;--line:#ddd6c6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 "Segoe UI",system-ui,sans-serif;display:flex;flex-direction:column;height:100vh}
header{padding:14px 20px 10px;border-bottom:1px solid var(--line)}
h1{font-size:17px;margin:0 0 4px;font-weight:600}h1 span{font-weight:400;color:var(--muted)}
.by{color:var(--muted);font-size:12.5px;max-width:110ch}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;margin-top:8px;font-size:12.5px;color:var(--muted)}
.legend i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px;vertical-align:-1px}
main{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);flex:1;min-height:0}
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
.txt{text-align:start;font:500 18px/1.9 "Noto Naskh Arabic","Amiri","Traditional Arabic",serif;overflow-wrap:anywhere}
.blk.title .txt{font-size:25px;text-align:center}.blk.author .txt,.blk.affiliation .txt{text-align:center}
.blk.heading .txt{font-weight:700}.blk.aside .txt{color:var(--muted);font-size:15px}
.blk.table .txt{font-size:14px;color:var(--muted)}
.lb{position:absolute;display:none;border:2px solid #e8590c;background:#e8590c26;border-radius:2px;pointer-events:none;z-index:2}.lb.on{display:block}
.lb.warned{display:block;border:2px dashed #c92a2a;background:none}.ln.warned{text-decoration:underline wavy #c92a2a;text-underline-offset:4px}
.pwarn{font-size:12px;color:#c92a2a;background:#c92a2a14;border-left:3px solid #c92a2a;padding:4px 8px;margin:0 0 6px}
.ln{border-radius:3px}.blk.verse .ln{display:block}.ln:hover,.ln.on{background:#e8590c2e;box-shadow:0 0 0 2px #e8590c2e}
details.figs{margin:2px 0;padding:6px 4px;border-radius:4px}details.figs summary{cursor:pointer;font-size:12px;color:var(--c)}details.figs summary b{display:inline-block;min-width:18px;height:18px;border-radius:9px;background:var(--c);color:#fff;font:600 10.5px/18px "Segoe UI",sans-serif;text-align:center;margin-right:6px;padding:0 5px}
.cell{position:absolute;border:1px dashed color-mix(in srgb,var(--c) 55%,transparent);pointer-events:none}
.cell.empty{border-style:dotted;border-color:color-mix(in srgb,var(--c) 30%,transparent)}
table.grid{border-collapse:collapse;margin:2px 0;font:500 15px/1.6 "Noto Naskh Arabic","Amiri","Traditional Arabic",serif;color:var(--ink);background:#fff}
table.grid td,table.grid th{border:1px solid #c9c1ae;padding:3px 8px;vertical-align:top;text-align:start;overflow-wrap:break-word}
table.grid th{background:#efe9dc;font-weight:700}
table.grid caption{caption-side:top;text-align:start;font-weight:700;padding:0 0 5px;color:var(--c)}
table.grid td.empty{background:repeating-linear-gradient(135deg,#fff 0 5px,#f6f3ec 5px 10px)}
table.grid.wide{font-size:12.5px;line-height:1.45}table.grid.wide td,table.grid.wide th{padding:2px 4px}
.tblnote{font:12px "Segoe UI",sans-serif;color:var(--muted);margin-top:3px}
.blk.table .txt,.blk.table{overflow-x:auto}
.dpg{font-size:12px;color:var(--muted);margin:10px 0 2px;padding-top:6px;border-top:1px dashed var(--line)}.dpg:first-child{border:0;margin-top:0;padding-top:0}
.follow{margin-left:14px;white-space:nowrap;cursor:pointer}.follow input{vertical-align:-2px;margin:0 4px 0 0}
.nav{margin-top:6px;font-size:12.5px;color:var(--muted)}.nav a{display:inline-block;min-width:22px;padding:0 4px;margin:0 1px;text-align:center;color:var(--ink);text-decoration:none;border:1px solid var(--line);border-radius:4px;cursor:pointer}.nav a:hover{background:var(--panel)}
@media(max-width:900px){body{height:auto;display:block}main{grid-template-columns:1fr}.pane{overflow:visible}}
"""

_JS = """
// the two sides follow each other: scrolling one brings the other to the same place of the same page
(()=>{const P=document.querySelectorAll('main>.pane');if(P.length<2)return;const S=P[0],D=P[1],box=document.getElementById('follow');
 let lead=null,busy=false;[S,D].forEach(p=>['wheel','touchstart','pointerdown','keydown'].forEach(v=>p.addEventListener(v,()=>{lead=p;},{passive:true})));
 const marks=(p,sel)=>[...p.querySelectorAll(sel)];
 const where=(p,sel)=>{const top=p.getBoundingClientRect().top,ms=marks(p,sel);let k=-1;
   for(let i=0;i<ms.length;i++){if(ms[i].getBoundingClientRect().top-top<=8)k=i;else break;}
   if(k<0)return null;const a=ms[k].getBoundingClientRect().top,b=k+1<ms.length?ms[k+1].getBoundingClientRect().top:a+p.scrollHeight-p.scrollTop;
   return {n:ms[k].id.slice(1),f:Math.min(Math.max((top-a)/Math.max(b-a,1),0),1)};};
 const go=(p,sel,pre,at)=>{const e=document.getElementById(pre+at.n);if(!e)return;const ms=marks(p,sel),i=ms.indexOf(e);
   const a=e.getBoundingClientRect().top,b=i+1<ms.length?ms[i+1].getBoundingClientRect().top:a+p.scrollHeight-p.scrollTop;
   p.scrollTop+=a+at.f*(b-a)-p.getBoundingClientRect().top;};
 const follow=(src)=>{if(busy||(box&&!box.checked)||lead!==src)return;busy=true;requestAnimationFrame(()=>{
   const at=src===S?where(S,'.pg'):where(D,'.dpg');if(at)(src===S?go(D,'.dpg','d',at):go(S,'.pg','s',at));busy=false;});};
 S.addEventListener('scroll',()=>follow(S),{passive:true});D.addEventListener('scroll',()=>follow(D),{passive:true});})();
document.querySelectorAll('.ln').forEach(e=>{const l=()=>document.getElementById(e.dataset.l);
  e.addEventListener('mouseenter',()=>l()&&l().classList.add('on'));e.addEventListener('mouseleave',()=>l()&&l().classList.remove('on'));
  e.addEventListener('click',ev=>{ev.stopPropagation();const b=l();if(b){b.classList.add('on');b.scrollIntoView({behavior:'smooth',block:'center'});}});});
document.querySelectorAll('.nav a').forEach(a=>a.addEventListener('click',()=>{const p=a.dataset.p;['s','d'].forEach(k=>{const e=document.getElementById(k+p);if(e)e.scrollIntoView({block:'start'});});}));
const on=(id,v)=>document.querySelectorAll('[data-b="'+id+'"]').forEach(e=>e.classList.toggle('on',v));
document.querySelectorAll('[data-b]').forEach(e=>{
  e.addEventListener('mouseenter',()=>on(e.dataset.b,true));
  e.addEventListener('mouseleave',()=>on(e.dataset.b,false));
  e.addEventListener('click',()=>{const o=[...document.querySelectorAll('[data-b="'+e.dataset.b+'"]')].find(x=>x!==e&&x.classList.contains(e.classList.contains('box')?'blk':'box'));if(o)o.scrollIntoView({behavior:'smooth',block:'center'});});
});
"""


def _dir(text: str) -> str:
    """rtl when the text has at least as many Arabic letters as Latin ones: an English or French block reads left to right."""
    ar = sum("\u0600" <= c <= "\u06ff" or "\u0750" <= c <= "\u077f" or "\ufb50" <= c <= "\ufeff" for c in text)
    la = sum(c.isascii() and c.isalpha() or "\u00c0" <= c <= "\u024f" for c in text)
    return "rtl" if ar >= la else "ltr"


def _png(gray: np.ndarray, width: int = 900) -> str:
    f = width / gray.shape[1]
    small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    # Level 6, not 9: the same pixels, files about 3% larger, and about ten times faster to write
    # (0.02 s a page instead of 0.25 s).
    return "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", small, [cv2.IMWRITE_PNG_COMPRESSION, 6])[1].tobytes()).decode()


def _table_html(b: dict, captions: dict) -> str:
    """A table block as an HTML table, columns in the table's own direction, empty places hatched."""
    t = b["table"]
    rtl = t["direction"] == "rtl"

    def row(cells, tag):
        out = []
        for c in cells:
            if c in ("^", "<"):
                continue  # covered by a cell that spans here
            if c is None:
                out.append(f'<{tag} class="empty"></{tag}>')
                continue
            span = (f' rowspan="{c["rowspan"]}"' if c.get("rowspan") else "") + (f' colspan="{c["colspan"]}"' if c.get("colspan") else "")
            out.append(f"<{tag}{span}>{html.escape(c['text'])}</{tag}>")
        return "<tr>" + "".join(out) + "</tr>"

    cap = captions.get(t.get("caption"), "")
    # one tbody for all: a header cell may span down into the first rows, which a thead would cut off
    body = ["<tbody>"] + ([row(t["header"], "th")] if t["header"] else []) + [row(r, "td") for r in t["rows"]] + ["</tbody>"]
    n = len(t["rows"]) + bool(t["header"])
    note = (f'{n} rows × {t["columns"]} columns, {"ruled" if t["ruled"] else "unruled: grid from alignment"}'
            + (", carries on over the page" if len(b["parts"]) > 1 else "")
            + (f', header repeated on the next page ({len(t["repeated_header_lines"])} lines, not shown again)' if t.get("repeated_header_lines") else ""))
    wide = " wide" if t["columns"] >= 7 else ""  # many columns: smaller type, so it fits beside the scan
    return (f'<table class="grid{wide}" dir="{"rtl" if rtl else "ltr"}" lang="{"ar" if rtl else "en"}">'
            + (f'<caption data-b="{t["caption"]}" style="--c:{COLORS["table_caption"]}">{html.escape(cap)}</caption>' if cap else "") + "".join(body) + f'</table><div class="tblnote">{note}</div>')


def _fold_figures(items: list) -> list[str]:
    """`items` alternate a tag ((block, page) for a figure block, else None) and the block's HTML. A picture read
    as text comes out as dozens of figure blocks of a letter or two each; three or more in a row on one page
    are folded into one line that opens to show them, so the text around them can still be followed."""
    out, run = [], []

    def flush():
        if len(run) >= 3:
            a, z = run[0][0]["order"], run[-1][0]["order"]
            out.append(f'<details class="figs" style="--c:{COLORS["figure"]}"><summary><b>{a}–{z}</b> Figure · {len(run)} pieces of '
                       f'a picture read as text (click to show)</summary>' + "".join(h for _, h in run) + "</details>")
        else:
            out.extend(h for _, h in run)
        run.clear()

    for tag, h in zip(items[0::2], items[1::2]):
        if tag is not None and run and run[-1][0]["parts"][0]["page"] == tag[1]:
            run.append((tag[0], h))
            continue
        flush()
        if tag is not None:
            run.append((tag[0], h))
        else:
            out.append(h)
    flush()
    return out


def _page_warnings(master: dict, number: int) -> str:
    """A banner over a scan page that may not have been read properly (pipeline._warn), and the lines that say so."""
    pg = next((p for p in master["pages"] if p["number"] == number), {})
    n = sum(1 for b in master["blocks"] for part in b.get("parts", []) if part["page"] == number for w in part.get("line_warnings", []) if w)
    said = list(pg.get("warnings", [])) + ([f"{n} line{'s' if n > 1 else ''} not read properly (dashed on the scan, wavy in the text)"] if n else [])
    if pg.get("pictures"):
        said.append(f"{len(pg['pictures'])} picture{'s' if len(pg['pictures']) > 1 else ''} not read")
    return f'<p class="pwarn">Not read properly: {html.escape("; ".join(said))}</p>' if said else ""


def write_viewer(master: dict, images: list[np.ndarray], path: Path, dpi: int = 300, seconds: float = 0.0) -> None:
    from .parse import join_sep

    name = lambda t: NAMES.get(t, t.replace("_", " ").capitalize())
    counts = Counter(b["type"] for b in master["blocks"])
    # every type present, also one this file has no colour for (it is drawn grey but still listed)
    used = [t for t in COLORS if t in counts] + sorted(t for t in counts if t not in COLORS)
    boxes: dict[int, list[str]] = {}
    for b in master["blocks"]:
        aside = "aside" in b
        if "table" in b:  # the table's cells, lightly, so one can see what was grouped
            t = b["table"]
            for c in [c for r in ([t["header"]] if t["header"] else []) + t["rows"] for c in r if isinstance(c, dict)]:
                pg = next(p for p in master["pages"] if p["number"] == c["page"])
                x0, y0, x1, y1 = c["box"]
                boxes.setdefault(c["page"], []).append(
                    f'<div class="cell{"" if c["text"] else " empty"}" style="left:{100 * x0 / pg["width"]:.2f}%;top:{100 * y0 / pg["height"]:.2f}%;'
                    f'width:{100 * (x1 - x0) / pg["width"]:.2f}%;height:{100 * (y1 - y0) / pg["height"]:.2f}%;--c:{COLORS["table"]}"></div>')
        for part in b["parts"]:
            pg = next(p for p in master["pages"] if p["number"] == part["page"])
            for x0, y0, x1, y1 in part["regions"]:
                style = (f"left:{100 * x0 / pg['width']:.2f}%;top:{100 * y0 / pg['height']:.2f}%;"
                         f"width:{100 * (x1 - x0) / pg['width']:.2f}%;height:{100 * (y1 - y0) / pg['height']:.2f}%;--c:{COLORS.get(b['type'], '#6b6b6b')}")
                badge = "" if aside else f"<b>{b['order']}</b>"
                boxes.setdefault(part["page"], []).append(
                    f'<div class="box{" aside" if aside else ""}" data-b="{b["id"]}" style="{style}" title="{name(b["type"])}">{badge}</div>')
    # each line of running text: a box on the scan, shown while the line is pointed at in the text
    spans: dict[str, str] = {}
    n_line = 0
    for b in master["blocks"]:
        if "table" in b or b["type"] == "noise" or not b["parts"] or not all("line_texts" in p for p in b["parts"]):
            continue
        got = []  # (line text, its id, its page)
        for part in b["parts"]:
            pg = next(p for p in master["pages"] if p["number"] == part["page"])
            warns = part.get("line_warnings") or [[] for _ in part["line_boxes"]]  # pipeline._warn: not read properly
            for (x0, y0, x1, y1), t, w in zip(part["line_boxes"], part["line_texts"], warns):
                if not t and not w:
                    continue
                n_line += 1
                boxes.setdefault(part["page"], []).append(
                    f'<div class="lb{" warned" if w else ""}" id="L{n_line}" style="left:{100 * x0 / pg["width"]:.2f}%;top:{100 * y0 / pg["height"]:.2f}%;'
                    f'width:{100 * (x1 - x0) / pg["width"]:.2f}%;height:{100 * (y1 - y0) / pg["height"]:.2f}%"></div>')
                got.append((t or "[?]", n_line, part["page"], w))
        out = ""
        for k, (t, n, pno, w) in enumerate(got):
            # a Latin word broken over two lines is shown whole, as the block's text has it (parse.join_sep)
            joined = k + 1 < len(got) and join_sep(t, got[k + 1][0]) == ""
            shown = t.rstrip()[:-1] if joined else t
            out += f'<span class="ln{" warned" if w else ""}" data-l="L{n}" title="scan page {pno}{html.escape(": not read properly: " + "; ".join(w)) if w else ""}">{html.escape(shown)}</span>' + ("" if joined or k + 1 == len(got) else " ")
        spans[b["id"]] = out
    scans = "".join(
        f'<p class="pg" id="s{i}">scan page {i}</p>{_page_warnings(master, i)}<div class="scan"><img src="{_png(img)}" decoding="async" alt="Scan of page {i}">{"".join(boxes.get(i, []))}</div>'
        for i, img in enumerate(images, master["pages"][0]["number"]))
    captions = {b["id"]: b["text"] for b in master["blocks"] if b["type"] == "table_caption"}
    doc, at = [], None
    for b in master["blocks"]:
        if b["type"] == "noise" or "caption_of" in b:
            continue
        p = b["parts"][0]["page"] if b["parts"] else at
        if p is not None and (at is None or p > at):  # a marker where the text reaches a new scan page
            doc += [None, f'<p class="dpg" id="d{p}">page {p}</p>']; at = p
        d = _dir(b["text"])
        doc.append((b, p) if b["type"] == "figure" and "aside" not in b else None)
        doc.append(
            f'<div class="blk {b["type"]}{" aside" if "aside" in b else ""}" data-b="{b["id"]}" style="--c:{COLORS.get(b["type"], "#6b6b6b")}">'
            f'<div class="tag">{"" if "aside" in b else "<b>" + str(b["order"]) + "</b>"}{name(b["type"])}</div>'
            + (f'<div>{_table_html(b, captions)}</div></div>' if "table" in b else
               f'<div class="txt" dir="{d}" lang="{"ar" if d == "rtl" else "en"}">{spans.get(b["id"]) or html.escape(b["text"]) or "&nbsp;"}</div></div>'))
    doc = "".join(_fold_figures(doc))
    nums = [p["number"] for p in master["pages"]]
    nav = ('<div class="nav">Go to page: ' + "".join(f'<a data-p="{n}">{n}</a>' for n in nums)
           + ' <label class="follow"><input type="checkbox" id="follow" checked> the two sides scroll together</label></div>') if len(nums) > 1 else ""
    legend = "".join(f'<span><i style="background:{COLORS.get(t, "#6b6b6b")}"></i>{name(t)} · {counts[t]}</span>' for t in used)
    speed = f" Parsed in {seconds:.0f} seconds ({seconds / max(len(images), 1):.1f} a page) on this machine's CPU." if seconds else ""
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{master['id']} · parsed locally</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@500;700&display=swap"><style>{_CSS}</style></head><body>
<header><h1>{master['id']} <span>· whole document, {len(images)} pages · parsed locally</span></h1>
<div class="by">Text: {html.escape(master['made_by']['text'])}. Structure: {html.escape(master['made_by']['structure'])}{speed}</div>
<div class="legend">{legend}</div>{nav}</header>
<main><div class="pane"><p class="cap">The scan, with every block boxed and numbered in reading order</p>{scans}</div>
<div class="pane"><p class="cap">Parsed document</p><div class="doc">{doc}</div></div></main>
<script>{_JS}</script></body></html>"""
    Path(path).write_text(page, encoding="utf-8")
