"""The measuring stick: the whole local pipeline on the 70 development pages, scored on what the owner sees.

Each page is drawn from its PDF by the crop box at 300 dpi (cached in out/e2e/pages/), parsed by the
local pipeline exactly as scripts/serve.py does (line finder, line reader, structure model, its own
lines and its own reading order), and the master file that comes out is scored against labels three
strong models agreed on. No person has checked these labels: they are the best reference this project has.

Azure's lines are the ruler, at scoring time only: each Azure line (out/labeller/dev_inputs/PAGE/lines.json)
is given to the line of ours whose box holds its centre (the smallest such box, 0.02 inch of slack, as
scripts/score_with_compare.py does). An Azure line in no line of ours is "lost" and counts as wrong.

What is scored, per page, per kind of page and overall:
  types    each Azure line takes the type of our block that holds it; the reference type is the one at
           least two of Opus, Sonnet and Gemini Argon gave (lines with no majority are left out).
  blocks   reference blocks (Opus's; Sonnet's and Argon's shown too) we reproduce exactly (same set of
           Azure lines), and how many we split or merge with another.
  order    consecutive lines in the reference's reading order that both landed: does ours put the first
           first? Pairs inside one block (tables apart) and pairs from one block to the next ("across").
           Running heads, feet, page numbers and scan marks are outside the reading order and left out.
  text     for Azure lines that are the only Azure line in their line of ours: character difference from
           Azure's text after folding (vowel marks dropped, Persian letters folded). This is AGREEMENT WITH
           AZURE, not accuracy: Azure makes mistakes too.
  speed    seconds a page: drawing (0 when cached), reading (line finding and reading), structure.

    uv run python scripts/e2e_judged.py --name base_v2                                  # the default pipeline
    uv run python scripts/e2e_judged.py --name v3 --typer out/structure_v3.pt
    uv run python scripts/e2e_judged.py --name v3_fast --typer out/structure_v3.pt --lines-from base_v2
                                                   # reuse the lines read by an earlier run: structure only, seconds
    uv run python scripts/e2e_judged.py --compare base_v2 v3
    uv run python scripts/e2e_judged.py --show base_v2                                  # the summary again
    uv run python scripts/e2e_judged.py --rescore base_v2      # score saved masters again (after a scorer change)

A second, untouched test set (--set check): about 200 pages from documents no part of this project had used,
chosen by scripts/check_pages.py (list in out/labeller/check_pages.json, inputs in out/labeller/check_inputs,
PDFs in /home/yassine/data/check_docs). Its reference is made the same way from three other labellers, Haiku,
Gemini Flash and Gemini Argon (out/labeller/check_haiku|check_flash|check_argon): a line's type is what two of
the three chose; blocks and order are scored against Haiku's blocks (Flash's and Argon's shown too). A saved run
remembers its set, so --show, --rescore and --compare need no --set. NEVER tune on these pages.
    uv run python scripts/e2e_judged.py --set check --name check_now --typer "out/structure_v2.pt+out/structure_v3.pt"

Results: out/e2e/NAME.json (everything, per page), out/e2e/NAME.offenders.txt (the lines we got wrong,
worst pages first), out/e2e/NAME/masters/PAGE.master.json (what the owner would see), and
out/e2e/NAME/infos/PAGE.pkl (the lines read, for --lines-from).

From Python (for a pipeline of your own; it must return a master file, as ocrl.doc.parse.to_master makes):
    sys.path.insert(0, "scripts"); import e2e_judged as E
    res = E.run("mine", E.LocalPipeline(typer="out/structure_v3.pt"))      # or any callable, see run()
    print(E.summary(res))
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse, json, pickle, re, resource, subprocess, sys, time, hashlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "out/labeller"
E2E = ROOT / "out/e2e"
PAGES = E2E / "pages"
DOCS = Path("/home/yassine/data/judged_docs")  # ID/ID.pdf, fetched from the owner's bucket
JUDGES = ("opus", "sonnet", "argon")
BLOCK_REF = "opus"  # blocks and reading order: Opus's labelling, the one that agrees most with the other two
SET = "dev"
# the test sets: their inputs, label folders (LAB/PREFIX_JUDGE), judges, block reference, PDFs and where each page's kind is
SETS = {
    "dev": {"inputs": "dev_inputs", "prefix": "dev", "judges": ("opus", "sonnet", "argon"), "block_ref": "opus", "docs": "/home/yassine/data/judged_docs"},
    "check": {"inputs": "check_inputs", "prefix": "check", "judges": ("haiku", "flash", "argon"), "block_ref": "haiku", "docs": "/home/yassine/data/check_docs"},
}


def use_set(name: str) -> None:
    """Switch every module-level setting to one test set (\"dev\", the default, or \"check\")."""
    global SET, JUDGES, BLOCK_REF, DOCS
    cfg = SETS[name]
    SET, JUDGES, BLOCK_REF, DOCS = name, cfg["judges"], cfg["block_ref"], Path(cfg["docs"])


def _who(names) -> list[str]:
    return [n.capitalize() for n in names]
ASIDE = {"running_head", "running_foot", "page_number", "noise"}  # outside the reading order
DPI = 300
SLACK = 0.02  # inch, as score_with_compare.py
READER = "/home/yassine/data/inkscript_labelled/experiments/89_final_reader/out/weights/v7/last.pt"
TYPER = "out/structure_v2.pt"
LINEFINDER = "out/linefinder.pt"
CKPT = "out/recognizer_v3.pt"


# ------------------------------------------------------------------------------------------------ reference
def _js(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8").split("=", 1)[1].rstrip().rstrip(";"))


def load_reference() -> dict[str, dict]:
    """Every page of the current set: Azure's lines, the majority type of each line, each judge's blocks, its kind."""
    cfg = SETS[SET]
    if SET == "dev":
        kinds = {p["id"]: p["kind"] for p in _js(ROOT / "out/census/dev_data.js")["pages"]}
    else:
        kinds = {p["page"]: p["kind"] for p in json.loads((LAB / f"{SET}_pages.json").read_text(encoding="utf-8"))["pages"]}
    refs = {}
    for d in sorted((LAB / cfg["inputs"]).iterdir()):
        if not (d / "lines.json").exists():
            continue
        page = d.name
        doc, n = page.rsplit("_p", 1)
        lines = json.loads((d / "lines.json").read_text(encoding="utf-8"))
        blocks = {}
        for w in JUDGES:
            f = LAB / f"{cfg['prefix']}_{w}" / f"{page}.json"
            if f.exists():
                blocks[w] = json.loads(f.read_text(encoding="utf-8"))["blocks"]
        types = {w: {r: b["type"] for b in bl for r in b["lines"]} for w, bl in blocks.items()}
        majority, votes = {}, {}
        for ln in lines:
            v = Counter(types[w].get(ln["id"]) for w in types)
            top, cnt = (v.most_common(1) or [(None, 0)])[0]  # no labels yet (a run made before labelling, --rescore later): no majority
            votes[ln["id"]] = {w: types[w].get(ln["id"]) for w in types}
            if top is not None and cnt >= 2:
                majority[ln["id"]] = top
        refs[page] = {"page": page, "doc": doc, "number": int(n), "kind": kinds.get(page, "?"), "lines": lines,
                      "majority": majority, "votes": votes, "blocks": blocks, "types": types}
    return refs


# ------------------------------------------------------------------------------------------------ drawing
def draw(doc: str, number: int):
    """The page as the product draws it: crop box, 300 dpi, grey. Cached as PNG under out/e2e/pages/."""
    import cv2

    PAGES.mkdir(parents=True, exist_ok=True)
    f = PAGES / f"{doc}_p{number}.png"
    if not f.exists():
        pdf = DOCS / doc / f"{doc}.pdf"
        if not pdf.exists():
            raise FileNotFoundError(f"{pdf} is missing: fetch it from s3://mandumah-source-docs/{doc}/{doc}.pdf")
        stem = PAGES / f"{doc}_p{number}.tmp"
        subprocess.run(["pdftoppm", "-cropbox", "-r", str(DPI), "-gray", "-png", "-f", str(number), "-l", str(number), "-singlefile",
                        str(pdf), str(stem)], check=True, capture_output=True, timeout=300)
        Path(str(stem) + ".png").rename(f)
    img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"could not read {f}")
    return img


# ------------------------------------------------------------------------------------------------ the pipeline
class LocalPipeline:
    """The product's pipeline, as scripts/serve.py runs it. Called as pipe(gray, page_number, doc_id, cached=None)
    and returns (master, infos, seconds_by_stage). With `cached` (the PageInfo list an earlier run read) the
    reading stage is skipped and everything after it in ocrl.doc.parse.parse runs as usual."""

    def __init__(self, reader: str = READER, typer: str = TYPER, linefinder: str = LINEFINDER, ckpt: str = CKPT, load_reader: bool = True):
        from ocrl.structure.tag import Typer

        self.reader, self.typer_path, self.linefinder, self.ckpt = reader, typer, linefinder, ckpt
        self.ocr = None
        if load_reader:
            from ocrl.pipeline import OCR
            self.ocr = OCR(ckpt, linefinder=linefinder or None, line_reader=reader or None)
        self.typer = Typer(typer) if typer else None

    def describe(self) -> dict:
        return {"reader": self.reader, "typer": self.typer_path, "linefinder": self.linefinder, "ckpt": self.ckpt}

    def __call__(self, gray, number: int, doc_id: str, cached=None):
        from ocrl.doc import parse as P

        if self.typer is not None:
            self.typer.open_item = False  # each page is its own document here
        t0 = time.time()
        mark = {}
        report = lambda done, total: mark.setdefault("read", time.time())
        import inspect
        kw = {"progress": report} if "progress" in inspect.signature(P.parse).parameters else {}  # older parse(): no stage split
        if cached is not None:
            real = P.read_page
            P.read_page = lambda ocr, g, n: cached[0]
            try:
                infos, blocks = P.parse(self.ocr, [gray], number, self.typer, **kw)
            finally:
                P.read_page = real
        else:
            if self.ocr is None:
                raise RuntimeError("this pipeline was made without its reader (load_reader=False): give it cached lines")
            infos, blocks = P.parse(self.ocr, [gray], number, self.typer, **kw)
        t_read = mark.get("read", t0 if cached is not None else time.time())  # no split known: all counted as reading
        how = f"OCRL structure model ({Path(self.typer_path).stem}) with the list rule, local." if self.typer else ""
        master = P.to_master(doc_id, infos, blocks, DPI, Path(self.reader).parent.name + " line reader" if self.reader else Path(self.ckpt).stem, how)
        t1 = time.time()
        attach_line_texts(master, infos)
        return master, infos, {"read": 0.0 if cached is not None else t_read - t0, "structure": t1 - t_read}


def attach_line_texts(master: dict, infos) -> None:
    """Our text of each line, beside its box (to_master keeps only the block's text)."""
    by_box = defaultdict(list)
    for p in infos:
        for ln in p.lines:
            by_box[(p.number, tuple(round(v / DPI, 3) for v in (ln.x0, ln.y0, ln.x1, ln.y1)))].append(ln.text)
    for b in master["blocks"]:
        for part in b["parts"]:
            if "line_texts" in part:
                continue
            texts = []
            for bx in part["line_boxes"]:
                got = by_box.get((part["page"], tuple(bx)))
                texts.append((got.pop(0) if len(got) > 1 else got[0]) if got else "")
            part["line_texts"] = texts


# ------------------------------------------------------------------------------------------------ scoring
_NORM = None


def norm(t: str) -> str:
    """The folding scripts/azure_agreement.py uses: Persian letters folded, vowel marks and tatweel dropped."""
    global _NORM
    if _NORM is None:
        from ocrl.recog.realink import fold
        from ocrl.text import strip_marks
        _NORM = lambda s: " ".join(strip_marks(" ".join(fold(w) for w in s.split())).split())
    return _NORM(t)


_TURNED = re.compile(r"turned upright \((\d+) degrees counter-clockwise\)")


def page_turn(pg: dict) -> int:
    """Quarter turns (np.rot90's k) the parser gave this page before reading it: the master's "turned", or, in
    masters saved before that was written, the degrees in the page's warning. 0 for a page read as it lay."""
    if "turned" in pg:
        return int(pg["turned"]) % 4
    for w in pg.get("warnings", []):
        m = _TURNED.search(w)
        if m:
            return (int(m.group(1)) // 90) % 4
    return 0


def unturn(box, k: int, W: float, H: float):
    """A box on the page turned by np.rot90(page, k) (W x H: the turned page's size) back on the page as it lay.
    rot90 by 1 sends (x, y) of the page as it lay to (y, W0 - x); by 3 to (H0 - y, x); by 2 to (W0 - x, H0 - y)."""
    x0, y0, x1, y1 = box
    if k == 1:   # page as it lay is H wide, W high
        return (H - y1, x0, H - y0, x1)
    if k == 3:
        return (y0, W - x1, y1, W - x0)
    if k == 2:
        return (W - x1, H - y1, W - x0, H - y0)
    return tuple(box)


def our_lines(master: dict, number: int) -> list[dict]:
    """Our lines on this page in the order the owner reads them: blocks as listed, lines as listed in each.
    Boxes are in thousandths of the page AS IT LAY (Azure's frame): a page the parser turned upright has its
    boxes turned back first."""
    pg = next(p for p in master["pages"] if p["number"] == number)
    W, H = pg["width"], pg["height"]
    k = page_turn(pg)
    out = []
    for bi, b in enumerate(master["blocks"]):
        boxes = []  # (box, text) in the order the owner reads them
        for part in b["parts"]:
            if part["page"] == number:
                boxes += zip(map(tuple, part["line_boxes"]), part.get("line_texts") or [""] * len(part["line_boxes"]))
        grid = b.get("table")
        if grid and (grid.get("header") or grid.get("rows")):
            # a parsed table is read cell by cell: header, then rows, each row in its reading order
            text_of, seen, ordered = dict(boxes), set(), []
            for row in ([grid["header"]] if grid.get("header") else []) + grid.get("rows", []):
                for cell in row:
                    if isinstance(cell, dict) and cell.get("page", number) == number:
                        for bx in map(tuple, cell.get("lines", [])):
                            if bx in text_of and bx not in seen:
                                seen.add(bx); ordered.append((bx, text_of[bx]))
            boxes = ordered + [(bx, t) for bx, t in boxes if bx not in seen]
        for bx, t in boxes:
            if k:
                ux = unturn(bx, k, W, H)
                W0, H0 = (H, W) if k % 2 else (W, H)
                box = (ux[0] / W0 * 1000, ux[1] / H0 * 1000, ux[2] / W0 * 1000, ux[3] / H0 * 1000)
            else:
                box = (bx[0] / W * 1000, bx[1] / H * 1000, bx[2] / W * 1000, bx[3] / H * 1000)
            out.append({"block": bi, "type": b["type"], "box": box, "text": t, "pos": len(out)})
    if k % 2:
        W, H = H, W
    return out, 1000 * SLACK / W, 1000 * SLACK / H


def assign(ref: dict, ours: list[dict], sx: float, sy: float) -> dict[str, dict]:
    """Azure line id -> the line of ours holding its centre (the smallest such)."""
    got = {}
    for ln in ref["lines"]:
        x0, y0, x1, y1 = ln["box"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hits = [o for o in ours if o["box"][0] - sx <= cx <= o["box"][2] + sx and o["box"][1] - sy <= cy <= o["box"][3] + sy]
        if hits:
            got[ln["id"]] = min(hits, key=lambda o: (o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1]))
    return got


def score_page(master: dict | None, ref: dict) -> dict:
    """Counts for one page (sum them to aggregate) and the lines we got wrong."""
    from rapidfuzz.distance import Levenshtein
    from ocrl.text import CHARSET

    c = Counter()
    wrong = []
    text = {ln["id"]: ln["text"] for ln in ref["lines"]}
    if master is None:  # the page failed: everything is lost
        ours, at = [], {}
    else:
        ours, sx, sy = our_lines(master, ref["number"])
        at = assign(ref, ours, sx, sy)
    c["az_lines"] = len(ref["lines"]); c["az_lost"] = sum(ln["id"] not in at for ln in ref["lines"])
    c["our_lines"] = len(ours)

    # types, against the majority and against each judge alone
    conf = Counter()
    for r, t in ref["majority"].items():
        o = at.get(r)
        mine = o["type"] if o else None
        c["maj_lines"] += 1; c["maj_right"] += mine == t; c["maj_lost"] += o is None
        if mine != t:
            conf[t, mine or "LOST"] += 1
            wrong.append({"id": r, "ref": t, "ours": mine or "LOST", "votes": ref["votes"][r], "azure": text[r][:120], "our_text": (o["text"][:120] if o else "")})
    for w, ty in ref["types"].items():
        for r, t in ty.items():
            o = at.get(r)
            c[f"t_{w}_lines"] += 1; c[f"t_{w}_right"] += (o["type"] if o else None) == t

    # blocks and reading order, against each judge's blocks
    for w, bl in ref["blocks"].items():
        mine_of = {r: at[r]["block"] for b in bl for r in b["lines"] if r in at}
        held = defaultdict(set)
        for r, bi in mine_of.items():
            held[bi].add(r)
        for b in bl:
            ids = set(b["lines"])
            if not ids:
                continue
            landed = {mine_of[r] for r in ids if r in mine_of}
            c[f"b_{w}_total"] += 1
            if not landed:
                c[f"b_{w}_lost"] += 1
                continue
            c[f"b_{w}_exact"] += len(landed) == 1 and held[next(iter(landed))] == ids and all(r in mine_of for r in ids)
            c[f"b_{w}_split"] += len(landed) > 1
            c[f"b_{w}_merged"] += any(held[bi] - ids for bi in landed)
        seq = [(r, k, b["type"]) for k, b in enumerate(bl) for r in b["lines"]]
        for (r1, k1, t1), (r2, k2, t2) in zip(seq, seq[1:]):
            if t1 in ASIDE or t2 in ASIDE or r1 not in at or r2 not in at:
                continue
            p1, p2 = at[r1]["pos"], at[r2]["pos"]
            if p1 == p2:  # both in one line of ours: no order to judge
                c[f"o_{w}_ties"] += 1
                continue
            kind = "cross" if k1 != k2 else "tab" if t1 == "table" else "in"
            c[f"o_{w}_{kind}_n"] += 1; c[f"o_{w}_{kind}_ok"] += p1 < p2

    # text: Azure lines that are the only Azure line in their line of ours
    share = Counter(id(o) for o in at.values())
    ok = set(CHARSET + " ")
    for ln in ref["lines"]:
        o = at.get(ln["id"])
        want = norm(ln["text"])
        if o is None or share[id(o)] != 1 or not want:
            continue
        got = norm(o["text"])
        e = Levenshtein.distance(got, want)
        c["x_lines"] += 1; c["x_chars"] += len(want); c["x_edits"] += e; c["x_empty"] += not got
        if all(ch in ok and not (ch.isascii() and ch.isalpha()) for ch in want):
            c["x_ar_chars"] += len(want); c["x_ar_edits"] += e
    return {"c": dict(c), "confusions": [[a, b, n] for (a, b), n in conf.most_common()], "wrong": wrong}


# ------------------------------------------------------------------------------------------------ aggregating
def pct(a, b):
    return 100 * a / b if b else float("nan")


def metrics(c: Counter) -> dict:
    """The headline numbers from summed counts. Percentages; nan where nothing was counted."""
    w = BLOCK_REF
    m = {
        "pages": c["pages"], "lines": c["maj_lines"],
        "types": pct(c["maj_right"], c["maj_lines"]),
        "lost": pct(c["maj_lost"], c["maj_lines"]),
        "blocks_exact": pct(c[f"b_{w}_exact"], c[f"b_{w}_total"]),
        "blocks_split": pct(c[f"b_{w}_split"], c[f"b_{w}_total"]),
        "blocks_merged": pct(c[f"b_{w}_merged"], c[f"b_{w}_total"]),
        "order_in": pct(c[f"o_{w}_in_ok"], c[f"o_{w}_in_n"]),
        "order_table": pct(c[f"o_{w}_tab_ok"], c[f"o_{w}_tab_n"]),
        "order_across": pct(c[f"o_{w}_cross_ok"], c[f"o_{w}_cross_n"]),
        "text_vs_azure": pct(c["x_edits"], c["x_chars"]),
        "text_vs_azure_arabic": pct(c["x_ar_edits"], c["x_ar_chars"]),
        "sec_page": (c["s_draw"] + c["s_read"] + c["s_structure"]) / c["pages"] if c["pages"] else float("nan"),
        "sec_read": c["s_read"] / c["pages"] if c["pages"] else float("nan"),
        "sec_structure": c["s_structure"] / c["pages"] if c["pages"] else float("nan"),
    }
    for j in JUDGES:
        m[f"types_vs_{j}"] = pct(c[f"t_{j}_right"], c[f"t_{j}_lines"])
        m[f"blocks_vs_{j}"] = pct(c[f"b_{j}_exact"], c[f"b_{j}_total"])
        m[f"across_vs_{j}"] = pct(c[f"o_{j}_cross_ok"], c[f"o_{j}_cross_n"])
    return m


def aggregate(res: dict) -> tuple[Counter, dict[str, Counter]]:
    tot, kinds = Counter(), defaultdict(Counter)
    for p in res["pages"].values():
        tot.update(p["c"]); kinds[p["kind"]].update(p["c"])
    return tot, kinds


def _f(v, w=5, d=1):
    return f"{'—':>{w}}" if v != v else f"{v:{w}.{d}f}"


def summary(res: dict) -> str:
    tot, kinds = aggregate(res)
    m = metrics(tot)
    cfg = res["config"]
    L = [f"run {res['name']}: {m['pages']} pages, {m['lines']} lines with a majority type | typer {cfg.get('typer') or 'rules'} | "
         f"reader {Path(cfg.get('reader') or cfg.get('ckpt') or '?').parent.name or cfg.get('reader')}" + (f" | lines from {cfg['lines_from']}" if cfg.get("lines_from") else "")]
    if res.get("errors"):
        L.append(f"!! {len(res['errors'])} pages FAILED and count as all lost: {', '.join(res['errors'])}")
    others = [j for j in JUDGES if j != BLOCK_REF]
    L.append(f"  line types right   {_f(m['types'])}%   (vs " + ", ".join(f"{n} {_f(m['types_vs_' + j])}" for n, j in zip(_who(JUDGES), JUDGES)) + ")")
    L.append(f"  lost lines         {_f(m['lost'])}%   (reference lines in no line of ours; counted wrong above)")
    L.append(f"  blocks exact       {_f(m['blocks_exact'])}%   split {_f(m['blocks_split'])}%  merged {_f(m['blocks_merged'])}%   "
             "(exact vs " + ", ".join(f"{n} {_f(m['blocks_vs_' + j])}" for n, j in zip(_who(others), others)) + ")")
    L.append(f"  reading order      in a block {_f(m['order_in'])}%  in a table {_f(m['order_table'])}%  across blocks {_f(m['order_across'])}%   "
             "(across vs " + ", ".join(f"{n} {_f(m['across_vs_' + j])}" for n, j in zip(_who(others), others)) + ")")
    L.append(f"  text vs AZURE      {_f(m['text_vs_azure'], d=2)}% characters differ (Arabic-only lines {_f(m['text_vs_azure_arabic'], d=2)}%; "
             f"{tot['x_lines']} lines matched one to one) - agreement with Azure, not accuracy")
    L.append(f"  speed              {_f(m['sec_page'], d=2)} s a page (reading {_f(m['sec_read'], d=2)}, structure {_f(m['sec_structure'], d=2)}, "
             f"drawing {_f(tot['s_draw'] / max(tot['pages'], 1), d=2)})")
    L.append("")
    L.append(f"  {'kind of page':38s} {'lines':>5s} {'types':>5s} {'lost':>5s} {'blkEx':>5s} {'split':>5s} {'merge':>5s} {'ordIn':>5s} {'ordX':>5s} {'txtAz':>5s} {'s/pg':>5s}")
    for k, c in sorted(kinds.items(), key=lambda kv: metrics(kv[1])["types"]):
        q = metrics(c)
        L.append(f"  {k:38s} {q['lines']:5d} {_f(q['types'])} {_f(q['lost'])} {_f(q['blocks_exact'])} {_f(q['blocks_split'])} {_f(q['blocks_merged'])} "
                 f"{_f(q['order_in'])} {_f(q['order_across'])} {_f(q['text_vs_azure'])} {_f(q['sec_page'])}")
    conf = Counter()
    for p in res["pages"].values():
        for a, b, n in p["confusions"]:
            conf[a, b] += n
    L.append("")
    L.append("  most common mistakes (reference -> ours): " + ", ".join(f"{a}->{b} x{n}" for (a, b), n in conf.most_common(10)))
    worst = sorted(res["pages"].items(), key=lambda kv: kv[1]["c"].get("maj_right", 0) - kv[1]["c"].get("maj_lines", 0))[:6]
    L.append("  pages losing most lines: " + ", ".join(f"{pg} ({p['c'].get('maj_lines', 0) - p['c'].get('maj_right', 0)}/{p['c'].get('maj_lines', 0)})" for pg, p in worst))
    L.append(f"  details: out/e2e/{res['name']}.json, wrong lines: out/e2e/{res['name']}.offenders.txt")
    return "\n".join(L)


def compare(a: dict, b: dict) -> str:
    ta, ka = aggregate(a); tb, kb = aggregate(b)
    ma, mb = metrics(ta), metrics(tb)
    keys = [("types", "line types right"), ("lost", "lost lines"), ("blocks_exact", "blocks exact"), ("blocks_split", "blocks split"),
            ("blocks_merged", "blocks merged"), ("order_in", "order in a block"), ("order_table", "order in a table"),
            ("order_across", "order across blocks"), ("text_vs_azure", "text differs from Azure"), ("sec_page", "seconds a page"),
            ] + [(f"types_vs_{j}", f"types vs {n}") for n, j in zip(_who(JUDGES), JUDGES) if j != BLOCK_REF]
    if SET != "dev":  # blocks and order against each labeller alone, so the choice of reference can be seen
        keys += [(f"blocks_vs_{j}", f"blocks exact vs {n}") for n, j in zip(_who(JUDGES), JUDGES) if j != BLOCK_REF]
        keys += [(f"across_vs_{j}", f"order across vs {n}") for n, j in zip(_who(JUDGES), JUDGES) if j != BLOCK_REF]
    L = [f"{'':28s} {a['name']:>12s} {b['name']:>12s} {'change':>8s}"]
    for k, what in keys:
        d = mb[k] - ma[k]
        L.append(f"  {what:26s} {_f(ma[k], 12, 2)} {_f(mb[k], 12, 2)} {_f(d, 8, 2)}")
    L.append("")
    L.append(f"  by kind of page: {a['name']} -> {b['name']} (change)")
    L.append(f"  {'kind of page':38s} {'lines':>5s} {'types':>16s} {'blocks exact':>16s} {'order across':>16s} {'text vs Az':>16s}")
    for k in sorted(set(ka) | set(kb), key=lambda k: metrics(kb.get(k, Counter()))["types"] - metrics(ka.get(k, Counter()))["types"]):
        qa, qb = metrics(ka.get(k, Counter())), metrics(kb.get(k, Counter()))
        cell = lambda x: f"{_f(qa[x])}>{_f(qb[x])}({_f(qb[x] - qa[x], 4)})"
        L.append(f"  {k:38s} {qb['lines']:5d} {cell('types')} {cell('blocks_exact')} {cell('order_across')} {cell('text_vs_azure')}")
    ch = []
    for pg in set(a["pages"]) & set(b["pages"]):
        ca, cb = a["pages"][pg]["c"], b["pages"][pg]["c"]
        ch.append((cb.get("maj_right", 0) - ca.get("maj_right", 0), pg, ca.get("maj_lines", 0)))
    ch.sort()
    L.append("")
    L.append("  pages that got worse (lines right lost): " + ", ".join(f"{pg} {d:+d}/{n}" for d, pg, n in ch[:6] if d < 0))
    L.append("  pages that got better (lines right won): " + ", ".join(f"{pg} {d:+d}/{n}" for d, pg, n in ch[::-1][:6] if d > 0))
    return "\n".join(L)


def write_offenders(res: dict, path: Path) -> None:
    rows = sorted(res["pages"].items(), key=lambda kv: -len(kv[1]["wrong"]))
    with open(path, "w", encoding="utf-8") as f:
        w = _who(JUDGES)
        f.write(f"# lines of run {res['name']} whose type differs from the majority of {', '.join(w[:-1])} and {w[-1]} (LOST: in no line of ours)\n")
        f.write("# page image: out/e2e/pages/PAGE.png; our master file: out/e2e/NAME/masters/PAGE.master.json\n")
        for pg, p in rows:
            if not p["wrong"]:
                continue
            f.write(f"\n== {pg} [{p['kind']}] {len(p['wrong'])} of {p['c'].get('maj_lines', 0)} wrong\n")
            for w in p["wrong"]:
                f.write(f"  {w['id']:>7s}  ref {w['ref']:<14s} ours {w['ours']:<14s} | azure: {w['azure']}\n")
                if w["our_text"]:
                    f.write(f"  {'':7s}  {'':33s} | ours:  {w['our_text']}\n")


# ------------------------------------------------------------------------------------------------ running
def provenance(out: Path | None = None) -> dict:
    """Which code made a run: the commit, where ocrl was imported from, and the uncommitted change to src
    (written in full to out/e2e/NAME/src.diff, so the run can be rebuilt)."""
    import ocrl

    src = Path(ocrl.__file__).resolve().parent

    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=30).stdout
        except Exception:
            return ""
    info = {"commit": git("rev-parse", "--short", "HEAD").strip(), "ocrl": str(src)}
    if src == (ROOT / "src/ocrl").resolve():
        diff = git("diff", "HEAD", "--", "src") + "".join(
            f"\n+++ untracked {f}\n" + (ROOT / f).read_text(encoding="utf-8", errors="replace")
            for f in git("ls-files", "--others", "--exclude-standard", "--", "src").split() if f.endswith(".py"))
        info.update(src_changed=bool(diff.strip()), src_diff_sha=hashlib.sha1(diff.encode()).hexdigest()[:10] if diff.strip() else "")
        if out is not None and diff.strip():
            (out / "src.diff").write_text(diff, encoding="utf-8")
    else:
        info["note"] = "ocrl imported from a copy (--src), not from the working tree"
    return info


def check_cached(cached) -> None:
    """Lines pickled by an earlier run are only good if the reading stage still makes the same objects."""
    import dataclasses
    from ocrl.doc import parse as P

    for obj, cls in [(cached[0], P.PageInfo)] + ([(cached[0].lines[0], P.TextLine)] if cached[0].lines else []):
        want = {f.name for f in dataclasses.fields(cls)}
        have = set(vars(obj))
        if want - have:
            raise RuntimeError(f"--lines-from: the cached {cls.__name__} lacks {sorted(want - have)}: the reading stage has changed "
                               f"since that run, so its lines cannot stand in for a fresh reading. Run without --lines-from.")


def run(name: str, pipeline=None, pages: list[str] | None = None, lines_from: str | None = None, quiet: bool = False) -> dict:
    """Parse and score. `pipeline` is a LocalPipeline or any callable (gray, page_number, doc_id, cached) ->
    (master, infos, {"read": s, "structure": s}); `infos` may be None if --lines-from will never be used on it.
    `pages` limits the run to some page names (as in out/labeller/dev_inputs). Writes out/e2e/NAME.json."""
    refs = load_reference()
    names = [p for p in refs if not pages or p in pages]
    pipeline = pipeline or LocalPipeline()
    out = E2E / name
    (out / "masters").mkdir(parents=True, exist_ok=True); (out / "infos").mkdir(parents=True, exist_ok=True)
    res = {"name": name, "set": SET, "config": {**(pipeline.describe() if hasattr(pipeline, "describe") else {"pipeline": repr(pipeline)}), "lines_from": lines_from},
           "provenance": provenance(out), "when": time.strftime("%Y-%m-%d %H:%M"), "pages": {}, "errors": []}
    began = time.time()
    for i, pg in enumerate(names, 1):
        ref = refs[pg]
        t = time.time()
        cached_png = (PAGES / f"{pg}.png").exists()
        master, secs = None, {"read": 0.0, "structure": 0.0}
        try:
            gray = draw(ref["doc"], ref["number"])
            t_draw = 0.0 if cached_png else time.time() - t
            cached = None
            if lines_from:
                with open(E2E / lines_from / "infos" / f"{pg}.pkl", "rb") as f:
                    cached = pickle.load(f)
                check_cached(cached)
            master, infos, secs = pipeline(gray, ref["number"], ref["doc"], cached)
            (out / "masters" / f"{pg}.master.json").write_text(json.dumps(master, ensure_ascii=False), encoding="utf-8")
            if infos is not None and not lines_from:
                with open(out / "infos" / f"{pg}.pkl", "wb") as f:
                    pickle.dump(infos, f)
        except Exception as e:  # a failed page is shown, never dropped: all its lines count as lost
            import traceback
            traceback.print_exc()
            res["errors"].append(pg)
            t_draw = 0.0
            print(f"!! {pg} failed: {type(e).__name__}: {e}", flush=True)
        s = score_page(master, ref)
        s["c"].update(pages=1, s_draw=t_draw, s_read=secs["read"], s_structure=secs["structure"])
        res["pages"][pg] = {"kind": ref["kind"], **s}
        if not quiet:
            c = s["c"]
            print(f"[{i:2d}/{len(names)}] {pg:24s} {ref['kind'][:26]:26s} types {c.get('maj_right', 0):3d}/{c.get('maj_lines', 0):<3d} "
                  f"lost {c.get('maj_lost', 0):3d}  {time.time() - t:5.1f} s   ({time.time() - began:.0f} s so far)", flush=True)
    res["seconds"] = round(time.time() - began, 1)
    save(res)
    return res


def save(res: dict) -> None:
    E2E.mkdir(parents=True, exist_ok=True)
    (E2E / f"{res['name']}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    write_offenders(res, E2E / f"{res['name']}.offenders.txt")


def rescore(name: str) -> dict:
    """Score the masters an earlier run saved, again, without parsing (after a change to the scorer)."""
    old = load(name)
    use_set(old.get("set", "dev"))
    refs = load_reference()
    for pg, p in old["pages"].items():
        f = E2E / name / "masters" / f"{pg}.master.json"
        master = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
        s = score_page(master, refs[pg])
        s["c"].update({k: p["c"].get(k, 0) for k in ("pages", "s_draw", "s_read", "s_structure")})
        old["pages"][pg] = {"kind": refs[pg]["kind"], **s}
    save(old)
    return old


def labelling_as_master(ref: dict, who: str) -> dict:
    """A judge's labelling dressed as one of our master files (Azure's boxes and text as our lines), for --selftest."""
    W, H = 8.5, 11.0  # any page size: boxes are given in thousandths and scored in thousandths
    by_id = {ln["id"]: ln for ln in ref["lines"]}
    blocks = []
    for b in ref["blocks"][who]:
        ls = [by_id[r] for r in b["lines"] if r in by_id]
        if ls:
            blocks.append({"type": b["type"], "parts": [{"page": ref["number"], "line_boxes": [[l["box"][0] * W / 1000, l["box"][1] * H / 1000, l["box"][2] * W / 1000,
                           l["box"][3] * H / 1000] for l in ls], "line_texts": [l["text"] for l in ls]}]})
    return {"pages": [{"number": ref["number"], "width": W, "height": H}], "blocks": blocks}


def selftest(who: str | None = None) -> dict:
    """Score a judge's own labelling as if it were our output. Against that judge everything should be 100%
    (and text 0%); anything short of that is the ruler's own error (Azure boxes that overlap, lines left out)."""
    who = who or BLOCK_REF
    res = {"name": f"selftest_{who}", "set": SET, "config": {"typer": f"{who}'s own labels", "reader": "azure"}, "pages": {}, "errors": []}
    for pg, ref in load_reference().items():
        s = score_page(labelling_as_master(ref, who), ref)
        s["c"]["pages"] = 1
        res["pages"][pg] = {"kind": ref["kind"], **s}
    return res


def selftest_folder(folder: str) -> dict:
    """Score any labeller's answers (FOLDER/PAGE.json, the labeller answer format) as if they were our output, on
    Azure's lines like --selftest. A page with no answer counts as an empty page (every line lost)."""
    folder = Path(folder)
    res = {"name": f"selftest_{folder.name}", "set": SET, "config": {"typer": f"labels in {folder}", "reader": "azure"}, "pages": {}, "errors": []}
    for pg, ref in load_reference().items():
        f = folder / f"{pg}.json"
        blocks = json.loads(f.read_text(encoding="utf-8"))["blocks"] if f.exists() else []
        if not f.exists():
            res["errors"].append(f"{pg}: no answer")
        s = score_page(labelling_as_master({**ref, "blocks": {**ref["blocks"], "_labeller": blocks}}, "_labeller"), ref)
        s["c"]["pages"] = 1
        res["pages"][pg] = {"kind": ref["kind"], **s}
    return res


def typer_on_azure(spec: str, ends: bool = True) -> dict:
    """The structure step alone: Azure's own lines (boxes, text, order) handed to the typer, page by page, and the
    blocks it makes scored like one of our master files. No line finding, no reading, none of parse.py's rules
    (the typer's own list rule and paragraph ends do run). Page size from the labeller input's picture."""
    import numpy as np
    from PIL import Image
    from ocrl.structure.tag import TO_TYPE, Typer
    typer = Typer(spec)
    res = {"name": f"azure_{SET}", "set": SET, "config": {"typer": spec, "reader": "azure lines"}, "pages": {}, "errors": []}
    for pg, ref in load_reference().items():
        folder = LAB / SETS[SET]["inputs"] / pg
        with Image.open(next(folder.glob("page_*.jpg"))) as im:
            W, H = im.width - 70, im.height
        ls = ref["lines"]
        boxes = np.array([[l["box"][0] * W / 1000, l["box"][1] * H / 1000, l["box"][2] * W / 1000, l["box"][3] * H / 1000] for l in ls], np.float32).reshape(-1, 4)
        typer.open_item = False
        roles, starts, _ = typer.tag(boxes, [l["text"] for l in ls], W, H) if ls else ([], [], None)
        blocks, prev = [], None
        for l, r, st in zip(ls, roles, starts):
            t = TO_TYPE.get(r, "other")
            if blocks and t == prev and (st < 0.5 or t == "table") and t not in ASIDE:
                blocks[-1]["lines"].append(l["id"])
            else:
                blocks.append({"type": t, "lines": [l["id"]]})
            prev = t
        s = score_page(labelling_as_master({**ref, "blocks": {**ref["blocks"], "_typer": blocks}}, "_typer"), ref)
        s["c"]["pages"] = 1
        res["pages"][pg] = {"kind": ref["kind"], **s}
    return res


def load(name: str) -> dict:
    """A saved run; also switches to the test set it was scored on (runs saved before --set existed are \"dev\")."""
    res = json.loads((E2E / f"{name}.json").read_text(encoding="utf-8"))
    use_set(res.get("set", "dev"))
    return res


def _cap_memory(gb: float = 12.0) -> None:
    """The laptop has 15 GB and has crashed from memory: never let this process take more than ~12 GB."""
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    want = int(gb * 1024 ** 3)
    if soft == resource.RLIM_INFINITY or soft > want:
        resource.setrlimit(resource.RLIMIT_AS, (want, hard))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", help="name of this run: results go to out/e2e/NAME.json")
    ap.add_argument("--set", default="dev", choices=sorted(SETS), help="which test set: the 70 development pages (dev) or the untouched check pages (check)")
    ap.add_argument("--typer", default=TYPER, help="structure model checkpoint ('' for the hand-written rules)")
    ap.add_argument("--reader", default=READER, help="line reader checkpoint ('' for the cluster reader)")
    ap.add_argument("--linefinder", default=LINEFINDER)
    ap.add_argument("--ckpt", default=CKPT, help="base (cluster) recogniser checkpoint")
    ap.add_argument("--pages", nargs="*", help="only these pages (names as in out/labeller/dev_inputs)")
    ap.add_argument("--kinds", nargs="*", help="only pages of these kinds (as the summary names them)")
    ap.add_argument("--lines-from", help="reuse the lines an earlier run read; only what comes after reading is run")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--src", help="import ocrl from this folder (one holding ocrl/) instead of src/: e.g. a frozen copy, "
                                  "made with: mkdir -p DIR && git archive HEAD src/ocrl | tar -x -C DIR --strip-components=1")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="print two saved runs side by side")
    ap.add_argument("--show", metavar="NAME", help="print a saved run's summary")
    ap.add_argument("--rescore", metavar="NAME", help="score a saved run's masters again")
    ap.add_argument("--selftest", nargs="?", const="", metavar="JUDGE", help="score a judge's own labels as our output: checks the ruler")
    ap.add_argument("--selftest-folder", metavar="DIR", help="score a labeller's answers (DIR/PAGE.json, e.g. out/labeller/dev_qwenft) as our output, on Azure's lines")
    ap.add_argument("--azure-typer", metavar="SPEC", help="the structure step alone on Azure's lines (see typer_on_azure); saved as --name")
    a = ap.parse_args()
    use_set(a.set)
    if a.azure_typer:
        if a.src:
            sys.path.insert(0, str(Path(a.src).resolve()))
        r = typer_on_azure(a.azure_typer); r["name"] = a.name or r["name"]; save(r); print(summary(r)); return
    if a.selftest_folder:
        r = selftest_folder(a.selftest_folder); save(r); print(summary(r)); return
    if a.selftest is not None:
        r = selftest(a.selftest); save(r); print(summary(r)); return
    if a.compare:
        ra, rb = load(a.compare[0]), load(a.compare[1])
        if ra.get("set", "dev") != rb.get("set", "dev"):
            ap.error(f"{a.compare[0]} was scored on the {ra.get('set', 'dev')} set and {a.compare[1]} on the {rb.get('set', 'dev')} set")
        print(compare(ra, rb)); return
    if a.show:
        print(summary(load(a.show))); return
    if a.rescore:
        print(summary(rescore(a.rescore))); return
    if not a.name:
        ap.error("give --name (or --compare / --show / --rescore)")
    _cap_memory()
    if a.src:
        sys.path.insert(0, str(Path(a.src).resolve()))
    import cv2, torch
    torch.set_num_threads(a.threads); cv2.setNumThreads(min(a.threads, 4))
    pages = a.pages
    if a.kinds:
        refs = load_reference()
        pages = [p for p, r in refs.items() if r["kind"] in a.kinds and (not a.pages or p in a.pages)]
    os.chdir(ROOT)
    pipe = LocalPipeline(reader=a.reader, typer=a.typer, linefinder=a.linefinder, ckpt=a.ckpt, load_reader=not a.lines_from)
    res = run(a.name, pipe, pages, a.lines_from)
    print()
    print(summary(res))


if __name__ == "__main__":
    main()
