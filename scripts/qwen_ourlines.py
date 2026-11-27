"""Training examples for the Qwen structure step built from OUR lines (the line finder and reader the product runs), with
Haiku's labels (made on Azure's lines) carried over to them.

Why: the adapter was trained only on Azure's lines; inside the pipeline it sees ours, which are cut differently (a table
cell or a hanging list mark may be its own line, neighbouring cells may be joined, specks become lines, a few lines are
lost) and read a little differently. These examples show it our lines.

For one labelled page (PDF + Azure lines.json + Haiku's {"blocks": [...]}):
  1. the page is drawn as the product draws it (crop box, 300 dpi, grey; scripts/e2e_judged.draw) and read as
     ocrl.doc.parse.parse reads it before typing: read_page, then unread_marks. The lines, sorted by (region, order)
     as QwenTyper.prepare sorts them, become the Qwen input through structure/qwen.py's own page_picture and
     page_lines (written as page_N.jpg + lines.json, the folder format scripts/qwen_finetune.py reads).
  2. Haiku's labels are carried over (transfer()):
     - each Azure line goes to the line of ours holding its centre (the smallest such, 0.02 inch of slack: the rule
       scripts/e2e_judged.py scores by). An Azure line in none is "lost".
     - a line of ours holding Azure lines takes their block; holding lines of two or more blocks ("straddles"), the
       block with the most characters (Azure's text) wins. The boundary cannot be kept: one line of ours can be in
       only one block. A block left with no line of ours disappears.
     - a line of ours holding no Azure line's centre: if at least half of its box lies inside one Azure line's box, it
       is a piece of that line (a list mark or a cell cut off from its neighbour) and joins that line's block, placed
       beside it in reading direction; otherwise, if its centre lies inside the area of a table or figure block, it
       joins that block, next to the nearest line of it; otherwise it is "noise", a block of its own, placed after
       the block of the nearest labelled line.
     - reading order: Haiku's block order; inside a block, our lines in Haiku's order of the Azure lines they hold.
  3. The page is also written in its Azure-lines form (Azure's boxes drawn on the same 300 dpi picture with the same
     page_picture; Haiku's answer unchanged), for the share of training kept on Azure's lines.
Pages where the carrying over looks unreliable are written but flagged (stats "flag"): more than 15% of Azure lines
lost, more than a third of our lines noise, or more than 20% of our lines straddling (a misplaced page frame).

    python scripts/qwen_ourlines.py build --pages pages.tsv --out /work/data/ourlines --workers 8
        pages.tsv: name <TAB> pdf <TAB> azure lines.json <TAB> haiku answer   (name as DOC_pN)
    python scripts/qwen_ourlines.py stats --out /work/data/ourlines
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, tempfile, time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DPI, SLACK = 300, 0.02
READER = "/home/yassine/data/inkscript_labelled/experiments/89_final_reader/out/weights/v7/last.pt"


# ------------------------------------------------------------------------------------------------ carrying labels over
def _inside(o, cx, cy, sx, sy):
    return o[0] - sx <= cx <= o[2] + sx and o[1] - sy <= cy <= o[3] + sy


def transfer(az: list[dict], ans: dict, ours: list[dict], w_in: float, h_in: float, ltr: bool) -> tuple[dict, dict]:
    """Haiku's answer on Azure's lines -> an answer on our lines. `az` and `ours` as lines.json lists them (id, box in
    thousandths, text). Returns ({"blocks": [...]}, stats)."""
    sx, sy = 1000 * SLACK / w_in, 1000 * SLACK / h_in
    st = Counter(az=len(az), ours=len(ours))
    blk, pos = {}, {}  # Azure id -> Haiku block index, place in Haiku's reading order
    for k, b in enumerate(ans["blocks"]):
        for r in b["lines"]:
            blk[r] = k; pos[r] = len(pos)
    abox = {l["id"]: l["box"] for l in az}
    held = defaultdict(list)  # our index -> Azure ids whose centre it holds
    for l in az:
        x0, y0, x1, y1 = l["box"]; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hits = [i for i, o in enumerate(ours) if _inside(o["box"], cx, cy, sx, sy)]
        if not hits:
            st["az_lost"] += 1; continue
        i = min(hits, key=lambda i: (ours[i]["box"][2] - ours[i]["box"][0]) * (ours[i]["box"][3] - ours[i]["box"][1]))
        held[i].append(l["id"])
    text = {l["id"]: l["text"] for l in az}
    got = {}  # our index -> (block, key)
    for i, ids in held.items():
        ids = [r for r in ids if r in blk]
        if not ids:
            continue
        ch = Counter()
        for r in ids:
            ch[blk[r]] += max(len(text[r].strip()), 1)
        if len(ch) > 1:
            st["straddle"] += 1
            st["straddle_types"] += len({ans["blocks"][k]["type"] for k in ch}) > 1
        k = max(ch, key=lambda k: (ch[k], -min(pos[r] for r in ids if blk[r] == k)))
        got[i] = (k, min(pos[r] for r in ids if blk[r] == k))
    # lines holding no Azure line's centre
    def cxcy(b):
        return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    area_of = {}
    for k, b in enumerate(ans["blocks"]):
        if b["type"] in ("table", "figure") and b["lines"]:
            bs = [abox[r] for r in b["lines"] if r in abox]
            if bs:
                area_of[k] = (min(x[0] for x in bs) - sx, min(x[1] for x in bs) - sy, max(x[2] for x in bs) + sx, max(x[3] for x in bs) + sy)
    noise = []
    for i, o in enumerate(ours):
        if i in got:
            continue
        st["unheld"] += 1
        b = o["box"]; a = max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
        best, bo = None, 0.0
        for l in az:
            q = l["box"]
            ov = max(0, min(b[2], q[2] + sx) - max(b[0], q[0] - sx)) * max(0, min(b[3], q[3] + sy) - max(b[1], q[1] - sy))
            if ov / a > bo:
                best, bo = l["id"], ov / a
        if best is not None and bo >= 0.5 and best in blk:
            # beside the line it was cut from: before it when it stands earlier in reading direction
            q = abox[best]; c, qc = cxcy(b)[0], cxcy(q)[0]
            before = c < qc if ltr else c > qc
            got[i] = (blk[best], pos[best] + (-0.4 if before else 0.4) + 0.001 * i)
            st["piece"] += 1; continue
        cx, cy = cxcy(b)
        inarea = [k for k, A in area_of.items() if A[0] <= cx <= A[2] and A[1] <= cy <= A[3]]
        if inarea:
            k = inarea[0]
            mates = [(j, g) for j, g in got.items() if g[0] == k]
            if mates:
                j, g = min(mates, key=lambda m: (cxcy(ours[m[0]]["box"])[0] - cx) ** 2 + 9 * (cxcy(ours[m[0]]["box"])[1] - cy) ** 2)
                got[i] = (k, g[1] + 0.3 + 0.001 * i)
            else:
                got[i] = (k, min(pos[r] for r in ans["blocks"][k]["lines"]) + 0.001 * i)
            st["in_area"] += 1; continue
        noise.append(i)
    # blocks in Haiku's order, then noise blocks after the block of the nearest labelled line
    by_blk = defaultdict(list)
    for i, (k, key) in got.items():
        by_blk[k].append((key, i))
    num = ours[0]["id"].split(":")[0] if ours else "0"
    out = []
    for k, b in enumerate(ans["blocks"]):
        if by_blk.get(k):
            out.append({"type": b["type"], "lines": [ours[i]["id"] for _, i in sorted(by_blk[k])], "_k": k})
    st["blocks_haiku"] = len(ans["blocks"]); st["blocks_vanished"] = len(ans["blocks"]) - len(out)
    for i in noise:
        cx, cy = cxcy(ours[i]["box"])
        if got:
            j = min(got, key=lambda j: (cxcy(ours[j]["box"])[0] - cx) ** 2 + 9 * (cxcy(ours[j]["box"])[1] - cy) ** 2)
            at = next(n for n, b in enumerate(out) if b.get("_k") == got[j][0]) + 1
        else:
            at = len(out)
        while at < len(out) and out[at].get("_k") is None:  # after earlier noise placed there
            at += 1
        out.insert(at, {"type": "noise", "lines": [ours[i]["id"]], "_k": None})
        st["noise"] += 1
    for b in out:
        b.pop("_k", None)
    used = [r for b in out for r in b["lines"]]
    assert sorted(used) == sorted(o["id"] for o in ours), (num, len(used), len(ours))
    st["flag"] = int(st["az_lost"] > 0.15 * max(st["az"], 1) or st["noise"] > max(3, len(ours) / 3) or st["straddle"] > max(3, 0.2 * len(ours)))
    return {"blocks": out}, dict(st)


# ------------------------------------------------------------------------------------------------ building
_OCR = None


def _init(threads: int):
    global _OCR
    os.environ["OMP_NUM_THREADS"] = str(threads)
    import cv2, torch
    torch.set_num_threads(threads); cv2.setNumThreads(threads)
    from ocrl.pipeline import OCR
    _OCR = OCR(str(ROOT / "out/recognizer_v3.pt"), linefinder=str(ROOT / "out/linefinder.pt"), line_reader=os.environ.get("OURLINES_READER", READER))


def _draw(pdf: str, number: int):
    import cv2
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-cropbox", "-r", str(DPI), "-gray", "-png", "-f", str(number), "-l", str(number), "-singlefile", pdf, f"{tmp}/p"],
                       check=True, capture_output=True, timeout=300)
        return cv2.imread(f"{tmp}/p.png", cv2.IMREAD_GRAYSCALE)


def one(row):
    name, pdf, azf, ansf, out = row
    out = Path(out)
    if (out / "stats" / f"{name}.json").exists():
        return name, "done"
    try:
        from ocrl.doc import parse as P
        from ocrl.structure.qwen import page_lines, page_picture
        number = int(name.rsplit("_p", 1)[1])
        t = time.time()
        gray = _draw(pdf, number)
        h, w = gray.shape
        info = P.read_page(_OCR, gray, number)
        if P.MARKS:
            P.unread_marks(info, gray)
        ls = sorted(info.lines, key=lambda l: (l.region, l.order))  # as QwenTyper.prepare
        boxes = [[l.x0, l.y0, l.x1, l.y1] for l in ls]; texts = [l.text for l in ls]
        az = json.loads(Path(azf).read_text(encoding="utf-8"))
        ans = json.loads(Path(ansf).read_text(encoding="utf-8"))
        latin = sum(c.isascii() and c.isalpha() for l in az for c in l["text"])
        ltr = latin > 0.5 * max(sum(c.isalpha() for l in az for c in l["text"]), 1)
        st = {"name": name, "w": w, "h": h, "n_ours": len(ls), "ltr": ltr}
        if ls:
            ours = page_lines(number, boxes, texts, w, h)
            tgt, s = transfer(az, ans, ours, w / DPI, h / DPI, ltr)
            st.update(s)
            d = out / "ours" / name; d.mkdir(parents=True, exist_ok=True)
            (d / f"page_{number}.jpg").write_bytes(page_picture(gray, boxes))
            (d / "lines.json").write_text(json.dumps(ours, ensure_ascii=False), encoding="utf-8")
            (out / "ours_ans").mkdir(parents=True, exist_ok=True)
            (out / "ours_ans" / f"{name}.json").write_text(json.dumps(tgt, ensure_ascii=False), encoding="utf-8")
        else:
            st["flag"] = 1
        # the Azure-lines form, drawn the same way
        d = out / "az" / name; d.mkdir(parents=True, exist_ok=True)
        azb = [[l["box"][0] * w / 1000, l["box"][1] * h / 1000, l["box"][2] * w / 1000, l["box"][3] * h / 1000] for l in az]
        (d / f"page_{number}.jpg").write_bytes(page_picture(gray, azb))
        (d / "lines.json").write_text(json.dumps(az, ensure_ascii=False), encoding="utf-8")
        (out / "az_ans").mkdir(parents=True, exist_ok=True)
        (out / "az_ans" / f"{name}.json").write_text(json.dumps(ans, ensure_ascii=False), encoding="utf-8")
        st["s"] = round(time.time() - t, 2)
        (out / "stats").mkdir(parents=True, exist_ok=True)
        (out / "stats" / f"{name}.json").write_text(json.dumps(st), encoding="utf-8")
        return name, "ok"
    except Exception as e:
        (out / "failed").mkdir(parents=True, exist_ok=True)
        (out / "failed" / f"{name}.txt").write_text(f"{type(e).__name__}: {e}", encoding="utf-8")
        return name, f"failed {type(e).__name__}: {e}"


def build(a):
    import multiprocessing as mp
    rows = [l.rstrip("\n").split("\t") + [a.out] for l in open(a.pages, encoding="utf-8") if l.strip()]
    ctx = mp.get_context("spawn")
    t0, c = time.time(), Counter()
    with ctx.Pool(a.workers, initializer=_init, initargs=(a.threads,), maxtasksperchild=400) as pool:
        for k, (name, r) in enumerate(pool.imap_unordered(one, rows, chunksize=4), 1):
            c[r.split()[0]] += 1
            if r.startswith("failed"):
                print(name, r, flush=True)
            if k % 200 == 0:
                print(f"{k}/{len(rows)} {dict(c)} {time.time() - t0:.0f} s", flush=True)
    print(f"finished {dict(c)} in {time.time() - t0:.0f} s", flush=True)


def stats(a):
    S = Counter(); pages = 0; flagged = []
    for f in sorted((Path(a.out) / "stats").glob("*.json")):
        s = json.loads(f.read_text()); pages += 1
        for k, v in s.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("w", "h", "s"):
                S[k] += v
        if s.get("flag"):
            flagged.append(s["name"])
    o = max(S["ours"], 1); z = max(S["az"], 1)
    print(f"{pages} pages, {len(flagged)} flagged; our lines {S['ours']}, Azure lines {S['az']}")
    print(f"  Azure lines lost (centre in no line of ours) {S['az_lost']} ({100 * S['az_lost'] / z:.1f}%)")
    print(f"  our lines holding no Azure centre {S['unheld']} ({100 * S['unheld'] / o:.1f}%): pieces {S['piece']} ({100 * S['piece'] / o:.1f}%), "
          f"in a table/figure area {S['in_area']} ({100 * S['in_area'] / o:.1f}%), noise {S['noise']} ({100 * S['noise'] / o:.1f}%)")
    print(f"  our lines straddling blocks {S['straddle']} ({100 * S['straddle'] / o:.1f}%), of which across types {S['straddle_types']} ({100 * S['straddle_types'] / o:.1f}%)")
    print(f"  Haiku blocks {S['blocks_haiku']}, vanished {S['blocks_vanished']} ({100 * S['blocks_vanished'] / max(S['blocks_haiku'], 1):.1f}%)")
    if a.list_flagged:
        print("\n".join(flagged))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--pages", required=True); b.add_argument("--out", required=True)
    b.add_argument("--workers", type=int, default=8); b.add_argument("--threads", type=int, default=4)
    s = sub.add_parser("stats"); s.add_argument("--out", required=True); s.add_argument("--list-flagged", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    build(a) if a.cmd == "build" else stats(a)
