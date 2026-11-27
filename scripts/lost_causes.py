"""Where do Azure's lines go missing? For every Azure line that no line of ours holds, the stage that lost it.

Each page is read by the whole pipeline (OCR.read_page: cleaning, line finder, regions, layout, line reader).
An Azure line no final line of ours is linked to (line_diag's link: one's centre in the other's middle band)
is put down to the first stage that lost it:
  faint on the page    hardly any dark ink in Azure's box on the page as drawn (show-through, a light watermark,
                       text Azure read from a picture's tones)
  picture mask         the cleaner painted the ink out as a photograph (its centre is in one of Cleaned.pictures)
  ruled line           the ink is there when ruled lines are kept, gone when they are painted out
  other cleaning       the cleaner removed the ink some other way (scanner border, a panel turned round)
  finder never marked  ink is there after cleaning, but no box of the line finder holds the line
                       (with "raw page would" when the finder marks it on the uncleaned page)
  dropped by layout    the finder marked it, but no line came out of layout (too small, a speck, no ink)
  read as empty        a line of ours holds it, but the reader read nothing
Lines matched one-to-one also give text agreement with Azure (folded, vowel marks dropped).

    uv run python scripts/lost_causes.py --pick out/lost/pages.json          # choose the fixed page list once
    uv run python scripts/lost_causes.py --pages out/lost/pages.json --name base [--draw out/lost/base_draw]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse, json, random, subprocess, sys, time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from line_diag import cases  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "out/lost/pages"
READER = "/home/yassine/data/inkscript_labelled/experiments/89_final_reader/out/weights/v7/last.pt"
CAUSES = ["faint on the page", "picture mask", "ruled line", "other cleaning", "finder never marked", "dropped by layout", "read as empty"]


def azure(doc_dir: Path, doc: str) -> dict:
    d = json.loads((doc_dir / doc / f"{doc}.json").read_text(encoding="utf-8"))
    return d.get("analyzeResult", d)


def page_image(doc_dir: Path, doc: str, page: int) -> np.ndarray:
    f = CACHE / doc / f"p{page:03d}.png"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["pdftoppm", "-cropbox", "-r", "300", "-gray", "-png", "-singlefile", "-f", str(page), "-l", str(page),
                        str(doc_dir / doc / f"{doc}.pdf"), str(f)[:-4]], check=True, stderr=subprocess.DEVNULL, timeout=300)
    return cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)


def pick(out: str, seed: int = 0) -> None:
    """Per unseen document: its most table-like page (most Azure lines with another level beside them), and one
    page at random. Plus the pages the by-eye review named."""
    rnd = random.Random(seed)
    docs = Path("/home/yassine/data/unseen_docs")
    pages = []
    for d in sorted(p.name for p in docs.iterdir()):
        try:
            az = azure(docs, d)
        except Exception:
            continue
        if not (docs / d / f"{d}.pdf").exists():
            continue
        scores = []
        for pi, pg in enumerate(az["pages"][:40], 1):
            bx = [(min(l["polygon"][0::2]), min(l["polygon"][1::2]), max(l["polygon"][0::2]), max(l["polygon"][1::2])) for l in pg.get("lines", [])]
            if len(bx) < 5:
                continue
            side = sum(any(j != i and min(a[3], b[3]) - max(a[1], b[1]) > 0.5 * min(a[3] - a[1], b[3] - b[1]) and (b[0] > a[2] or a[0] > b[2]) for j, b in enumerate(bx)) for i, a in enumerate(bx))
            scores.append((side / len(bx), pi))
        if not scores:
            continue
        best = max(scores)
        chosen = {best[1]} if best[0] >= 0.3 else set()
        chosen.add(rnd.choice(scores)[1])
        for pi in sorted(chosen):
            pages.append({"dir": str(docs), "doc": d, "page": pi, "kind": "rows" if pi == best[1] and best[0] >= 0.3 else "any"})
    review = [("mandumah_s3", "0888-058-007,008-003", [2, 4]), ("mandumah_s3", "0236-016-003-003", [7]), ("mandumah_s3", "0138-011-013-012", [27]),
              ("judged_docs", "0674-021-003-007", [2]), ("mandumah_s3", "1594-000-013-006", [9, 11])]
    for folder, d, ps in review:
        base = Path("/home/yassine/data") / folder
        if not (base / d / f"{d}.pdf").exists():
            base = Path("/home/yassine/data/judged_docs") if folder == "mandumah_s3" else Path("/home/yassine/data/mandumah_s3")
        for pi in ps:
            pages.append({"dir": str(base), "doc": d, "page": pi, "kind": "review"})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(pages, open(out, "w"), indent=0)
    print(f"{len(pages)} pages -> {out}", Counter(p["kind"] for p in pages))


def ink_in(img: np.ndarray, b) -> int:
    x0, y0, x1, y1 = (int(round(v)) for v in b)
    return int((img[max(y0, 0) : y1, max(x0, 0) : x1] < 128).sum())


def linked(az_boxes, ours) -> list[bool]:
    """Is each Azure line linked to any of `ours`?"""
    if not len(ours):
        return [False] * len(az_boxes)
    return [c != "missing" for c in cases([tuple(o) for o in ours], az_boxes)]


def run(a) -> None:
    import torch
    from rapidfuzz.distance import Levenshtein
    from ocrl.blobs.clean import clean_page
    from ocrl.pipeline import OCR
    from ocrl.recog.realink import fold
    from ocrl.text import strip_marks

    torch.set_num_threads(4); cv2.setNumThreads(4)
    import importlib
    for kv in a.set:  # module.NAME=value, e.g. ocrl.blobs.clean.DRAWINGS=1
        k, v = kv.split("=")
        mod, name = k.rsplit(".", 1)
        old = getattr(importlib.import_module(mod), name)
        setattr(importlib.import_module(mod), name, type(old)(float(v)) if not isinstance(old, bool) else bool(int(v)))
        print("set", k, "=", getattr(importlib.import_module(mod), name))
    norm = lambda t: " ".join(strip_marks(" ".join(fold(w) for w in t.split())).split())
    ocr = OCR(str(ROOT / "out/recognizer_v3.pt"), linefinder=str(ROOT / a.finder), line_reader=None if a.no_read else READER)
    pages = json.load(open(a.pages))
    if a.only:
        pages = [p for p in pages if p["kind"] in a.only.split(",")]
    tot, by_kind, txt = Counter(), {}, Counter()
    rows = []
    examples: dict[str, list] = {c: [] for c in CAUSES}
    t0 = time.time()
    for p in pages:
        gray = page_image(Path(p["dir"]), p["doc"], p["page"])
        pg = azure(Path(p["dir"]), p["doc"])["pages"][p["page"] - 1]
        sx, sy = gray.shape[1] / pg["width"], gray.shape[0] / pg["height"]
        az = [(min(l["polygon"][0::2]) * sx, min(l["polygon"][1::2]) * sy, max(l["polygon"][0::2]) * sx, max(l["polygon"][1::2]) * sy) for l in pg.get("lines", [])]
        az_text = [l["content"] for l in pg.get("lines", [])]
        try:
            doc = ocr.read_page(gray)
        except Exception:
            import traceback; traceback.print_exc(); tot["crashed"] += 1
            continue
        if getattr(doc, "turned", 0):  # read turned upright: put Azure's boxes on the turned page too
            k = doc.turned % 4
            H0, W0 = gray.shape
            for _ in range(k):  # one quarter turn counter-clockwise: (x, y) -> (y, W - x)
                az = [(y0, W0 - x1, y1, W0 - x0) for x0, y0, x1, y1 in az]
                H0, W0 = W0, H0
        final, texts, lwarn = [], [], []
        for r, res in doc.regions:
            ox, oy = r.x0 - 16, r.y0 - 16
            for ln in res.lines:
                if not ln.clusters:
                    continue
                bx = res.layout.boxes[[b for c in ln.clusters for b in c.blobs]]
                final.append((bx[:, 0].min() + ox, bx[:, 1].min() + oy, bx[:, 2].max() + ox, bx[:, 3].max() + oy)); texts.append(ln.text)
                lwarn.append(list(getattr(ln, "warnings", [])))
        for w in lwarn:
            for x in w:
                tot["line warning: " + x] += 1
        cl = doc.cleaned
        read = [f for f, t in zip(final, texts) if t.strip()]
        cs_all = cases([tuple(o) for o in final], az) if final else ["missing"] * len(az)
        cs_read = cases([tuple(o) for o in read], az) if read else ["missing"] * len(az)
        lost = [j for j, c in enumerate(cs_read) if c == "missing"]
        per = Counter(cs_read)
        cause = {}
        if getattr(doc, "turned", 0):
            gray = np.ascontiguousarray(np.rot90(gray, doc.turned))
        if lost:
            kept = clean_page(gray, drop_rules=False).gray
            found = doc.found if doc.found is not None else np.zeros((0, 4))
            f_link = linked(az, found)
            raw_link = None
            for j in lost:
                b = az[j]
                raw_ink, clean_ink = ink_in(gray, b), ink_in(cl.gray, b)
                area = max((b[2] - b[0]) * (b[3] - b[1]), 1)
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                if raw_ink < 0.01 * area:
                    c = "faint on the page"
                elif clean_ink < 0.35 * raw_ink:
                    if any(x0 - 30 <= cx <= x1 + 30 and y0 - 30 <= cy <= y1 + 30 for x0, y0, x1, y1 in cl.pictures):
                        c = "picture mask"
                    elif ink_in(kept, b) >= 0.6 * raw_ink:
                        c = "ruled line"
                    else:
                        c = "other cleaning"
                elif not f_link[j]:
                    c = "finder never marked"
                    if raw_link is None:
                        raw_link = linked(az, ocr.finder.lines(gray))
                    if raw_link[j]:
                        tot["finder: raw page would mark it"] += 1
                elif cs_all[j] == "missing":
                    c = "dropped by layout"
                else:
                    c = "read as empty"
                cause[j] = c
                tot[c] += 1
                by_kind.setdefault(p["kind"], Counter())[c] += 1
                if len(examples[c]) < 400:
                    examples[c].append((p["doc"], p["page"], az_text[j][:40]))
        # text agreement on lines one-to-one
        t_before = (txt["chars"], txt["edits"])
        for j, c in enumerate(cs_read):
            if c != "one-to-one":
                continue
            z = az[j]
            zcx, zcy = (z[0] + z[2]) / 2, (z[1] + z[3]) / 2
            k = next((i for i, o in enumerate(read) if o[0] - 4 <= zcx <= o[2] + 4 and o[1] <= zcy <= o[3]), None)
            if k is None:
                continue
            want = norm(az_text[j]); got = norm([t for t in texts if t.strip()][k])
            if want:
                e = Levenshtein.distance(got, want)
                txt["chars"] += len(want); txt["edits"] += e
                ws = [w for w, t in zip(lwarn, texts) if t.strip()][k]
                tag = "warned" if ws else "clean"
                txt[tag + "_chars"] += len(want); txt[tag + "_edits"] += e
        for c, n in per.items():
            tot["case:" + c] += n
        by_kind.setdefault(p["kind"], Counter())["lines"] += len(az)
        by_kind[p["kind"]]["lost"] += len(lost)
        tot["lines"] += len(az); tot["lost"] += len(lost); tot["ours"] += len(final); tot["ours_empty"] += len(final) - len(read)
        rows.append({"doc": p["doc"], "page": p["page"], "kind": p["kind"], "azure": len(az), "ours": len(final), "lost": len(lost),
                     "text_chars": txt["chars"] - t_before[0], "text_edits": txt["edits"] - t_before[1], "cases": dict(per), "causes": dict(Counter(cause.values())), "warnings": list(getattr(doc, "warnings", []))})
        if not a.quiet:
            print(f"{p['doc']} p{p['page']:<3d} {p['kind']:6s} azure {len(az):4d} ours {len(final):4d} lost {len(lost):3d} " +
                  " ".join(f"{k}={v}" for k, v in Counter(cause.values()).most_common()) + f" | {getattr(doc, 'warnings', [])} {doc.cleaned.notes}", flush=True)
        if a.draw and (lost or a.draw_all):
            Path(a.draw).mkdir(parents=True, exist_ok=True)
            img = cv2.cvtColor(cl.gray, cv2.COLOR_GRAY2BGR)
            for x0, y0, x1, y1 in cl.pictures:
                cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 200, 255), 3)
            for o in final:
                cv2.rectangle(img, (int(o[0]), int(o[1])), (int(o[2]), int(o[3])), (160, 90, 0), 2)
            colour = dict(zip(CAUSES, [(128, 128, 128), (0, 160, 255), (200, 0, 200), (0, 200, 200), (0, 0, 255), (0, 120, 0), (255, 0, 0)]))
            for j, c in cause.items():
                b = az[j]
                cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), colour[c], 5)
            s = 1400 / img.shape[1]
            cv2.imwrite(str(Path(a.draw) / f"{p['doc']}_p{p['page']:03d}.jpg"), cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 70])
    L = max(tot["lines"], 1)
    print(f"\n{len(pages)} pages, {tot['lines']} Azure lines, {time.time() - t0:.0f} s")
    print(f"one-to-one {100 * tot['case:one-to-one'] / L:.1f}% | joins a row {100 * tot['case:ours joins a row'] / L:.1f}% | joins lines {100 * tot['case:ours joins lines'] / L:.1f}% "
          f"| splits {100 * tot['case:ours splits'] / L:.1f}% | tangled {100 * tot['case:tangled'] / L:.1f}% | LOST {100 * tot['lost'] / L:.2f}%")
    print(f"text differing from Azure on one-to-one lines: {100 * txt['edits'] / max(txt['chars'], 1):.2f}% of {txt['chars']} characters | our lines read empty: {tot['ours_empty']}")
    print(f"   one-to-one lines with a warning: {100 * txt['warned_edits'] / max(txt['warned_chars'], 1):.1f}% differ ({txt['warned_chars']} chars); without: {100 * txt['clean_edits'] / max(txt['clean_chars'], 1):.2f}%")
    print("   line warnings:", {k[14:]: v for k, v in tot.items() if k.startswith("line warning: ")})
    print("lost lines by cause:")
    for c in CAUSES:
        print(f"   {c:22s} {tot[c]:5d}  ({100 * tot[c] / max(tot['lost'], 1):4.1f}% of lost)")
    print(f"   (finder never marked, but would on the raw page: {tot['finder: raw page would mark it']})")
    for k, v in sorted(by_kind.items()):
        print(f"   kind {k:6s}: {v['lines']} lines, lost {100 * v['lost'] / max(v['lines'], 1):.2f}% | " + " ".join(f"{c}={v[c]}" for c in CAUSES if v[c]))
    if a.name:
        out = ROOT / "out/lost" / f"{a.name}.json"
        json.dump({"total": dict(tot), "text": dict(txt), "by_kind": {k: dict(v) for k, v in by_kind.items()}, "pages": rows, "examples": examples}, open(out, "w"), ensure_ascii=False, indent=1)
        print("->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", default=""); ap.add_argument("--pages", default=""); ap.add_argument("--name", default="")
    ap.add_argument("--draw", default=""); ap.add_argument("--draw-all", action="store_true"); ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--finder", default="out/linefinder.pt"); ap.add_argument("--no-read", action="store_true", help="skip the line reader (no 'read as empty', no text)")
    ap.add_argument("--set", nargs="*", default=[], help="module switches: ocrl.blobs.clean.DRAWINGS=1")
    ap.add_argument("--only", default="", help="kinds of page to run, e.g. rows,review")
    a = ap.parse_args()
    if a.pick:
        pick(a.pick)
    else:
        run(a)
