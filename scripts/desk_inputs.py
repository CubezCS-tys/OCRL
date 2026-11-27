"""Labeller inputs for the pages of the owner's Desk database (labelled earlier by Gemini), one folder per page.

Two steps, on two machines:
    export  (laptop)  read the database: the labelled pages of every current version, minus the locked documents,
                      the inkscript_v2 test documents and the documents holding a judged development page;
                      writes one JSON line per document with each page's size and Azure lines in the database's order.
    draw    (machine) for each document: download its PDF from a short-lived link, draw each wanted page as a
                      labeller input (scan about 1150 px wide + 70 px white strip, every line outlined in blue with
                      its index in red), delete the PDF. Up to --workers documents at a time, so disk use stays small.

Line ids are "PAGE:INDEX" with INDEX = the database's azure line_no - 1 (line_no runs 1..n on every page);
each line in lines.json also carries "line_no", the database's own number, so labels map straight back.
Boxes are thousandths of the page, from the database's x, y, w, h over the page's width_px, height_px.
Pictures are drawn from the PDF's crop box (the frame Azure's coordinates are in).

    uv run python scripts/desk_inputs.py export out/labeller/desk_pages.jsonl.gz
    python scripts/desk_inputs.py draw desk_pages.jsonl.gz links.tsv out/labeller/desk_inputs --workers 32
"""
import argparse, gzip, json, os, shutil, sqlite3, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

DB = "/home/yassine/data/inkscript_labelled/data/desk/releases/R20261104T151029.db"


def export(out: str):
    locked = set(json.load(open("data/locked_exam_ids.json")))
    test = set(Path("/home/yassine/inkscript_v2/data/testset/ids.txt").read_text().split())
    dev = {p.name.rsplit("_p", 1)[0] for p in Path("out/labeller/dev_inputs").iterdir()}
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    vers = db.execute("select v.stem, v.version_id, d.legacy_only from versions v join documents d on d.stem=v.stem where v.is_current=1 order by v.stem").fetchall()
    why = {"locked exam": 0, "inkscript_v2 test": 0, "judged dev page": 0, "no Azure lines (legacy PDF only)": 0}
    pages_skipped = {"over 400 lines": 0}
    docs = pages = lines = 0
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for stem, vid, legacy in vers:
            k = "locked exam" if stem in locked else "inkscript_v2 test" if stem in test else "judged dev page" if stem in dev else "no Azure lines (legacy PDF only)" if legacy else None
            if k:
                why[k] += 1; continue
            wanted = [p for (p,) in db.execute("select distinct page from line_map where version_id=? order by page", (vid,))]
            if not wanted:
                why["no Azure lines (legacy PDF only)"] += 1; continue
            size = {p: (w, h, a) for p, w, h, a in db.execute("select page, width_px, height_px, angle from pages where stem=?", (stem,))}
            ps = []
            for p in wanted:
                ls = db.execute("select line_no, x, y, w, h, text from azure_lines where stem=? and page=? order by line_no", (stem, p)).fetchall()
                if len(ls) > 400:
                    pages_skipped["over 400 lines"] += 1; continue
                assert [l[0] for l in ls] == list(range(1, len(ls) + 1)), (stem, p)
                w, h, ang = size[p]
                ps.append({"page": p, "w": w, "h": h, "angle": ang, "lines": [[l[1], l[2], l[3], l[4], l[5] or ""] for l in ls]})
            f.write(json.dumps({"stem": stem, "pages": ps}, ensure_ascii=False) + "\n")
            docs += 1; pages += len(ps); lines += sum(len(p["lines"]) for p in ps)
    print(f"{docs} documents, {pages} pages, {lines} lines | documents skipped: {why} | pages skipped: {pages_skipped}")


def draw_doc(args):
    doc, url, dest = args
    import cv2
    cv2.setNumThreads(1)
    out = Path(dest); made = 0; odd = []
    todo = [p for p in doc["pages"] if not (out / f"{doc['stem']}_p{p['page']}" / "lines.json").exists()]
    if not todo:
        return doc["stem"], len(doc["pages"]), 0, "already"
    with tempfile.TemporaryDirectory(dir=os.environ.get("DESK_TMP")) as tmp:
        pdf = Path(tmp) / "d.pdf"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=180) as r, open(pdf, "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
                break
            except Exception as e:
                pdf.unlink(missing_ok=True); err = f"download: {type(e).__name__}"; time.sleep(3)
        else:
            return doc["stem"], 0, 0, err
        nbytes = pdf.stat().st_size
        for pg in todo:
            p = pg["page"]
            try:
                subprocess.run(["pdftoppm", "-cropbox", "-f", str(p), "-l", str(p), "-scale-to-x", "1150", "-scale-to-y", "-1", "-gray", "-png", str(pdf), f"{tmp}/p"],
                               check=True, stderr=subprocess.DEVNULL, timeout=300)
                f = next(Path(tmp).glob("p-*.png"))
                img = cv2.imread(str(f)); f.unlink()
                if abs(img.shape[0] / img.shape[1] - pg["h"] / pg["w"]) > 0.01 * pg["h"] / pg["w"]:
                    odd.append(p)  # the drawn page's shape disagrees with Azure's page: wrong frame; leave the page out
                    continue
                if img.shape[1] != 1150:
                    img = cv2.resize(img, (1150, round(img.shape[0] * 1150 / img.shape[1])), interpolation=cv2.INTER_AREA)
                img = cv2.copyMakeBorder(img, 0, 0, 0, 70, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                sx, sy = 1150 / pg["w"], img.shape[0] / pg["h"]
                d = out / f"{doc['stem']}_p{p}"; d.mkdir(parents=True, exist_ok=True); lines = []
                for i, (x, y, w, h, text) in enumerate(pg["lines"]):
                    x0, y0, x1, y1 = x, y, x + w, y + h
                    cv2.rectangle(img, (int(x0 * sx), int(y0 * sy)), (int(x1 * sx), int(y1 * sy)), (200, 120, 0), 1)
                    cv2.putText(img, str(i), (int(x1 * sx) + 3, int((y0 + y1) / 2 * sy) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 220), 1, cv2.LINE_AA)
                    lines.append({"id": f"{p}:{i}", "line_no": i + 1, "box": [round(1000 * x0 / pg["w"]), round(1000 * y0 / pg["h"]), round(1000 * x1 / pg["w"]), round(1000 * y1 / pg["h"])], "text": text})
                cv2.imwrite(str(d / f"page_{p}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                (d / "lines.json").write_text(json.dumps(lines, ensure_ascii=False), encoding="utf-8"); made += 1
            except Exception:
                pass
        return doc["stem"], made, nbytes, "ok" if made == len(todo) else f"{len(todo) - made} pages failed" + (f" ({len(odd)} with the wrong shape: {odd[:5]})" if odd else "")


def draw(pages: str, links: str, dest: str, workers: int, min_free_gb: float):
    from multiprocessing import Pool
    url = {}
    for l in open(links):
        if l.strip():
            i, ext, u = l.rstrip("\n").split("\t")
            if ext == "pdf": url[i] = u
    docs = [json.loads(l) for l in gzip.open(pages, "rt", encoding="utf-8")]
    jobs = [(d, url[d["stem"]], dest) for d in docs if d["stem"] in url]
    print(f"{len(jobs)} documents with a link, {sum(len(j[0]['pages']) for j in jobs)} pages; {len(docs) - len(jobs)} without a link", flush=True)
    Path(dest).mkdir(parents=True, exist_ok=True)
    made = nbytes = n = 0; bad = {}; t = time.time()
    with Pool(workers, maxtasksperchild=20) as pool:
        for stem, m, b, note in pool.imap_unordered(draw_doc, jobs):
            n += 1; made += m; nbytes += b
            if note not in ("ok", "already"): bad[stem] = note
            if n % 50 == 0:
                free = shutil.disk_usage(dest).free / 1e9
                print(f"{n}/{len(jobs)} documents, {made} pages drawn, {nbytes / 1e9:.2f} GB downloaded, {len(bad)} with problems, {free:.1f} GB free, {time.time() - t:.0f} s", flush=True)
                if free < min_free_gb:
                    print(f"STOPPING: only {free:.1f} GB free", flush=True); pool.terminate(); break
    print(f"finished: {n} documents, {made} pages drawn, {nbytes / 1e9:.2f} GB downloaded, {time.time() - t:.0f} s")
    for s, v in list(bad.items())[:30]: print(" ", s, v)
    json.dump(bad, open(Path(dest).parent / "desk_draw_problems.json", "w"), indent=0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("what", choices=["export", "draw"]); ap.add_argument("files", nargs="+")
    ap.add_argument("--workers", type=int, default=32); ap.add_argument("--min-free-gb", type=float, default=9.0)
    a = ap.parse_args()
    if a.what == "export":
        export(a.files[0])
    else:
        draw(*a.files[:3], a.workers, a.min_free_gb)
