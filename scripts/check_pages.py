"""The second, untouched test set: about 200 pages from documents this project never used for anything.

A document qualifies only if it is in none of: out/labeller/bulk_inputs and dev_inputs (labelled pages, structure_v3's
training), data/locked_exam_ids.json, the inkscript_v2 test set ids, the Desk database (the structure model's training
data, scripts/build_structure_data.py), data/reader/*_ids*.json (the line reader), data/realink/*_ids.json (the line
finder, scripts/build_lines_data.py), the census (out/census/pages.npz) and the judged documents. Journals (the first
four digits of an id) never seen in bulk_inputs or the Desk database come first; one document per journal.

    uv run python scripts/check_pages.py candidates --n 330          # ids to fetch -> out/labeller/check_candidates.json
    uv run python scripts/check_pages.py fetch --what json --workers 3   # Azure's JSON -> /home/yassine/data/check_docs/ID/
    uv run python scripts/check_pages.py select --pages 200          # pages spread over the 14 kinds -> out/labeller/check_pages.json
    uv run python scripts/check_pages.py fetch --what pdf --workers 3    # PDFs of the chosen documents only
    uv run python scripts/check_pages.py inputs                      # labeller inputs -> out/labeller/check_inputs/PAGE/

--set unseen: the same select and inputs steps on 110 documents from 110 other unseen journals (not the exam's), giving
out/labeller/unseen_pages.json and unseen_inputs. That set is a DEVELOPMENT set for rules on unseen journals (tune on it,
never train on it); the exam stays untouched.

THIS IS A TEST SET. Keep out/labeller/check_* and /home/yassine/data/check_docs out of every training input, and do
not tune anything on these pages.
"""
import argparse, json, sqlite3, subprocess, sys, tempfile, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "out/labeller"
DOCS = Path("/home/yassine/data/check_docs")
PFX = "check"  # --set unseen: PFX "unseen", DOCS /home/yassine/data/unseen_docs (a development set for rules, see the docstring)
IDS_MAIN = Path("/home/yassine/inkscript/experiments/44_bucket_map/out/ids_main.txt")
DESK = "/home/yassine/data/inkscript_labelled/data/desk/releases/R20261104T151029.db"
AWS = "/home/yassine/inkscript/.venv/bin/aws"
BUCKET = "mandumah-source-docs"
MAX_MB = 40
SEED = 0
journal = lambda s: s[:4]


# ------------------------------------------------------------------------------------------------ who was used
def used_ids() -> dict[str, set]:
    """Every id this project used, by where it was used."""
    S = {}
    S["bulk_inputs"] = {p.name.rsplit("_p", 1)[0] for p in (LAB / "bulk_inputs").iterdir()}
    S["dev_inputs"] = {p.name.rsplit("_p", 1)[0] for p in (LAB / "dev_inputs").iterdir()}
    S["locked"] = set(json.load(open(ROOT / "data/locked_exam_ids.json")))
    S["inkscript_v2_testset"] = set(Path("/home/yassine/inkscript_v2/data/testset/ids.txt").read_text().split())
    db = sqlite3.connect(f"file:{DESK}?mode=ro", uri=True)
    S["desk"] = {r[0] for r in db.execute("select distinct stem from versions")} | {r[0] for r in db.execute("select distinct stem from pages")}
    S["reader"] = set().union(*(set(json.load(open(f))) for f in (ROOT / "data/reader").glob("*_ids*.json")))
    S["realink_linefinder"] = set().union(*(set(json.load(open(f))) for f in (ROOT / "data/realink").glob("*_ids.json")))
    S["census"] = set(np.load(ROOT / "out/census/pages.npz")["stem"].tolist())
    S["judged_docs"] = {p.name for p in Path("/home/yassine/data/judged_docs").iterdir()}
    S["mandumah_s3"] = {p.name for p in Path("/home/yassine/data/mandumah_s3").iterdir()}
    return S


def candidates(n: int) -> None:
    S = used_ids()
    used = set().union(*S.values())
    seen_j = {journal(s) for s in S["bulk_inputs"] | S["desk"]}
    seen_any = {journal(s) for s in used}
    ids = [l.strip() for l in open(IDS_MAIN) if l.strip()]
    free = [i for i in ids if i not in used]
    by_j = defaultdict(list)
    for i in free:
        by_j[journal(i)].append(i)
    rng = np.random.default_rng(SEED)
    fresh_j = sorted(j for j in by_j if j not in seen_j)
    other_j = sorted(j for j in by_j if j in seen_j)
    rng.shuffle(fresh_j); rng.shuffle(other_j)
    # one document per journal: never-seen journals first (those not even in the reader's or census's ids before those)
    fresh_j.sort(key=lambda j: j in seen_any)
    pick = []
    for j in (fresh_j + other_j)[:n]:
        pick.append({"id": by_j[j][int(rng.integers(len(by_j[j])))], "journal": j, "journal_unseen": j not in seen_j, "journal_unseen_anywhere": j not in seen_any})
    out = {"seed": SEED, "made": time.strftime("%Y-%m-%d %H:%M"), "excluded": {k: len(v) for k, v in S.items()},
           "ids_listed": len(ids), "ids_free": len(free), "journals_free": len(by_j), "journals_never_seen": len(fresh_j), "candidates": pick}
    (LAB / f"{PFX}_candidates.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{len(ids):,} ids listed, {len(free):,} never used, in {len(by_j):,} journals ({len(fresh_j):,} never seen in bulk_inputs or Desk); "
          f"{len(pick)} candidates, {sum(p['journal_unseen'] for p in pick)} from never-seen journals")
    print("excluded:", out["excluded"])


# ------------------------------------------------------------------------------------------------ fetching
def _aws(*args, timeout=600):
    return subprocess.run([AWS, *args, "--only-show-errors"] if args[0] == "s3" and args[1] == "cp" else [AWS, *args],
                          capture_output=True, text=True, timeout=timeout)


def fetch_one(args):
    doc, what = args
    d = DOCS / doc
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{doc}.{what}"
    if f.exists() and f.stat().st_size > 0:
        return doc, "have", f.stat().st_size
    try:
        if what == "json":  # sizes of both files first: skip documents over MAX_MB, or with no PDF
            r = _aws("s3", "ls", f"s3://{BUCKET}/{doc}/", timeout=120)
            sizes = {ln.split()[-1]: int(ln.split()[-2]) for ln in r.stdout.splitlines() if len(ln.split()) >= 4}
            (d / "_sizes.json").write_text(json.dumps(sizes))
            if f"{doc}.json" not in sizes or f"{doc}.pdf" not in sizes:
                return doc, "missing in bucket", 0
            if sizes[f"{doc}.pdf"] > MAX_MB * 2 ** 20:
                return doc, f"pdf over {MAX_MB} MB", sizes[f"{doc}.pdf"]
        tmp = d / f".{doc}.{what}.part"
        r = _aws("s3", "cp", f"s3://{BUCKET}/{doc}/{doc}.{what}", str(tmp), timeout=900)
        if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            return doc, "refused or failed: " + (r.stderr.strip().splitlines() or ["?"])[-1][:160], 0
        tmp.rename(f)
        return doc, "fetched", f.stat().st_size
    except subprocess.TimeoutExpired:
        (d / f".{doc}.{what}.part").unlink(missing_ok=True)
        return doc, "stalled (timeout)", 0


def fetch(what: str, workers: int) -> None:
    if what == "json":
        docs = [c["id"] for c in json.loads((LAB / f"{PFX}_candidates.json").read_text())["candidates"]]
    else:
        docs = sorted({p["doc"] for p in json.loads((LAB / f"{PFX}_pages.json").read_text())["pages"]})
    t, got, log = time.time(), 0, []
    with ThreadPoolExecutor(workers) as ex:
        for k, (doc, st, n) in enumerate(ex.map(fetch_one, [(d, what) for d in docs]), 1):
            got += n if st == "fetched" else 0
            log.append({"doc": doc, "what": what, "status": st, "bytes": n})
            if st not in ("fetched", "have"):
                print(f"  {doc}: {st}", flush=True)
            if k % 20 == 0:
                print(f"[{k}/{len(docs)}] {got / 2 ** 20:.0f} MB in {time.time() - t:.0f} s", flush=True)
    with open(DOCS / "_fetch.jsonl", "a") as f:
        for r in log:
            f.write(json.dumps(r) + "\n")
    print(what, Counter(r["status"] if not r["status"].startswith("refused") else "refused or failed" for r in log), f"{got / 2 ** 20:.0f} MB")


# ------------------------------------------------------------------------------------------------ kinds
def kind_model():
    """The census's 14 kinds as nearest centres: scripts/census.py `kinds` re-run exactly (same pages, seed 0), so a new
    page is given the kind whose centre it is nearest, in the census's own scaling. Cached in out/census/kind_centres.npz."""
    cache = ROOT / "out/census/kind_centres.npz"
    if cache.exists():
        z = np.load(cache)
        return z["lo"], z["hi"], z["mean"], z["std"], z["C"]
    sys.path.insert(0, str(ROOT / "scripts"))
    from census import kmeans
    d = np.load(ROOT / "out/census/pages.npz"); X, stem, page = d["X"], d["stem"], d["page"]
    Z = X.copy(); Z[:, 0] = np.log1p(Z[:, 0]); Z[:, 1] = np.log1p(Z[:, 1])
    lo, hi = np.percentile(Z, 1, axis=0), np.percentile(Z, 99, axis=0)
    mean, std = Z.mean(0), Z.std(0) + 1e-6
    Z = (np.clip(Z, lo, hi) - mean) / std
    rng = np.random.default_rng(0)
    lab, C = min((kmeans(Z, 14, rng) for _ in range(4)), key=lambda r: ((Z - r[1][r[0]]) ** 2).sum())
    kinds = json.loads((ROOT / "out/census/kinds.json").read_text())["kinds"]
    where = {(s, int(p)): i for i, (s, p) in enumerate(zip(stem.tolist(), page.tolist()))}
    ok = total = 0
    for k in kinds:  # the examples the census drew for each kind must come out in that kind again
        for e in k["examples"]:
            total += 1; ok += lab[where[e["stem"], e["page"]]] == k["kind"]
    sizes_ok = all(int((lab == k["kind"]).sum()) == k["pages"] for k in kinds)
    print(f"census kinds re-made: {ok}/{total} example pages in their kind, page counts {'identical' if sizes_ok else 'DIFFER'}", flush=True)
    if ok != total or not sizes_ok:
        raise SystemExit("the census's kinds could not be re-made exactly: stop")
    np.savez(cache, lo=lo, hi=hi, mean=mean, std=std, C=C)
    return lo, hi, mean, std, C


def kind_names() -> dict[int, str]:
    c = json.loads((ROOT / "out/census/census_data.js").read_text(encoding="utf-8").split("=", 1)[1].rstrip().rstrip(";"))
    return {k["kind"]: k["name"] for k in c["kinds"]}


def page_rows() -> list[dict]:
    """Census features for every page (first 40) of every fetched document, and its nearest kind."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from census import doc_rows
    lo, hi, mean, std, C = kind_model()
    names = kind_names()
    rows = []
    for d in sorted(p for p in DOCS.iterdir() if (p / f"{p.name}.json").exists()):
        az = json.loads((d / f"{d.name}.json").read_text(encoding="utf-8")); az = az.get("analyzeResult", az)
        nl = {pg["pageNumber"]: len(pg.get("lines", [])) for pg in az["pages"]}
        for stem, pno, _, r in doc_rows((str(DOCS), d.name, 40)):
            z = np.array(r, np.float32); z[0] = np.log1p(z[0]); z[1] = np.log1p(z[1])
            z = (np.clip(z, lo, hi) - mean) / std
            k = int(((C - z) ** 2).sum(1).argmin())
            rows.append({"doc": stem, "number": int(pno), "lines": nl[pno], "kind_id": k, "kind": names[k], "pages_in_doc": len(az["pages"])})
    return rows


def select(n_pages: int, per_kind: int) -> None:
    rows = [r for r in page_rows() if 1 <= r["lines"] <= 400]
    rng = np.random.default_rng(SEED)
    docs = sorted({r["doc"] for r in rows}); rng.shuffle(docs)
    rank = {d: i for i, d in enumerate(docs)}
    by_doc = defaultdict(list)
    for r in rows:
        by_doc[r["doc"]].append(r)
    chosen, count = {}, Counter()  # (doc, number) -> row
    per_doc = Counter()

    def take(r):
        chosen[r["doc"], r["number"]] = r; count[r["kind"]] += 1; per_doc[r["doc"]] += 1

    # 1. the scarce kinds first: up to per_kind pages each, one or two pages a document, documents in a fixed shuffled order
    have_kind = {k: sorted({r["doc"] for r in rows if r["kind"] == k}, key=rank.get) for k in {r["kind"] for r in rows}}
    for k in sorted(have_kind, key=lambda k: len(have_kind[k])):
        for pas in (0, 1):  # first documents with nothing chosen yet, then documents with one page chosen
            for d in have_kind[k]:
                if count[k] >= per_kind:
                    break
                if per_doc[d] != pas:
                    continue
                opts = [r for r in by_doc[d] if r["kind"] == k and (d, r["number"]) not in chosen]
                if opts:
                    take(opts[int(rng.integers(len(opts)))])
    # 2. then one page from each document that has none, then a second page, until n_pages
    for pas in (0, 1):
        for d in docs:
            if len(chosen) >= n_pages:
                break
            if per_doc[d] != pas:
                continue
            opts = [r for r in by_doc[d] if (d, r["number"]) not in chosen]
            if opts:
                take(opts[int(rng.integers(len(opts)))])
    cf = LAB / f"{PFX}_candidates.json"
    cand = {c["id"]: c for c in json.loads(cf.read_text())["candidates"]} if cf.exists() else {}
    pages = sorted(chosen.values(), key=lambda r: (r["doc"], r["number"]))
    for r in pages:
        r["page"] = f"{r['doc']}_p{r['number']}"; r["journal_unseen"] = cand.get(r["doc"], {}).get("journal_unseen")
    out = {"seed": SEED, "made": time.strftime("%Y-%m-%d %H:%M"), "note": "TEST SET: never train or tune on these pages",
           "documents": len({r["doc"] for r in pages}), "pages": pages,
           "kinds": dict(Counter(r["kind"] for r in pages).most_common()),
           "kinds_available": dict(Counter(r["kind"] for r in rows).most_common())}
    (LAB / f"{PFX}_pages.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(pages)} pages from {out['documents']} documents ({sum(bool(r['journal_unseen']) for r in pages)} pages from never-seen journals); "
          f"{len(rows)} eligible pages in {len(docs)} documents")
    for k, v in out["kinds"].items():
        print(f"  {k:40s} {v:4d}   (eligible {out['kinds_available'][k]})")


# ------------------------------------------------------------------------------------------------ labeller inputs
def inputs() -> None:
    """As scripts/labeller_dev_inputs.py made the development pages: drawn by the crop box at 150 dpi, scaled to 1150
    wide with a 70-pixel margin, Azure's lines boxed in blue and numbered in red, lines.json in thousandths."""
    import cv2
    out = LAB / f"{PFX}_inputs"
    pages = json.loads((LAB / f"{PFX}_pages.json").read_text())["pages"]
    total = 0
    for p in pages:
        d = out / p["page"]
        if (d / "lines.json").exists():
            continue
        az = json.loads((DOCS / p["doc"] / f"{p['doc']}.json").read_text(encoding="utf-8")); az = az.get("analyzeResult", az)
        pg = next(q for q in az["pages"] if q["pageNumber"] == p["number"])
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["pdftoppm", "-cropbox", "-r", "150", "-gray", "-png", "-f", str(p["number"]), "-l", str(p["number"]),
                            str(DOCS / p["doc"] / f"{p['doc']}.pdf"), f"{tmp}/p"], check=True, stderr=subprocess.DEVNULL, timeout=300)
            img = cv2.imread(str(next(Path(tmp).glob("p-*.png"))))
        s = 1150 / img.shape[1]; img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        img = cv2.copyMakeBorder(img, 0, 0, 0, 70, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        sx, sy = 1150 / pg["width"], img.shape[0] / pg["height"]
        lines = []
        for i, l in enumerate(pg["lines"]):
            q = l["polygon"]; x0, y0, x1, y1 = min(q[0::2]), min(q[1::2]), max(q[0::2]), max(q[1::2])
            cv2.rectangle(img, (int(x0 * sx), int(y0 * sy)), (int(x1 * sx), int(y1 * sy)), (200, 120, 0), 1)
            cv2.putText(img, str(i), (int(x1 * sx) + 3, int((y0 + y1) / 2 * sy) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 220), 1, cv2.LINE_AA)
            lines.append({"id": f"{p['number']}:{i}", "box": [round(1000 * x0 / pg["width"]), round(1000 * y0 / pg["height"]),
                          round(1000 * x1 / pg["width"]), round(1000 * y1 / pg["height"])], "text": l["content"]})
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"page_{p['number']}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        (d / "lines.json").write_text(json.dumps(lines, ensure_ascii=False, indent=0), encoding="utf-8")
        total += len(lines)
    print(f"{len(pages)} pages, {total} lines written to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["candidates", "fetch", "select", "inputs", "kinds"])
    ap.add_argument("--n", type=int, default=330); ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--what", choices=["json", "pdf"], default="json", help="fetch: Azure's JSON of the candidates, or the PDFs of the chosen pages")
    ap.add_argument("--pages", type=int, default=200); ap.add_argument("--per-kind", type=int, default=10)
    ap.add_argument("--set", default="check", choices=["check", "unseen"], help="check: the exam (default); unseen: the development set from "
                    "unseen journals (110 documents in /home/yassine/data/unseen_docs, fetched elsewhere; select and inputs only)")
    a = ap.parse_args()
    if a.set == "unseen":
        PFX, DOCS = "unseen", Path("/home/yassine/data/unseen_docs")
    if a.cmd == "candidates": candidates(a.n)
    elif a.cmd == "fetch": fetch(a.what, a.workers)
    elif a.cmd == "select": select(a.pages, a.per_kind)
    elif a.cmd == "inputs": inputs()
    elif a.cmd == "kinds": kind_model()
