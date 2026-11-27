"""Labeller inputs for bulk labelling: every page of the documents kept on this machine, one folder per page.

Documents that hold any of the judged development pages are left out, so the test stays clean by document.
    uv run python scripts/labeller_bulk_inputs.py --workers 3
"""
import argparse, json, subprocess, tempfile
from multiprocessing import Pool
from pathlib import Path

S3 = Path("/home/yassine/data/mandumah_s3")
OUT = Path("out/labeller/bulk_inputs")


def one(args):
    stem, max_pages = args
    import cv2
    cv2.setNumThreads(1)
    try:
        az = json.loads((S3 / stem / f"{stem}.json").read_text(encoding="utf-8")); az = az.get("analyzeResult", az)
        n = min(len(az["pages"]), max_pages); made = 0
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["pdftoppm", "-cropbox", "-r", "110", "-gray", "-png", "-l", str(n), str(S3 / stem / f"{stem}.pdf"), f"{tmp}/p"], check=True, stderr=subprocess.DEVNULL, timeout=300)
            for pi, f in enumerate(sorted(Path(tmp).glob("p-*.png"))[:n]):
                pg = az["pages"][pi]; ls = pg.get("lines", [])
                if not (3 <= len(ls) <= 260) or abs(pg.get("angle") or 0) > 20:
                    continue
                img = cv2.imread(str(f))
                s = 1150 / img.shape[1]; img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                img = cv2.copyMakeBorder(img, 0, 0, 0, 70, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                sx, sy = 1150 / pg["width"], img.shape[0] / pg["height"]
                p = pg["pageNumber"]; d = OUT / f"{stem}_p{p}"; d.mkdir(parents=True, exist_ok=True); lines = []
                for i, l in enumerate(ls):
                    q = l["polygon"]; x0, y0, x1, y1 = min(q[0::2]), min(q[1::2]), max(q[0::2]), max(q[1::2])
                    cv2.rectangle(img, (int(x0 * sx), int(y0 * sy)), (int(x1 * sx), int(y1 * sy)), (200, 120, 0), 1)
                    cv2.putText(img, str(i), (int(x1 * sx) + 3, int((y0 + y1) / 2 * sy) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 220), 1, cv2.LINE_AA)
                    lines.append({"id": f"{p}:{i}", "box": [round(1000 * x0 / pg["width"]), round(1000 * y0 / pg["height"]), round(1000 * x1 / pg["width"]), round(1000 * y1 / pg["height"])], "text": l["content"]})
                cv2.imwrite(str(d / f"page_{p}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                (d / "lines.json").write_text(json.dumps(lines, ensure_ascii=False), encoding="utf-8"); made += 1
        return made
    except Exception:
        return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=3); ap.add_argument("--max-pages", type=int, default=14)
    a = ap.parse_args()
    judged = {p.name.rsplit("_p", 1)[0] for p in Path("out/labeller/dev_inputs").iterdir()}
    docs = sorted(p.name for p in S3.iterdir() if p.name not in judged and (p / f"{p.name}.pdf").exists())
    with Pool(a.workers, maxtasksperchild=10) as pool:
        made = sum(pool.imap_unordered(one, [(s, a.max_pages) for s in docs], chunksize=2))
    print(f"{made} pages from {len(docs)} documents ({len(judged & {p.name for p in S3.iterdir()})} documents left out because a judged page comes from them)")
