"""For given (doc, page, azure box): which of our lines, its blobs (role, box) and what sticks out of Azure's box."""
import sys, json, os
sys.path.insert(0, "/home/yassine/OCRL/scripts")
import numpy as np, cv2, torch
torch.set_num_threads(4)
import lost_causes as LC
from ocrl.pipeline import OCR
targets = json.loads(sys.argv[1])  # [[doc, page, [x0,y0,x1,y1]], ...]
out = sys.argv[2]
pages = {(p["doc"], p["page"]): p for p in json.load(open(LC.ROOT / "out/lost/pages.json"))}
ocr = OCR(str(LC.ROOT / "out/recognizer_v3.pt"), linefinder=str(LC.ROOT / "out/linefinder.pt"), line_reader=LC.READER)
os.makedirs(out, exist_ok=True)
for doc, page, z in targets:
    p = pages[(doc, page)]
    gray = LC.page_image(__import__("pathlib").Path(p["dir"]), doc, page)
    d = ocr.read_page(gray)
    g = d.cleaned.gray
    zc = ((z[0] + z[2]) / 2, (z[1] + z[3]) / 2)
    for r, res in d.regions:
        ox, oy = r.x0 - 16, r.y0 - 16
        lay = res.layout
        for li, ln in enumerate(res.lines):
            if not ln.clusters: continue
            ids = [b for c in ln.clusters for b in c.blobs]
            bx = lay.boxes[ids]
            box = (bx[:, 0].min() + ox, bx[:, 1].min() + oy, bx[:, 2].max() + ox, bx[:, 3].max() + oy)
            if box[0] <= zc[0] <= box[2] and box[1] <= zc[1] <= box[3]:
                print(doc, page, "az", z, "ours", box, repr(ln.text[:50]), ln.warnings)
                for b in ids:
                    B = lay.boxes[b] + [ox, oy, ox, oy]
                    if B[1] < z[1] - 3 or B[3] > z[3] + 3:
                        print("   sticks out: blob", b, "box", B.tolist(), "area", int(lay.area[b]), "role", int(lay.role[b]), "host", int(lay.host[b]), "line", int(lay.line[b]))
                img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
                for b in ids:
                    B = (lay.boxes[b] + [ox, oy, ox, oy]).astype(int)
                    cv2.rectangle(img, (B[0], B[1]), (B[2], B[3]), (0, 0, 255) if lay.role[b] == 1 else (255, 0, 0), 1)
                cv2.rectangle(img, (int(z[0]), int(z[1])), (int(z[2]), int(z[3])), (0, 160, 0), 1)
                y0, y1 = int(max(min(z[1], box[1]) - 60, 0)), int(max(z[3], box[3]) + 60)
                x0, x1 = int(max(min(z[0], box[0]) - 30, 0)), int(max(z[2], box[2]) + 30)
                cv2.imwrite(f"{out}/{doc}_p{page}_{int(z[1])}.png", img[y0:y1, x0:x1])
