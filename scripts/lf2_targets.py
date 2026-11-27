"""Line-finder targets from saved Azure boxes (see lf2_train.py)."""
import cv2
import numpy as np

from ocrl.blobs.linefinder import core_mask


def segments(z, split_gap):
    """Line boxes -> boxes to mark (the same, or each line cut at wide gaps between its words)."""
    boxes = z["boxes"]
    if split_gap <= 0 or len(z["words"]) == 0:
        return boxes
    out = []
    wl, wb = z["word_line"], z["words"]
    for i, b in enumerate(boxes):
        ws = wb[wl == i]
        h = b[3] - b[1]
        if len(ws) < 2 or h <= 0 or h > 1.5 * (b[2] - b[0]):  # tall narrow boxes (sideways text) are left whole
            out.append(b); continue
        ws = ws[np.argsort(ws[:, 0])]
        start, right = ws[0, 0], ws[0, 2]
        for w in ws[1:]:
            if w[0] - right > split_gap * h:
                out.append([max(start, b[0]), b[1], min(right, b[2]), b[3]]); start = w[0]
            right = max(right, w[2])
        out.append([max(start, b[0]), b[1], min(right, b[2]), b[3]])
    return np.array(out, np.float32).reshape(-1, 4)


def uprights(img):
    """Upright ruled lines: dark runs far taller than any letter."""
    ink = (img < 140).astype(np.uint8)
    return cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45)))


def target(img, z, split_gap=0.0, rule_cut=False):
    m = core_mask(img.shape, segments(z, split_gap))
    if rule_cut:
        r = uprights(img)
        if r.any():
            m[cv2.dilate(r, np.ones((1, 3), np.uint8)) > 0] = 0
    return m


def table_score(boxes):
    """Share of a page's Azure lines that sit in a row with two or more others (cells of a table row)."""
    if len(boxes) < 6:
        return 0.0
    b = np.asarray(boxes, np.float32)
    h = b[:, 3] - b[:, 1]
    lo, hi = np.maximum(b[:, None, 1], b[None, :, 1]), np.minimum(b[:, None, 3], b[None, :, 3])
    ov = (hi - lo) > 0.5 * np.minimum(h[:, None], h[None, :])
    apart = (b[:, None, 2] < b[None, :, 0]) | (b[None, :, 2] < b[:, None, 0])
    k = (ov & apart).sum(1)
    return float((k >= 2).mean())


def clean_small(small, width=2480):
    """The pipeline's clean_page() run on a page already scaled for the finder: scaled up to about 300 dpi, cleaned,
    scaled back. A stand-in for drawing at 300 dpi and cleaning (what the finder is given at work)."""
    from ocrl.blobs.clean import clean_page
    H, W = small.shape
    big = cv2.resize(small, (width, int(round(H * width / W))), interpolation=cv2.INTER_CUBIC)
    g = clean_page(big).gray
    return cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
