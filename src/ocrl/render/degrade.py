"""Make a clean rendered page look scanned. The ownership map follows any geometric change."""

from __future__ import annotations

import cv2
import numpy as np

from .page import PageTruth


def degrade(page: PageTruth, rng: np.random.Generator, strength: float = 1.0) -> PageTruth:
    cov = (255 - page.image).astype(np.float32) / 255.0
    owner = page.owner
    s = strength
    ops = []

    if rng.random() < 0.5 * s:  # slight rotation, as from a page laid crooked on the glass
        angle = rng.uniform(-0.35, 0.35)
        ops.append("rotate")
        h, w = cov.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        cov = cv2.warpAffine(cov, M, (w, h), flags=cv2.INTER_LINEAR)
        owner = cv2.warpAffine(owner.astype(np.float32), M, (w, h), flags=cv2.INTER_NEAREST).astype(np.int32)
    # A black-and-white scan forces every pixel to ink or paper. Decided first, because
    # whoever makes such a scan sets the threshold so the text survives: starved ink and a
    # harsh threshold together would erase the page, which is not a page anyone keeps.
    bilevel = rng.random() < 0.35 * s
    starved = False
    if rng.random() < 0.38 * s:  # ink spread or starved ink
        spread = rng.random() < 0.7 or page.px < 40
        big = rng.random() < 0.4 and (spread or page.px >= 50)  # a 3x3 erosion wipes out small type
        k = np.ones((3, 3) if big else (2, 2), np.uint8)
        ops.append(f"{'spread' if spread else 'thin'}{k.shape[0]}")
        cov = cv2.dilate(cov, k) if spread else cv2.erode(cov, k)
        starved = not spread
    if rng.random() < 0.7 * s:  # optical blur
        sigma = rng.uniform(0.3, 1.1)
        ops.append("blur-strong" if sigma > 0.7 else "blur-mild")
        cov = cv2.GaussianBlur(cov, (0, 0), sigma)
    if rng.random() < 0.7 * s:  # exposure: thicker or thinner strokes once thresholded
        gamma = rng.uniform(0.6, 1.2 if starved else 1.7)
        ops.append("exposure-heavy" if gamma < 0.9 else "exposure-light" if gamma > 1.2 else "exposure-mid")
        cov = np.power(np.clip(cov, 0, 1), gamma)
        starved = starved or gamma > 1.2

    if bilevel:
        ops.append("bilevel")
        cov = (cov > (rng.uniform(0.2, 0.35) if starved else rng.uniform(0.3, 0.55))).astype(np.float32)
        paper, ink = 255.0, 0.0
    else:
        paper, ink = rng.uniform(215, 255), rng.uniform(0, 70)
    img = paper - cov * (paper - ink)
    if not bilevel and rng.random() < 0.7 * s:
        ops.append("noise")
        img = img + rng.normal(0, rng.uniform(1, 10), img.shape).astype(np.float32)
    if rng.random() < 0.4 * s:  # specks: single pixels, and the larger grains a scanner bed leaves
        ops.append("specks")
        n = int(rng.integers(5, 60))
        ys, xs = rng.integers(0, img.shape[0], n), rng.integers(0, img.shape[1], n)
        img[ys, xs] = ink
        if rng.random() < 0.6:
            for _ in range(int(rng.integers(5, 80))):
                c = (int(rng.integers(img.shape[1])), int(rng.integers(img.shape[0])))
                cv2.circle(img, c, int(rng.integers(1, 3)), float(ink), -1)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if not bilevel and rng.random() < 0.3 * s:
        ops.append("jpeg")
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 90))])
        img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return PageTruth(img, owner, page.units, page.lines, page.baselines, page.font, page.px, {**page.meta, "ops": ops})
