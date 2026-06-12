#!/usr/bin/env python3
"""Extrage ROI-uri din cutii portocalii pe imaginea de referință a utilizatorului."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _orange_mask(img: np.ndarray) -> np.ndarray:
    return (
        (img[:, :, 2] > 190)
        & (img[:, :, 1] > 110)
        & (img[:, :, 1] < 240)
        & (img[:, :, 0] < 100)
    )


def _pair_horizontal_edges(orange: np.ndarray) -> list[tuple[int, int, int, int]]:
    hlines: list[tuple[int, int, int]] = []
    for y in range(orange.shape[0]):
        xs = np.where(orange[y])[0]
        if len(xs) < 20:
            continue
        splits = np.where(np.diff(xs) > 3)[0]
        start = 0
        for sp in list(splits) + [len(xs) - 1]:
            seg = xs[start : sp + 1]
            if len(seg) >= 20:
                hlines.append((y, int(seg[0]), int(seg[-1])))
            start = sp + 1

    boxes: set[tuple[int, int, int, int]] = set()
    for i, (y1, x1a, x1b) in enumerate(hlines):
        for y2, x2a, x2b in hlines[i + 1 :]:
            dy = y2 - y1
            if dy < 12 or dy > 70:
                continue
            xa = max(x1a, x2a)
            xb = min(x1b, x2b)
            if xb - xa < 30:
                continue
            boxes.add((xa, y1, xb - xa + 1, dy))
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def main() -> None:
    ref = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else ROOT.parent.parent
        / "assets"
        / "c__Users_catal_AppData_Roaming_Cursor_User_workspaceStorage_bbec55b68a24f96c93f98c5c3c80639f_images_image-1e1c05f9-536c-461e-abba-ac9ef0444dce.png"
    )
    img = cv2.imdecode(np.fromfile(str(ref), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Nu pot citi {ref}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boxes = _pair_horizontal_edges(_orange_mask(img))
    print(f"Image {img.shape[1]}x{img.shape[0]} — {len(boxes)} boxes")
    for x, y, w, h in boxes:
        roi = gray[y : y + h, x : x + w]
        ink = float((roi < 145).sum()) / roi.size * 100
        print(f"  {x:4},{y:4} {w:4}x{h:3} ink={ink:5.1f}%")


if __name__ == "__main__":
    main()
