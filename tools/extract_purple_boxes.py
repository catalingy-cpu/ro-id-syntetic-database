#!/usr/bin/env python3
"""Extrage cutii mov/magenta de pe imaginea de corecție a utilizatorului."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def purple_mask(img: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(img)
    heuristic = (r.astype(np.int16) + b.astype(np.int16) - 2 * g.astype(np.int16)) > 80
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, np.array((140, 30, 100)), np.array((180, 255, 255)))
    mask = np.maximum(heuristic.astype(np.uint8) * 255, hsv_mask)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), 1)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), 2)


def extract_boxes(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    orange = ((img[:, :, 2] > 190) & (img[:, :, 1] > 110) & (img[:, :, 0] < 100))
    green = ((img[:, :, 1] > 170) & (img[:, :, 2] < 120) & (img[:, :, 0] < 120))
    purple = purple_mask(img)
    purple[orange | green] = 0

    n, _, stats, _ = cv2.connectedComponentsWithStats(purple, connectivity=8)
    boxes: list[tuple[int, int, int, int, int]] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 1200 or w < 25 or h < 12:
            continue
        boxes.append((x, y, w, h, area))

    merged: list[tuple[int, int, int, int]] = []
    for x, y, w, h, _ in sorted(boxes, key=lambda b: (-b[4], b[1], b[0])):
        rect = (x, y, w, h)
        if any(
            abs(x - mx) < 8 and abs(y - my) < 8 and abs(w - mw) < 15 and abs(h - mh) < 15
            for mx, my, mw, mh in merged
        ):
            continue
        merged.append(rect)

    # include large horizontal spans from dilated mask rows
    mask = purple.copy()
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), 2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 8000 or w < 80 or h < 25:
            continue
        rect = (x, y, w, h)
        if not any(abs(x - mx) < 10 and abs(y - my) < 10 for mx, my, _, _ in merged):
            merged.append(rect)

    return sorted(merged, key=lambda b: (b[1], b[0]))


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        ROOT.parent.parent
        / "assets"
        / "c__Users_catal_AppData_Roaming_Cursor_User_workspaceStorage_bbec55b68a24f96c93f98c5c3c80639f_images_image-c1d2de59-58f2-4408-a976-c75dd840a82c.png"
    )
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    boxes = extract_boxes(img)
    print(f"{path.name}: {len(boxes)} boxes")
    for i, (x, y, w, h) in enumerate(boxes):
        print(f"  {i:2} x={x:4} y={y:4} w={w:4} h={h:3}")


if __name__ == "__main__":
    main()
