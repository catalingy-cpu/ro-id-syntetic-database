#!/usr/bin/env python3
"""
Importă ROI-uri din imaginea de referință cu cutii portocalii (aceeași rezoluție ca template-ul).
Scrie coordonate direct în config/template_fields.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "config" / "template_fields.json"


def _orange_mask(img: np.ndarray) -> np.ndarray:
    return (
        (img[:, :, 2] > 190)
        & (img[:, :, 1] > 110)
        & (img[:, :, 1] < 240)
        & (img[:, :, 0] < 100)
    )


def _text_bounds(
    gray: np.ndarray,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    thr: int = 145,
) -> tuple[int, int, int, int] | None:
    roi = gray[y0:y1, x0:x1]
    mask = roi < thr
    if mask.sum() < 20:
        return None
    ys, xs = np.where(mask)
    return (
        x0 + int(xs.min()),
        y0 + int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    )


def _pad_box(x: int, y: int, w: int, h: int, *, px: int = 2) -> tuple[int, int, int, int]:
    return max(0, x - px), max(0, y - px), w + 2 * px, h + 2 * px


def extract_fields(ref_path: Path) -> dict[str, dict[str, int | str]]:
    img = cv2.imdecode(np.fromfile(str(ref_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(ref_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = gray.copy()
    g[_orange_mask(img)] = 255

    specs: list[tuple[str, int, int, int, int, str, str]] = [
        ("series", 90, 140, 580, 660, "dark", "left"),
        ("number", 90, 140, 660, 830, "dark", "left"),
        ("cnp", 130, 180, 360, 850, "cnp_multicolor", "left"),
        ("surname", 185, 235, 360, 820, "dark", "left"),
        ("given_name", 230, 285, 360, 820, "dark", "left"),
        ("nationality", 295, 345, 360, 660, "dark", "left"),
        ("birth_place", 345, 400, 320, 770, "dark", "left"),
        ("address_line1", 400, 455, 320, 770, "dark", "left"),
        ("address_line2", 455, 495, 320, 770, "dark", "left"),
        ("sex", 285, 340, 850, 980, "dark", "center"),
        ("issued_by", 495, 540, 320, 690, "dark", "left"),
        ("validity", 490, 540, 690, 1000, "dark", "left"),
    ]

    fields: dict[str, dict[str, int | str]] = {}
    for key, y0, y1, x0, x1, ink, align in specs:
        box = _text_bounds(g, y0=y0, y1=y1, x0=x0, x1=x1)
        if box is None:
            raise RuntimeError(f"Nu am găsit text pentru câmpul '{key}'")
        x, y, w, h = _pad_box(*box)
        fields[key] = {
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "ink": ink,
            "align": align,
        }
    return fields


def main() -> None:
    parser = argparse.ArgumentParser(description="Import ROI din referința portocalie")
    parser.add_argument("reference", type=str, help="Imagine 1024×725 cu cutii portocalii")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--template", type=str, default="templates/classic_reference.png")
    args = parser.parse_args()

    ref = Path(args.reference).resolve()
    out = Path(args.out).resolve()
    fields = extract_fields(ref)

    payload = {
        "template_image": args.template,
        "width": 1024,
        "height": 725,
        "kind": "classic",
        "description": f"ROI importate din {ref.name}",
        "fields": fields,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Salvat {len(fields)} câmpuri în {out}")
    for key, spec in fields.items():
        print(f"  {key:16} x={spec['x']} y={spec['y']} w={spec['width']} h={spec['height']}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
