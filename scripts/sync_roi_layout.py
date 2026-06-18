#!/usr/bin/env python3
"""Generează ROI-uri normalizate din config/template_fields.json (sursă unică de adevăr)."""

from __future__ import annotations

import json
from pathlib import Path


def _merge_boxes(
    boxes: list[tuple[int, int, int, int]],
) -> tuple[float, float, float, float]:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return x0, y0, x1 - x0, y1 - y0


def _norm(x: float, y: float, w: float, h: float, tw: int, th: int) -> tuple[float, float, float, float]:
    return (
        round(x / tw, 4),
        round(y / th, 4),
        round(w / tw, 4),
        round(h / th, 4),
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    tpl_path = root / "config" / "template_fields.json"
    data = json.loads(tpl_path.read_text(encoding="utf-8"))
    tw = int(data["width"])
    th = int(data["height"])
    f = data["fields"]

    def box(name: str) -> tuple[int, int, int, int]:
        spec = f[name]
        return int(spec["x"]), int(spec["y"]), int(spec["width"]), int(spec["height"])

    serie_numar = _merge_boxes([box("series"), box("number")])
    adresa = _merge_boxes([box("address_line1"), box("address_line2")])

    mapping: list[tuple[str, tuple[int, int, int, int], int]] = [
        ("serie_numar", serie_numar, 10),
        ("cnp", box("cnp"), 20),
        ("nume", box("surname"), 30),
        ("prenume", box("given_name"), 40),
        ("cetatenie", box("nationality"), 50),
        ("loc_nastere", box("birth_place"), 60),
        ("adresa", adresa, 70),
        ("emis_de", box("issued_by"), 80),
        ("valabilitate", box("validity"), 85),
        ("sex", box("sex"), 88),
    ]

    print("# ROI-uri clasice — generate din template_fields.json")
    print("ID_CARD_ROIS_CLASSIC: tuple[FieldRoi, ...] = (")
    for name, (x, y, w, h), prio in mapping:
        nx, ny, nw, nh = _norm(x, y, w, h, tw, th)
        print(f'    FieldRoi("{name}", {nx}, {ny}, {nw}, {nh}, {prio}),')
    print(")")
    cnp = box("cnp")
    nx, ny, nw, nh = _norm(cnp[0], cnp[1], cnp[2], cnp[3], tw, th)
    print(f'\nCNP_PROBE_CLASSIC = FieldRoi("_probe", {nx}, {ny}, {nw}, {nh})')


if __name__ == "__main__":
    main()
