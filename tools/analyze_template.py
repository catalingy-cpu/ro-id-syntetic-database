#!/usr/bin/env python3
"""
Phase 1 — analiză template + vizualizare câmpuri JSON.

Exemple:
  python tools/analyze_template.py --preview
  python tools/analyze_template.py --set surname --x 392 --y 250 --width 430 --height 38
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ro_id_synth.template_config import load_template_spec, save_fields_overlay  # noqa: E402
DEFAULT_FIELDS = ROOT / "config" / "template_fields.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analiză template CI + câmpuri editabile")
    parser.add_argument("--fields", type=str, default=str(DEFAULT_FIELDS))
    parser.add_argument("--preview", action="store_true", help="Salvează overlay ROI în debug/")
    parser.add_argument("--set", type=str, help="Actualizează un câmp (ex: surname)")
    parser.add_argument("--x", type=int)
    parser.add_argument("--y", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--ink", type=str, choices=["dark", "red", "cnp_multicolor", "auto"])
    args = parser.parse_args()

    fields_path = Path(args.fields).resolve()
    raw = json.loads(fields_path.read_text(encoding="utf-8"))

    if args.set:
        if args.set not in raw["fields"]:
            raw["fields"][args.set] = {"x": 0, "y": 0, "width": 100, "height": 30}
        field = raw["fields"][args.set]
        if args.x is not None:
            field["x"] = args.x
        if args.y is not None:
            field["y"] = args.y
        if args.width is not None:
            field["width"] = args.width
        if args.height is not None:
            field["height"] = args.height
        if args.ink:
            field["ink"] = args.ink
        fields_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Actualizat câmp '{args.set}' în {fields_path}")

    spec = load_template_spec(fields_path, ROOT)
    img = spec.load_image()
    h, w = img.shape[:2]
    print(f"Template: {spec.image_path.name} ({w}×{h})")
    print(f"Câmpuri: {len(spec.fields)}")
    for key, f in spec.fields.items():
        print(f"  {key:16} x={f.x:4} y={f.y:4} w={f.width:4} h={f.height:3} ink={f.ink}")

    if args.preview or args.set:
        out = ROOT / "debug" / "fields_overlay.jpg"
        save_fields_overlay(spec, out)
        print(f"Preview: {out}")


if __name__ == "__main__":
    main()
