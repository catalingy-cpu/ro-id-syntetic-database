#!/usr/bin/env python3
"""
Convertește labels/train.txt (format FRCHub) în fișiere pentru antrenare rec PaddleOCR.

Intrare: dataset/labels/train.txt
         fiecare rând: images/0001.jpg<TAB>{"transcription": "NUME: ...\\n..."}

Ieșire:
  dataset/paddle_rec/train_list.txt
  dataset/paddle_rec/val_list.txt
  dataset/paddle_rec/README.txt  (hint cale absolută pentru Paddle)

Utilizare:
  python scripts/convert_to_paddle_rec.py --dataset dataset
  python scripts/convert_to_paddle_rec.py --dataset dataset --val-ratio 0.05
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="dataset", help="Folder cu images/ și labels/train.txt")
    p.add_argument("--val-ratio", type=float, default=0.05, help="Fracție pentru validare (0–0.3)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_rows(train_file: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in train_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "\t" not in line:
            continue
        rel_path, payload = line.split("\t", 1)
        try:
            data = json.loads(payload)
            text = str(data.get("transcription", "")).strip()
        except json.JSONDecodeError:
            text = payload.strip()
        if not text:
            continue
        # Paddle rec: o singură linie etichetă (fără TAB-uri în text)
        text = text.replace("\t", " ").replace("\r\n", "\n").replace("\r", "\n")
        rows.append((rel_path.replace("\\", "/"), text))
    return rows


def write_list(path: Path, items: list[tuple[str, str]], image_root: Path) -> None:
    lines: list[str] = []
    for rel, label in items:
        abs_img = (image_root / rel).resolve()
        # Paddle acceptă de obicei calea relativă la fișierul listă sau absolută
        rel_to_root = rel.replace("\\", "/")
        safe_label = label.replace("\t", " ")
        lines.append(f"{rel_to_root}\t{safe_label}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.dataset).resolve()
    train_txt = root / "labels" / "train.txt"
    if not train_txt.is_file():
        raise SystemExit(f"Lipsește {train_txt} — rulează mai întâi generate.py")

    rows = load_rows(train_txt)
    if not rows:
        raise SystemExit("Nicio etichetă validă în train.txt")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_ratio)) if args.val_ratio > 0 else 0
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    out_dir = root / "paddle_rec"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_root = root

    write_list(out_dir / "train_list.txt", train_rows, image_root)
    write_list(out_dir / "val_list.txt", val_rows, image_root)

    hint = (
        f"Rădăcină dataset (absolut): {root}\n"
        f"Antrenare: {len(train_rows)} imagini\n"
        f"Validare: {len(val_rows)} imagini\n"
        f"Liste: train_list.txt, val_list.txt (căi relative la {root})\n"
    )
    (out_dir / "README.txt").write_text(hint, encoding="utf-8")

    print(hint)


if __name__ == "__main__":
    main()
