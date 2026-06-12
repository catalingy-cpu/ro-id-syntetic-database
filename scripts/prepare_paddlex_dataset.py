#!/usr/bin/env python3
"""Pregătește dataset/ pentru PaddleX MSTextRecDataset (train.txt, val.txt, dict.txt)."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="dataset")
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subdir", type=str, default="paddlex", help="Subfolder sub dataset/")
    p.add_argument(
        "--line-crops",
        action="store_true",
        default=True,
        help="Genereaza line-crops OCR din fiecare card (default: activ).",
    )
    p.add_argument(
        "--target-height",
        type=int,
        default=64,
        help="Inaltime tinta pentru fiecare line-crop.",
    )
    p.add_argument(
        "--max-width",
        type=int,
        default=1024,
        help="Latime maxima dupa resize pentru line-crops.",
    )
    return p.parse_args()


def load_rows(train_file: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in train_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "\t" not in line:
            continue
        rel, payload = line.split("\t", 1)
        try:
            text = str(json.loads(payload).get("transcription", "")).strip()
        except json.JSONDecodeError:
            text = payload.strip()
        if not text:
            continue
        text = text.replace("\t", " ").replace("\r\n", "\n").replace("\r", "\n")
        rows.append((rel.replace("\\", "/"), text))
    return rows


def _normalize_label(label: str) -> str:
    return " ".join(label.replace("\t", " ").split()).strip()


def _line_texts_from_transcription(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.strip()
        if not line:
            continue
        if ":" in line:
            _, value = line.split(":", 1)
            value = _normalize_label(value)
            if value:
                lines.append(value)
        else:
            value = _normalize_label(line)
            if value:
                lines.append(value)
    return lines


def _resize_crop(img: Image.Image, target_height: int, max_width: int) -> Image.Image:
    w, h = img.size
    if h <= 0:
        return img
    new_w = max(1, int(round(w * (target_height / float(h)))))
    new_h = target_height
    if new_w > max_width:
        scale = max_width / float(new_w)
        new_w = max(1, int(round(new_w * scale)))
        new_h = max(1, int(round(new_h * scale)))
    return img.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)


def make_line_crops(
    root: Path, rows: list[tuple[str, str]], px_dir: Path, target_height: int, max_width: int
) -> list[tuple[str, str]]:
    line_dir = px_dir / "line_images"
    if line_dir.exists():
        shutil.rmtree(line_dir)
    line_dir.mkdir(parents=True, exist_ok=True)

    samples: list[tuple[str, str]] = []
    for rel, transcript in rows:
        src = root / rel
        if not src.is_file():
            continue
        lines = _line_texts_from_transcription(transcript)
        if not lines:
            continue
        try:
            with Image.open(src) as im:
                im = im.convert("RGB")
                w, h = im.size
                if w <= 0 or h <= 0:
                    continue
                stem = Path(rel).stem
                n = len(lines)
                for idx, line in enumerate(lines):
                    y0 = max(0, int((idx / n) * h))
                    y1 = min(h, int(((idx + 1) / n) * h))
                    if y1 - y0 < 4:
                        continue
                    # Mic padding vertical pentru robustete pe layout-uri usor diferite.
                    pad = max(1, int(0.02 * (y1 - y0)))
                    y0p = max(0, y0 - pad)
                    y1p = min(h, y1 + pad)
                    crop = im.crop((0, y0p, w, y1p))
                    crop = _resize_crop(crop, target_height, max_width)
                    name = f"{stem}_l{idx:02d}.jpg"
                    out_file = line_dir / name
                    crop.save(out_file, format="JPEG", quality=90, optimize=True)
                    samples.append((f"line_images/{name}", _normalize_label(line)))
        except OSError:
            continue
    return samples


def build_dict(labels: list[str]) -> str:
    chars: set[str] = set()
    for label in labels:
        for ch in label:
            if ch == "\n":
                chars.add(" ")
            else:
                chars.add(ch)
    ordered = sorted(chars)
    return "\n".join(ordered) + "\n"


def link_images(px_dir: Path, images_dir: Path) -> None:
    link = px_dir / "images"
    if link.exists():
        return
    if not images_dir.is_dir():
        raise SystemExit(f"Lipsește {images_dir}")
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(images_dir.resolve())],
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            link.symlink_to(images_dir.resolve(), target_is_directory=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(
            f"Nu pot lega {link} -> {images_dir} ({exc}).\n"
            "Windows: rulează terminal Administrator, sau copiază manual images/ în dataset/paddlex/images"
        ) from exc


def main() -> None:
    args = parse_args()
    root = Path(args.dataset).resolve()
    train_txt = root / "labels" / "train.txt"
    if not train_txt.is_file():
        raise SystemExit(f"Lipsește {train_txt}")

    rows = load_rows(train_txt)
    px_dir = root / args.subdir
    px_dir.mkdir(parents=True, exist_ok=True)

    if args.line_crops:
        all_samples = make_line_crops(root, rows, px_dir, args.target_height, args.max_width)
        if not all_samples:
            raise SystemExit("Nu am putut genera line-crops. Verifica dataset/images si labels/train.txt")
        rows = all_samples
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_ratio)) if args.val_ratio > 0 else 0
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]
    if not args.line_crops:
        link_images(px_dir, root / "images")

    def write_list(name: str, items: list[tuple[str, str]]) -> None:
        lines = [f"{rel}\t{lab.replace(chr(10), ' ')}" for rel, lab in items]
        (px_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_list("train.txt", train_rows)
    write_list("val.txt", val_rows)
    (px_dir / "dict.txt").write_text(build_dict([lab for _, lab in rows]), encoding="utf-8")

    print(f"PaddleX dataset: {px_dir}")
    print(f"  train={len(train_rows)} val={len(val_rows)}")
    if args.line_crops:
        print(f"  line_images={len(rows)}")
    print(f"  dict.txt caractere: {len((px_dir / 'dict.txt').read_text(encoding='utf-8').splitlines())}")


if __name__ == "__main__":
    main()
