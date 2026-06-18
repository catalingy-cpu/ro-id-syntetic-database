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
        default=48,
        help="Inaltime tinta pentru fiecare line-crop (PP-OCRv5 rec: 48).",
    )
    p.add_argument(
        "--max-width",
        type=int,
        default=320,
        help="Latime fixa dupa resize+pad pentru line-crops (PP-OCRv5 rec: 320).",
    )
    p.add_argument(
        "--max-label-len",
        type=int,
        default=25,
        help="Lungime maxima eticheta OCR; randurile mai lungi sunt omise.",
    )
    p.add_argument(
        "--fields",
        type=str,
        default="",
        help="template_fields.json pentru crop pe ROI calibrat (recomandat).",
    )
    return p.parse_args()


def default_fields_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "template_fields.json"


# Cheie din transcript (NUME, CNP, ...) -> camp in template_fields.json
TRANSCRIPT_FIELD_TO_ROI: dict[str, str] = {
    "NUME": "surname",
    "PRENUME": "given_name",
    "CNP": "cnp",
    "SERIE": "series",
    "NUMAR": "number",
    "SEX": "sex",
    "CETATENIE": "nationality",
    "EMISA_DE": "issued_by",
    "DATA_EXPIRARE": "validity",
    "ADRESA": "address_line1",
    "ADRESA2": "address_line2",
    "LOC_NASTERE": "birth_place",
}


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


def _line_texts_from_transcription(text: str) -> list[tuple[str, str]]:
    """Returneaza (CHEIE, valoare) pentru fiecare rand din transcript."""
    lines: list[tuple[str, str]] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.strip()
        if not line:
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            value = _normalize_label(value)
            if value:
                lines.append((key.strip().upper(), value))
        else:
            value = _normalize_label(line)
            if value:
                lines.append(("", value))
    return lines


def _load_template_fields(path: Path) -> tuple[int, int, dict[str, dict[str, int]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tpl_w = int(data.get("width") or 0)
    tpl_h = int(data.get("height") or 0)
    fields = data.get("fields") or {}
    if tpl_w <= 0 or tpl_h <= 0 or not isinstance(fields, dict):
        raise SystemExit(f"template_fields invalid: {path}")
    out: dict[str, dict[str, int]] = {}
    for name, spec in fields.items():
        if not isinstance(spec, dict):
            continue
        out[str(name)] = {
            "x": int(spec["x"]),
            "y": int(spec["y"]),
            "width": int(spec["width"]),
            "height": int(spec["height"]),
        }
    return tpl_w, tpl_h, out


def _crop_from_roi(
    im: Image.Image,
    field: dict[str, int],
    tpl_w: int,
    tpl_h: int,
    *,
    pad_px: int = 2,
) -> Image.Image | None:
    w, h = im.size
    if w <= 0 or h <= 0:
        return None
    sx = w / float(tpl_w)
    sy = h / float(tpl_h)
    x0 = max(0, int(round(field["x"] * sx)) - pad_px)
    y0 = max(0, int(round(field["y"] * sy)) - pad_px)
    x1 = min(w, int(round((field["x"] + field["width"]) * sx)) + pad_px)
    y1 = min(h, int(round((field["y"] + field["height"]) * sy)) + pad_px)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return im.crop((x0, y0, x1, y1))


def _resize_crop(img: Image.Image, target_height: int, max_width: int) -> Image.Image:
    """Resize proportional to target_height, then pad to fixed (max_width, target_height)."""
    w, h = img.size
    if h <= 0 or w <= 0:
        return Image.new("RGB", (max_width, target_height), (255, 255, 255))
    new_w = max(1, int(round(w * (target_height / float(h)))))
    if new_w > max_width:
        new_w = max_width
    resized = img.resize((new_w, target_height), resample=Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (max_width, target_height), (255, 255, 255))
    canvas.paste(resized, (0, 0))
    return canvas


def make_line_crops(
    root: Path,
    rows: list[tuple[str, str]],
    px_dir: Path,
    target_height: int,
    max_width: int,
    max_label_len: int,
    fields_path: Path | None,
) -> list[tuple[str, str]]:
    line_dir = px_dir / "line_images"
    if line_dir.exists():
        shutil.rmtree(line_dir)
    line_dir.mkdir(parents=True, exist_ok=True)

    tpl_w = tpl_h = 0
    roi_fields: dict[str, dict[str, int]] = {}
    if fields_path and fields_path.is_file():
        tpl_w, tpl_h, roi_fields = _load_template_fields(fields_path)
        print(f"  crop ROI: {fields_path.name} ({tpl_w}x{tpl_h}, {len(roi_fields)} campuri)")
    else:
        raise SystemExit(
            "Lipseste template_fields.json — crop-urile pe benzi orizontale antreneaza gresit.\n"
            "Foloseste: --fields config/template_fields.json"
        )

    samples: list[tuple[str, str]] = []
    skipped_long = 0
    skipped_no_roi = 0
    for rel, transcript in rows:
        src = root / rel
        if not src.is_file():
            continue
        labeled = _line_texts_from_transcription(transcript)
        if not labeled:
            continue
        try:
            with Image.open(src) as im:
                im = im.convert("RGB")
                w, h = im.size
                if w <= 0 or h <= 0:
                    continue
                stem = Path(rel).stem
                crop_idx = 0
                for key, line in labeled:
                    if len(line) > max_label_len:
                        skipped_long += 1
                        continue
                    roi_key = TRANSCRIPT_FIELD_TO_ROI.get(key)
                    if not roi_key or roi_key not in roi_fields:
                        skipped_no_roi += 1
                        continue
                    raw_crop = _crop_from_roi(im, roi_fields[roi_key], tpl_w, tpl_h)
                    if raw_crop is None:
                        skipped_no_roi += 1
                        continue
                    crop = _resize_crop(raw_crop, target_height, max_width)
                    name = f"{stem}_{roi_key}_{crop_idx:02d}.jpg"
                    out_file = line_dir / name
                    crop.save(out_file, format="JPEG", quality=90, optimize=True)
                    samples.append((f"line_images/{name}", _normalize_label(line)))
                    crop_idx += 1
        except OSError:
            continue
    if skipped_long:
        print(
            f"  atentie: {skipped_long} line-crops omise (eticheta > {max_label_len} caractere)"
        )
    if skipped_no_roi:
        print(f"  atentie: {skipped_no_roi} randuri fara ROI mapat (date etc.)")
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
    except (OSError, subprocess.CalledProcessError):
        print(f"Symlink indisponibil, copiez {images_dir} → {link}")
        if link.exists():
            return
        shutil.copytree(images_dir, link)


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
        fields_path = Path(args.fields).resolve() if args.fields else default_fields_path()
        all_samples = make_line_crops(
            root,
            rows,
            px_dir,
            args.target_height,
            args.max_width,
            args.max_label_len,
            fields_path,
        )
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
