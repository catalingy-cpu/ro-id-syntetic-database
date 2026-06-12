"""Randare text integrat natural în document — fără benzi albe."""

from __future__ import annotations

import os
import platform

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ro_id_synth.font_metrics import target_font_size
from ro_id_synth.inpaint import inpaint_text_regions
from ro_id_synth.records import SyntheticIdRecord
from ro_id_synth.roi import FieldRoi, TemplateConfig

_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _font_candidates(kind: str) -> list[str]:
    win = os.environ.get("WINDIR", r"C:\Windows")
    fonts_dir = os.path.join(win, "Fonts")
    if kind == "classic":
        names = ["arialbd.ttf", "Arialbd.ttf", "calibrib.ttf", "consolab.ttf"]
    else:
        names = ["calibrib.ttf", "arialbd.ttf", "arialn.ttf", "calibri.ttf"]
    if platform.system() == "Windows":
        return [os.path.join(fonts_dir, n) for n in names] + names
    return names + ["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"]


def _resolve_font(size: int, kind: str = "classic") -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    key = (kind, size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    for path in _font_candidates(kind):
        try:
            font = ImageFont.truetype(path, size=size)
            _FONT_CACHE[key] = font
            return font
        except OSError:
            continue
    font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def _resolve_ink_bgr(template_bgr: np.ndarray, roi: FieldRoi, preset: str, kind: str) -> tuple[int, int, int]:
    if preset == "red":
        return (30, 30, 195) if kind == "classic" else (35, 35, 190)
    h, w = template_bgr.shape[:2]
    x1, y1, x2, y2 = roi.to_pixels(w, h)
    patch = template_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return (18, 18, 18)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    dark = patch[gray < np.percentile(gray, 42)]
    if dark.size == 0:
        dark = patch.reshape(-1, 3)
    med = np.median(dark, axis=0)
    return int(med[0]), int(med[1]), int(med[2])


def _fit_font_size(
    text: str,
    box_w: int,
    target: int,
    kind: str,
    *,
    min_size: int = 8,
) -> int:
    """Păstrează dimensiunea țintă (±10% deja aplicat); micșorează doar dacă depășește lățimea."""
    size = target
    for _ in range(12):
        font = _resolve_font(size, kind)
        bbox = font.getbbox(text)
        tw = bbox[2] - bbox[0]
        if tw <= int(box_w * 0.98) or size <= min_size:
            return size
        size -= 1
    return max(min_size, size)


def _draw_field_with_key(
    canvas_bgr: np.ndarray,
    template_bgr: np.ndarray,
    roi: FieldRoi,
    field_key: str,
    text: str,
    kind: str,
    template_key: str,
    rng: np.random.Generator,
) -> None:
    h, w = canvas_bgr.shape[:2]
    x1, y1, x2, y2 = roi.to_pixels(w, h)
    box_w, box_h = x2 - x1, y2 - y1
    if box_w < 4 or box_h < 4 or not text.strip():
        return

    ink_bgr = _resolve_ink_bgr(template_bgr, roi, roi.ink, kind)
    target = target_font_size(template_key, field_key, roi, box_h, kind, rng)
    size = _fit_font_size(text, box_w, target, kind)
    font = _resolve_font(size, kind)
    bbox = font.getbbox(text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    if roi.align == "center":
        tx = x1 + max(0, (box_w - tw) // 2)
    else:
        tx = x1 + max(0, int((x2 - x1 - tw) * 0.01))
    ty = y1 + max(0, (box_h - th) // 2) - bbox[1]

    sub = canvas_bgr[y1:y2, x1:x2].copy()
    sub_rgb = cv2.cvtColor(sub, cv2.COLOR_BGR2RGB)
    pil_sub = Image.fromarray(sub_rgb)
    overlay = Image.new("RGBA", pil_sub.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.text((tx - x1, ty - y1), text, font=font, fill=(ink_bgr[2], ink_bgr[1], ink_bgr[0], 240))
    ov = np.array(overlay)
    alpha = ov[:, :, 3:4].astype(np.float32) / 255.0
    color = ov[:, :, :3].astype(np.float32)
    base = np.array(pil_sub).astype(np.float32)
    blended = base * (1.0 - alpha) + color * alpha
    result = np.clip(blended, 0, 255).astype(np.uint8)
    canvas_bgr[y1:y2, x1:x2] = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def _field_values(cfg: TemplateConfig, record: SyntheticIdRecord) -> list[tuple[str, str]]:
    doc_id = f"{record.serie}{record.numar}"
    if cfg.kind == "eid":
        return [
            ("nume", record.nume),
            ("prenume", record.prenume),
            ("sex", record.sex),
            ("cetatenie", record.cetatenie_short),
            ("data_nasterii", record.data_nasterii_display),
            ("cnp", record.cnp),
            ("document_id", doc_id),
        ]
    return [
        ("serie", record.serie),
        ("numar", record.numar),
        ("cnp", record.cnp),
        ("nume", record.nume),
        ("prenume", record.prenume),
        ("cetatenie", record.cetatenie),
        ("loc_nastere", record.loc_nastere),
        ("sex", record.sex),
        ("adresa_1", record.adresa_line1),
        ("adresa_2", record.adresa_line2),
    ]


def render_record_on_template(
    template_bgr: np.ndarray,
    cfg: TemplateConfig,
    record: SyntheticIdRecord,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Template → șterge text specimen → randare text fictiv (card aproape identic)."""
    if rng is None:
        rng = np.random.default_rng()
    img = inpaint_text_regions(template_bgr, cfg.fields)

    for key, text in _field_values(cfg, record):
        roi = cfg.fields.get(key)
        if roi is None or not text:
            continue
        _draw_field_with_key(img, template_bgr, roi, key, text, cfg.kind, cfg.key, rng)

    return img
