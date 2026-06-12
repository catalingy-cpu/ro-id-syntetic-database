"""Phase 3 — randare text realist pe card (fără dreptunghiuri albe)."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ro_id_synth.fonts import resolve_font
from ro_id_synth.records import SyntheticIdRecord
from ro_id_synth.template_config import FieldRect, TemplateSpec
from ro_id_synth.text_style import TextStyle, estimate_style

CNP_COLORS_RGB = [
    (30, 70, 200),
    (200, 40, 40),
    (30, 70, 200),
    (200, 40, 40),
    (30, 70, 200),
    (200, 40, 40),
    (20, 20, 20),
    (20, 20, 20),
    (20, 20, 20),
    (20, 20, 20),
    (20, 20, 20),
    (20, 20, 20),
    (20, 20, 20),
]


def _fit_width(
    text: str,
    max_w: int,
    size: int,
    kind: str,
    spacing: float,
    *,
    min_size: int = 8,
) -> int:
    floor = max(min_size, size)
    for s in range(floor, min_size - 1, -1):
        font = resolve_font(s, kind)
        w = _text_width(text, font, spacing)
        if w <= int(max_w * 0.98):
            return s
    return min_size


def _text_width(text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, spacing: float) -> int:
    if spacing == 1.0:
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0]
    total = 0
    for i, ch in enumerate(text):
        bbox = font.getbbox(ch)
        total += int((bbox[2] - bbox[0]) * spacing)
    return total


def _draw_text_line(
    canvas_bgr: np.ndarray,
    x: int,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    ink_rgb: tuple[int, int, int],
    spacing: float,
) -> None:
    h, w = canvas_bgr.shape[:2]
    sub_x1, sub_y1 = max(0, x), max(0, y - 2)
    sub_x2 = min(w, x + _text_width(text, font, spacing) + 8)
    sub_y2 = min(h, y + int(font.size * 1.4))
    if sub_x2 <= sub_x1 or sub_y2 <= sub_y1:
        return

    sub = canvas_bgr[sub_y1:sub_y2, sub_x1:sub_x2].copy()
    pil = Image.fromarray(cv2.cvtColor(sub, cv2.COLOR_BGR2RGB))
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cx = x - sub_x1
    cy = y - sub_y1
    alpha = 255
    if spacing == 1.0:
        draw.text((cx, cy), text, font=font, fill=(*ink_rgb, alpha))
    else:
        pen_x = cx
        for ch in text:
            draw.text((pen_x, cy), ch, font=font, fill=(*ink_rgb, alpha))
            bbox = font.getbbox(ch)
            pen_x += int((bbox[2] - bbox[0]) * spacing)

    ov = np.array(overlay)
    alpha = ov[:, :, 3:4].astype(np.float32) / 255.0
    color = ov[:, :, :3].astype(np.float32)
    base = np.array(pil).astype(np.float32)
    blended = np.clip(base * (1.0 - alpha) + color * alpha, 0, 255).astype(np.uint8)
    canvas_bgr[sub_y1:sub_y2, sub_x1:sub_x2] = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)


def _draw_cnp_multicolor(
    canvas_bgr: np.ndarray,
    field: FieldRect,
    cnp: str,
    style: TextStyle,
    kind: str,
) -> None:
    x1, y1, x2, y2 = field.to_pixels()
    box_w = x2 - x1
    box_h = y2 - y1
    min_cnp = max(style.font_size, int(box_h * 0.78))
    size = _fit_width(
        cnp,
        box_w,
        style.font_size,
        kind,
        style.letter_spacing,
        min_size=min_cnp,
    )
    font = resolve_font(size, kind)
    bbox = font.getbbox("0")
    th = bbox[3] - bbox[1]
    ty = y1 + max(0, (y2 - y1 - th) // 2) - bbox[1]
    pen_x = x1 + 2
    for i, ch in enumerate(cnp[:13]):
        color = CNP_COLORS_RGB[min(i, len(CNP_COLORS_RGB) - 1)]
        _draw_text_line(canvas_bgr, pen_x, ty, ch, font, color, 1.0)
        cb = font.getbbox(ch)
        pen_x += cb[2] - cb[0]


def _draw_field(
    canvas_bgr: np.ndarray,
    template_bgr: np.ndarray,
    field: FieldRect,
    text: str,
    spec: TemplateSpec,
    rng: np.random.Generator,
) -> None:
    if not text.strip():
        return
    style = estimate_style(template_bgr, field, spec.kind, rng)

    if field.ink == "cnp_multicolor" and text.isdigit():
        _draw_cnp_multicolor(canvas_bgr, field, text, style, spec.kind)
        return

    x1, y1, x2, y2 = field.to_pixels()
    box_w, box_h = x2 - x1, y2 - y1
    min_size = max(9, int(style.font_size * 0.92), int(box_h * 0.52))
    size = _fit_width(
        text,
        box_w,
        style.font_size,
        spec.kind,
        style.letter_spacing,
        min_size=min_size,
    )
    font = resolve_font(size, spec.kind)
    tw = _text_width(text, font, style.letter_spacing)
    bbox = font.getbbox(text)
    th = bbox[3] - bbox[1]

    if field.align == "center":
        tx = x1 + max(0, (box_w - tw) // 2)
    else:
        tx = x1 + max(0, int(box_w * 0.01))
    ty = y1 + max(0, (box_h - th) // 2) - bbox[1]

    ink_rgb = (style.ink_bgr[2], style.ink_bgr[1], style.ink_bgr[0])
    _draw_text_line(canvas_bgr, tx, ty, text, font, ink_rgb, style.letter_spacing)


def field_values(record: SyntheticIdRecord) -> dict[str, str]:
    doc_id = f"{record.serie}{record.numar}"
    return {
        "series": record.serie,
        "serie": record.serie,
        "number": record.numar,
        "numar": record.numar,
        "cnp": record.cnp,
        "surname": record.nume,
        "nume": record.nume,
        "given_name": record.prenume,
        "prenume": record.prenume,
        "nationality": record.cetatenie,
        "cetatenie": record.cetatenie,
        "birth_place": record.loc_nastere,
        "loc_nastere": record.loc_nastere,
        "address_line1": record.adresa_line1,
        "address_line2": record.adresa_line2,
        "adresa_1": record.adresa_line1,
        "adresa_2": record.adresa_line2,
        "issued_by": record.issued_by,
        "validity": record.validity_display,
        "sex": record.sex,
        "data_nasterii": record.birth_date_display,
        "document_id": doc_id,
    }


def render_text_on_card(
    canvas_bgr: np.ndarray,
    template_bgr: np.ndarray,
    spec: TemplateSpec,
    record: SyntheticIdRecord,
    rng: np.random.Generator,
) -> np.ndarray:
    out = canvas_bgr.copy()
    values = field_values(record)
    for key, field in spec.fields.items():
        text = values.get(key, "")
        if text:
            _draw_field(out, template_bgr, field, text, spec, rng)
    return out
