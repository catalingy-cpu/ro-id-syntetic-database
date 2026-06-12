"""Estimare dimensiune font din textul original din template (±10% la randare)."""

from __future__ import annotations

import cv2
import numpy as np

from ro_id_synth.roi import FieldRoi, TemplateConfig

_FONT_METRICS: dict[tuple[str, str], int] = {}


def _text_pixel_height(patch_bgr: np.ndarray) -> int | None:
    if patch_bgr.size == 0:
        return None
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    thr = int(np.percentile(gray, 38))
    mask = (gray < thr).astype(np.uint8) * 255
    if mask.sum() < 30:
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heights: list[int] = []
    for cnt in contours:
        _x, _y, _w, h = cv2.boundingRect(cnt)
        if h >= 4 and cv2.contourArea(cnt) >= 12:
            heights.append(h)
    if not heights:
        return None
    return int(np.median(heights))


def estimate_pil_font_size(pixel_height: int, kind: str) -> int:
    """Conversie înălțime pixeli → size PIL (calibrat pe template-uri specimen)."""
    scale = 1.05 if kind == "eid" else 1.12
    return max(8, int(round(pixel_height * scale)))


def build_font_metrics(template_bgr: np.ndarray, cfg: TemplateConfig) -> None:
    h, w = template_bgr.shape[:2]
    for name, roi in cfg.fields.items():
        x1, y1, x2, y2 = roi.to_pixels(w, h)
        patch = template_bgr[y1:y2, x1:x2]
        px_h = _text_pixel_height(patch)
        if px_h is None:
            px_h = max(8, int((y2 - y1) * roi.font_ratio * 0.82))
        size = estimate_pil_font_size(px_h, cfg.kind)
        _FONT_METRICS[(cfg.key, name)] = size


def target_font_size(
    template_key: str,
    field_name: str,
    roi: FieldRoi,
    box_h: int,
    kind: str,
    rng: np.random.Generator,
) -> int:
    base = _FONT_METRICS.get((template_key, field_name))
    cap = estimate_pil_font_size(max(8, int(box_h * roi.font_ratio * 0.78)), kind)
    if base is None:
        base = cap
    else:
        base = min(base, cap)
    jitter = float(rng.uniform(0.90, 1.10))
    return max(8, int(round(base * jitter)))


def clear_font_metrics() -> None:
    _FONT_METRICS.clear()
