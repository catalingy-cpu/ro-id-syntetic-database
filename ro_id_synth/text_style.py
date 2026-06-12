"""Phase 3 — estimare stil text din regiunea originală (mărime, culoare, baseline)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ro_id_synth.template_config import FieldRect


@dataclass(frozen=True)
class TextStyle:
    font_size: int
    ink_bgr: tuple[int, int, int]
    baseline_offset: int
    letter_spacing: float


def _text_pixel_mask(patch_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    if gray.size == 0:
        return np.zeros_like(gray)
    thr = int(min(118, np.percentile(gray, 32)))
    mask = (gray < thr).astype(np.uint8) * 255
    if mask.sum() < 16:
        return np.zeros_like(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.dilate(mask, kernel, iterations=1)


def _value_band_pixel_height(patch_bgr: np.ndarray, box_h: int) -> int | None:
    """Înălțimea textului valorii (nu etichetele mici din partea de sus a ROI-ului)."""
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h < 8 or w < 8:
        return None
    y0 = int(h * 0.32)
    band = gray[y0:, :]
    thr = int(min(120, np.percentile(band, 28)))
    dark = band < thr
    if dark.sum() < 40:
        dark = band < min(140, thr + 18)
    if dark.sum() < 24:
        return None
    ys, xs = np.where(dark)
    px_h = int(ys.max() - ys.min() + 1)
    px_w = int(xs.max() - xs.min() + 1)
    if px_h < 6 or px_w < 10:
        return None
    return px_h


def _estimate_pixel_height(mask: np.ndarray, patch_bgr: np.ndarray, box_h: int) -> int | None:
    from_band = _value_band_pixel_height(patch_bgr, box_h)
    if from_band is not None:
        return from_band

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heights = [
        cv2.boundingRect(c)[3]
        for c in contours
        if cv2.contourArea(c) >= 24 and cv2.boundingRect(c)[2] >= 8
    ]
    if heights:
        return int(np.percentile(heights, 75))
    return None


def _estimate_baseline(mask: np.ndarray, box_h: int) -> int:
    ys = np.where(mask > 0)[0]
    if len(ys) == 0:
        return int(box_h * 0.72)
    return int(ys.max())


def _estimate_ink_bgr(patch_bgr: np.ndarray, mask: np.ndarray, ink_preset: str) -> tuple[int, int, int]:
    if ink_preset == "cnp_multicolor":
        return (35, 35, 200)
    if ink_preset == "red":
        return (30, 30, 195)
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    if mask.any():
        vals = gray[mask > 0]
        dark = int(np.percentile(vals, 5))
        dark = max(10, min(dark, 28))
        return (dark, dark, dark)
    return (12, 12, 12)


def _pil_size_from_pixel_height(px_h: int, kind: str, *, box_h: int, ink: str) -> int:
    if ink == "cnp_multicolor":
        scale = 1.26 if kind == "classic" else 1.18
        floor = int(box_h * 0.78)
    else:
        scale = 1.14 if kind == "classic" else 1.08
        floor = int(box_h * 0.58)
    return max(floor, max(9, int(round(px_h * scale))))


def estimate_style(
    template_bgr: np.ndarray,
    field: FieldRect,
    kind: str,
    rng: np.random.Generator,
) -> TextStyle:
    x1, y1, x2, y2 = field.to_pixels()
    patch = template_bgr[y1:y2, x1:x2]
    box_h = y2 - y1
    mask = _text_pixel_mask(patch)

    px_h = _estimate_pixel_height(mask, patch, box_h) or max(12, int(box_h * 0.68))
    base_size = _pil_size_from_pixel_height(px_h, kind, box_h=box_h, ink=field.ink)
    jitter = float(rng.uniform(0.97, 1.03))
    font_size = max(9, int(round(base_size * jitter)))

    ink = _estimate_ink_bgr(patch, mask, field.ink)
    baseline = _estimate_baseline(mask, box_h)
    spacing = float(rng.uniform(0.95, 1.05))

    return TextStyle(
        font_size=font_size,
        ink_bgr=ink,
        baseline_offset=baseline,
        letter_spacing=spacing,
    )
