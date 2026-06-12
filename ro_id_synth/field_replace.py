"""Phase 2 — înlocuire câmpuri: mască locală + inpaint doar pixeli text."""

from __future__ import annotations

import cv2
import numpy as np

from ro_id_synth.template_config import FieldRect, TemplateSpec


def _text_pixel_mask(patch_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    if gray.size == 0:
        return np.zeros_like(gray)
    thr = int(min(115, np.percentile(gray, 48)))
    mask = (gray < thr).astype(np.uint8) * 255
    # Captură și text roșu CNP
    b, g, r = cv2.split(patch_bgr)
    red_text = (r.astype(np.int16) - np.maximum(g, b).astype(np.int16)) > 35
    mask = np.maximum(mask, (red_text.astype(np.uint8) * 255))
    if mask.sum() < 12:
        return np.zeros_like(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(mask, kernel, iterations=2)


def _restore_texture(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return img_bgr
    out = img_bgr.copy()
    ys, xs = np.where(mask > 0)
    x1, x2 = max(0, xs.min() - 3), min(img_bgr.shape[1], xs.max() + 4)
    y1, y2 = max(0, ys.min() - 3), min(img_bgr.shape[0], ys.max() + 4)
    region = out[y1:y2, x1:x2].astype(np.float32)
    m = mask[y1:y2, x1:x2] > 0
    if not np.any(m):
        return out
    border = region[~m] if np.any(~m) else region.reshape(-1, 3)
    std = float(np.std(border)) + 0.5
    noise = np.random.normal(0, std * 0.22, region.shape)
    region[m] += noise[m]
    out[y1:y2, x1:x2] = np.clip(region, 0, 255).astype(np.uint8)
    return out


def _inpaint_field(img_bgr: np.ndarray, field: FieldRect, *, radius: int = 3) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = field.to_pixels()
    patch = img_bgr[y1:y2, x1:x2]
    local = _text_pixel_mask(patch)
    if not local.any():
        return img_bgr

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = local

    out = cv2.inpaint(img_bgr, full_mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    out = cv2.inpaint(out, full_mask, inpaintRadius=max(2, radius - 1), flags=cv2.INPAINT_NS)
    return _restore_texture(out, full_mask)


def replace_all_fields(template_bgr: np.ndarray, spec: TemplateSpec) -> np.ndarray:
    """Șterge textul original din fiecare câmp — păstrează guilloché și grafică."""
    out = template_bgr.copy()
    for field in spec.fields.values():
        out = _inpaint_field(out, field)
    return out
