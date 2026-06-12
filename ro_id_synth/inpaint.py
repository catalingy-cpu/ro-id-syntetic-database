"""Ștergere text original — doar pixelii întunecați, păstrează textura guilloché."""

from __future__ import annotations

import cv2
import numpy as np

from ro_id_synth.roi import FieldRoi


def _text_pixel_mask(patch_bgr: np.ndarray) -> np.ndarray:
    """Mască pixeli text (întunecați), nu întreg ROI-ul."""
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    if gray.size == 0:
        return np.zeros_like(gray)
    thr = int(np.percentile(gray, 40))
    mask = (gray < thr).astype(np.uint8) * 255
    if mask.sum() < 20:
        return np.zeros_like(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _field_text_mask(img_bgr: np.ndarray, roi: FieldRoi) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = roi.to_pixels(w, h)
    patch = img_bgr[y1:y2, x1:x2]
    local = _text_pixel_mask(patch)
    if local.any():
        mask[y1:y2, x1:x2] = local
    return mask


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
    noise = np.random.normal(0, std * 0.25, region.shape)
    region[m] += noise[m]
    out[y1:y2, x1:x2] = np.clip(region, 0, 255).astype(np.uint8)
    return out


def inpaint_text_regions(
    img_bgr: np.ndarray,
    fields: dict[str, FieldRoi],
    *,
    radius: int = 2,
) -> np.ndarray:
    """Elimină textul specimen câmp cu câmp — fără dreptunghiuri albe."""
    out = img_bgr.copy()
    for roi in fields.values():
        mask = _field_text_mask(out, roi)
        if not np.any(mask):
            continue
        out = cv2.inpaint(out, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
        out = _restore_texture(out, mask)
    return out
