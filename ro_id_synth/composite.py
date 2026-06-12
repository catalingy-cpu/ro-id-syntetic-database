"""Compunere foto realistă: umbră, lumină, deget, blend card în scenă."""

from __future__ import annotations

import cv2
import numpy as np


def _feather_mask(h: int, w: int, feather: int = 3) -> np.ndarray:
    mask = np.ones((h, w), dtype=np.float32)
    if feather <= 0:
        return mask
    for i in range(feather):
        alpha = (i + 1) / feather
        mask[i, :] *= alpha
        mask[-(i + 1), :] *= alpha
        mask[:, i] *= alpha
        mask[:, -(i + 1)] *= alpha
    return mask


def _drop_shadow(shape: tuple[int, int], card_wh: tuple[int, int], pos: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    h, w = shape
    ch, cw = card_wh
    x, y = pos
    shadow = np.zeros((h, w), dtype=np.float32)
    ox = int(rng.integers(4, 14))
    oy = int(rng.integers(5, 16))
    sx, sy = x + ox, y + oy
    if sx + cw > 0 and sy + ch > 0 and sx < w and sy < h:
        x1, y1 = max(0, sx), max(0, sy)
        x2, y2 = min(w, sx + cw), min(h, sy + ch)
        patch = np.ones((y2 - y1, x2 - x1), dtype=np.float32) * float(rng.uniform(0.25, 0.45))
        shadow[y1:y2, x1:x2] = np.maximum(shadow[y1:y2, x1:x2], patch)
    blur = int(rng.integers(8, 18)) | 1
    return cv2.GaussianBlur(shadow, (blur, blur), 0)


def _color_grade_card(card: np.ndarray, bg_patch: np.ndarray, strength: float) -> np.ndarray:
    if bg_patch.size == 0 or strength <= 0:
        return card
    bg_mean = bg_patch.reshape(-1, 3).mean(axis=0)
    card_f = card.astype(np.float32)
    card_mean = card_f.reshape(-1, 3).mean(axis=0)
    delta = (bg_mean - card_mean) * strength
    return np.clip(card_f + delta, 0, 255).astype(np.uint8)


def paste_card(
    background: np.ndarray,
    card: np.ndarray,
    pos: tuple[int, int],
    rng: np.random.Generator,
    *,
    shadow: bool = True,
    feather: int = 2,
) -> np.ndarray:
    """Lipește cardul cu umbră și margini estompate."""
    out = background.copy()
    h, w = out.shape[:2]
    ch, cw = card.shape[:2]
    x, y = pos
    x = int(np.clip(x, 0, max(0, w - cw)))
    y = int(np.clip(y, 0, max(0, h - ch)))

    if shadow:
        sh = _drop_shadow((h, w), (ch, cw), (x, y), rng)
        factor = np.clip(1.0 - sh * 0.55, 0, 1)[..., None]
        out = np.clip(out.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    bg_patch = out[y : y + ch, x : x + cw]
    graded = _color_grade_card(card, bg_patch, float(rng.uniform(0.08, 0.22)))
    alpha = _feather_mask(ch, cw, feather)[..., None]
    blended = (
        bg_patch.astype(np.float32) * (1 - alpha) + graded.astype(np.float32) * alpha
    ).astype(np.uint8)
    out[y : y + ch, x : x + cw] = blended
    return out


def add_finger_hold(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulează degetul care ține cardul (colț inferior)."""
    h, w = img.shape[:2]
    side = int(rng.integers(0, 2))
    fw = int(w * rng.uniform(0.14, 0.26))
    fh = int(h * rng.uniform(0.10, 0.20))
    skin = np.array(
        [rng.integers(175, 210), rng.integers(125, 155), rng.integers(95, 125)],
        dtype=np.float32,
    )
    if side == 0:
        x0, y0 = int(rng.integers(-fw // 4, fw // 4)), h - fh - int(rng.integers(0, h // 12))
    else:
        x0, y0 = w - fw - int(rng.integers(0, fw // 4)), h - fh - int(rng.integers(0, h // 12))
    x0 = int(np.clip(x0, 0, w - fw))
    y0 = int(np.clip(y0, 0, h - fh))

    overlay = img.astype(np.float32).copy()
    yy, xx = np.mgrid[0:fh, 0:fw]
    cx, cy = fw * 0.55, fh * 0.45
    dist = np.sqrt(((xx - cx) / fw) ** 2 + ((yy - cy) / fh) ** 2)
    finger_alpha = np.clip(1.0 - dist * 1.15, 0, 1) * float(rng.uniform(0.55, 0.82))
    finger_alpha = cv2.GaussianBlur(finger_alpha.astype(np.float32), (0, 0), 2.5)

    region = overlay[y0 : y0 + fh, x0 : x0 + fw]
    for c in range(3):
        region[:, :, c] = region[:, :, c] * (1 - finger_alpha) + skin[c] * finger_alpha
    overlay[y0 : y0 + fh, x0 : x0 + fw] = region
    return np.clip(overlay, 0, 255).astype(np.uint8)


def add_camera_glare(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    overlay = img.astype(np.float32)
    cx = int(rng.integers(int(w * 0.15), int(w * 0.85)))
    cy = int(rng.integers(int(h * 0.1), int(h * 0.6)))
    radius = int(rng.integers(min(h, w) // 6, min(h, w) // 3))
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
    glare = np.clip(1.0 - dist / radius, 0, 1) ** 2
    glare *= float(rng.uniform(0.12, 0.35))
    for c in range(3):
        overlay[:, :, c] = np.clip(overlay[:, :, c] + glare * 80, 0, 255)
    return overlay.astype(np.uint8)


def add_indoor_lighting(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    grad = np.linspace(float(rng.uniform(0.82, 0.95)), float(rng.uniform(1.02, 1.12)), h, dtype=np.float32)
    if rng.random() < 0.5:
        grad = grad[::-1]
    out = img.astype(np.float32) * grad[:, None, None]
    return np.clip(out, 0, 255).astype(np.uint8)
