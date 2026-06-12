"""Texturi de fundal realiste — țesătură, birou, bokeh, hârtie."""

from __future__ import annotations

import cv2
import numpy as np


def _perlin_noise_2d(shape: tuple[int, int], scale: float, rng: np.random.Generator) -> np.ndarray:
    h, w = shape
    gh = max(2, int(h / scale) + 2)
    gw = max(2, int(w / scale) + 2)
    grid = rng.random((gh, gw), dtype=np.float32)
    ys = np.linspace(0, gh - 1, h, dtype=np.float32)
    xs = np.linspace(0, gw - 1, w, dtype=np.float32)
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    x1 = np.clip(x0 + 1, 0, gw - 1)
    y1 = np.clip(y0 + 1, 0, gh - 1)
    dx = (xs - x0)[None, :]
    dy = (ys - y0)[:, None]
    n00 = grid[y0[:, None], x0[None, :]]
    n10 = grid[y1[:, None], x0[None, :]]
    n01 = grid[y0[:, None], x1[None, :]]
    n11 = grid[y1[:, None], x1[None, :]]
    nx0 = n00 * (1 - dx) + n01 * dx
    nx1 = n10 * (1 - dx) + n11 * dx
    return nx0 * (1 - dy) + nx1 * dy


def _fabric_texture(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Țesătură / material — ca fundalul pozelor cu mână."""
    hue = rng.choice(
        [
            (168, 155, 142),
            (142, 138, 134),
            (98, 92, 88),
            (185, 172, 158),
            (120, 108, 98),
            (210, 198, 188),
        ]
    )
    base = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        n1 = _perlin_noise_2d((h, w), rng.uniform(18, 32), rng)
        n2 = _perlin_noise_2d((h, w), rng.uniform(4, 8), rng)
        wx = np.sin(np.linspace(0, rng.uniform(40, 80), w)[None, :] + n2 * 2.5)
        wy = np.sin(np.linspace(0, rng.uniform(30, 60), h)[:, None] + n2 * 2.0)
        weave = wx * wy
        base[:, :, c] = hue[c] + n1 * 22 + weave * 8
    noise = rng.normal(0, 5, (h, w, 3))
    out = np.clip(base + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.6, 1.8)))
    return out


def _desk_wood(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    n = _perlin_noise_2d((h, w), rng.uniform(12, 24), rng)
    wave = np.linspace(0, rng.uniform(10, 22), w)[None, :]
    grain = np.sin(wave + n * 3.0)
    tone = rng.uniform(85, 125)
    gray = tone + grain * rng.uniform(18, 32) + n * 20
    img = np.stack([gray * 0.82, gray * 0.68, gray * 0.48], axis=-1)
    noise = rng.normal(0, 6, (h, w, 3))
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def _bokeh_background(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Fundal estompat tip cameră telefon."""
    c1 = rng.integers(60, 140, size=3)
    c2 = rng.integers(80, 180, size=3)
    n = _perlin_noise_2d((h, w), rng.uniform(20, 40), rng)
    blend = n[..., None]
    img = c1 * (1 - blend) + c2 * blend
    for _ in range(int(rng.integers(4, 10))):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(20, min(h, w) // 5))
        color = rng.integers(140, 230, size=3)
        cv2.circle(img.astype(np.uint8), (cx, cy), r, color.tolist(), -1, lineType=cv2.LINE_AA)
    out = cv2.GaussianBlur(img.astype(np.uint8), (0, 0), float(rng.uniform(8, 18)))
    vignette = np.linspace(0.75, 1.05, h, dtype=np.float32)[:, None]
    return np.clip(out.astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)


def _paper_scan(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Suprafață deschisă — scan / masă deschisă."""
    tone = rng.integers(215, 242)
    img = np.full((h, w, 3), tone, dtype=np.float32)
    n = _perlin_noise_2d((h, w), rng.uniform(30, 60), rng)
    img += n[..., None] * rng.uniform(4, 10)
    noise = rng.normal(0, 3, (h, w, 3))
    out = np.clip(img + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        angle = float(rng.uniform(-1.5, 1.5))
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out = cv2.warpAffine(out, m, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return out


def _dark_surface(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    tone = rng.integers(28, 55)
    n = _perlin_noise_2d((h, w), rng.uniform(14, 28), rng)
    img = tone + n * 18
    out = np.stack([img * 0.95, img * 0.92, img * 0.88], axis=-1)
    noise = rng.normal(0, 4, (h, w, 3))
    return np.clip(out + noise, 0, 255).astype(np.uint8)


def sample_realworld_background(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    kind = int(rng.integers(0, 5))
    if kind == 0:
        return _fabric_texture(h, w, rng)
    if kind == 1:
        return _desk_wood(h, w, rng)
    if kind == 2:
        return _bokeh_background(h, w, rng)
    if kind == 3:
        return _paper_scan(h, w, rng)
    return _dark_surface(h, w, rng)
