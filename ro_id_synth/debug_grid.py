"""Contact sheets pentru inspecție vizuală."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from ro_id_synth.worker import generate_one_image, init_pool  # noqa: F401


def _render_grid(
    images: list[np.ndarray],
    output_path: Path,
    *,
    cols: int = 10,
    thumb_w: int = 360,
    thumb_h: int = 220,
) -> Path:
    count = len(images)
    rows = math.ceil(count / cols)
    canvas = np.full((rows * thumb_h, cols * thumb_w, 3), 48, dtype=np.uint8)
    for i, img in enumerate(images):
        thumb = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, cols)
        y0, x0 = r * thumb_h, c * thumb_w
        canvas[y0 : y0 + thumb_h, x0 : x0 + thumb_w] = thumb
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tofile(str(output_path))
    return output_path


def build_debug_grid(
    config_path: str,
    base_dir: str,
    output_path: Path,
    *,
    count: int = 100,
    cols: int = 10,
    seed: int = 42,
    thumb_w: int = 360,
    thumb_h: int = 220,
) -> Path:
    init_pool(config_path, base_dir)
    rng = np.random.default_rng(seed)
    images: list[np.ndarray] = []
    for i in range(count):
        img, _, _ = generate_one_image(i, seed, rng, skip_quality=True)
        images.append(img)
    return _render_grid(images, output_path, cols=cols, thumb_w=thumb_w, thumb_h=thumb_h)


def build_debug_grid_at(
    config_path: str,
    base_dir: str,
    output_path: Path,
    *,
    count: int = 100,
    seed: int = 42,
) -> Path:
    return build_debug_grid(config_path, base_dir, output_path, count=count, seed=seed)
