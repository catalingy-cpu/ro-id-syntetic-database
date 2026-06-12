"""Augmentări ușoare — pipeline principal e în scenarios.py (după randare)."""

from __future__ import annotations

import cv2
import numpy as np


def light_sharpen(img_bgr: np.ndarray, amount: float = 0.15) -> np.ndarray:
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), 1.0)
    out = cv2.addWeighted(img_bgr, 1.0 + amount, blurred, -amount, 0)
    return np.clip(out, 0, 255).astype(np.uint8)
