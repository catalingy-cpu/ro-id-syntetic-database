"""
Scenarii foto realiste — realism > diversitate augmentare.

Distribuție dificultate: 70% clean, 20% medium, 10% hard.
Scară card: 70% la 60–95%, 20% la 40–60%, 10% la 15–40% lățime imagine.

Pipeline: card randat → plasare fundal → efecte cameră (doar la final).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from ro_id_synth.composite import add_camera_glare, add_finger_hold, add_indoor_lighting, paste_card
from ro_id_synth.textures import sample_realworld_background

Difficulty = Literal["clean", "medium", "hard"]


@dataclass(frozen=True)
class SceneParams:
    difficulty: Difficulty
    card_width_frac: float
    background_kind: int


def pick_difficulty(rng: np.random.Generator) -> Difficulty:
    roll = float(rng.random())
    if roll < 0.80:
        return "clean"
    if roll < 0.95:
        return "medium"
    return "hard"


def pick_card_width_fraction(rng: np.random.Generator) -> float:
    roll = float(rng.random())
    if roll < 0.70:
        return float(rng.uniform(0.60, 0.95))
    if roll < 0.90:
        return float(rng.uniform(0.40, 0.60))
    return float(rng.uniform(0.15, 0.40))


def _canvas_size(
    card_w: int,
    card_h: int,
    width_frac: float,
    difficulty: Difficulty,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Canvas compact — cardul ocupă width_frac din lățime, fără fundaluri uriașe goale."""
    frac = max(0.15, min(0.98, width_frac))
    out_w = max(card_w + 20, int(card_w / frac))
    if difficulty == "clean":
        v_margin = float(rng.uniform(1.04, 1.14))
    elif difficulty == "medium":
        v_margin = float(rng.uniform(1.08, 1.28))
    else:
        v_margin = float(rng.uniform(1.12, 1.45))
    out_h = max(card_h + 16, int(card_h * v_margin))
    return out_w, out_h


def _resize_card_to_width(card: np.ndarray, target_w: int) -> np.ndarray:
    ch, cw = card.shape[:2]
    if cw == target_w:
        return card
    scale = target_w / cw
    nh = max(20, int(ch * scale))
    return cv2.resize(card, (target_w, nh), interpolation=cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA)


def _mild_perspective(img: np.ndarray, rng: np.random.Generator, pct: float) -> np.ndarray:
    h, w = img.shape[:2]
    if min(h, w) < 80 or pct <= 0:
        return img
    dx = max(1, int(w * pct))
    dy = max(1, int(h * pct))
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32(
        [
            [rng.integers(0, dx), rng.integers(0, dy)],
            [w - rng.integers(0, dx), rng.integers(0, dy)],
            [w - rng.integers(0, dx), h - rng.integers(0, dy)],
            [rng.integers(0, dx), h - rng.integers(0, dy)],
        ]
    )
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _motion_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k = int(rng.integers(3, 6)) | 1
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)


def _partial_crop(img: np.ndarray, rng: np.random.Generator, max_pct: float = 0.08) -> np.ndarray:
    h, w = img.shape[:2]
    left = int(w * rng.uniform(0, max_pct))
    top = int(h * rng.uniform(0, max_pct * 0.6))
    right = int(w * rng.uniform(0, max_pct))
    bottom = int(h * rng.uniform(0, max_pct * 0.6))
    return img[top : h - bottom, left : w - right]


def _sample_background(h: int, w: int, difficulty: Difficulty, rng: np.random.Generator) -> np.ndarray:
    from ro_id_synth.textures import _desk_wood, _fabric_texture, _paper_scan

    if difficulty == "clean":
        kind = int(rng.integers(0, 3))
        if kind == 0:
            return _paper_scan(h, w, rng)
        if kind == 1:
            return _desk_wood(h, w, rng)
        return _fabric_texture(h, w, rng)
    return sample_realworld_background(h, w, rng)


def place_card_on_background(card: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, SceneParams]:
    """Plasează cardul randat pe fundal realist."""
    difficulty = pick_difficulty(rng)
    width_frac = pick_card_width_fraction(rng)
    ch, cw = card.shape[:2]
    out_w, out_h = _canvas_size(cw, ch, width_frac, difficulty, rng)
    target_w = max(60, int(out_w * width_frac))
    resized = _resize_card_to_width(card, target_w)

    bg_kind = int(rng.integers(0, 5))
    bg = _sample_background(out_h, out_w, difficulty, rng)

    rh, rw = resized.shape[:2]
    margin_x = max(0, out_w - rw)
    margin_y = max(0, out_h - rh)
    x = int(rng.integers(0, max(1, margin_x)))
    y = int(rng.integers(0, max(1, margin_y)))

    use_shadow = difficulty != "clean" or rng.random() < 0.35
    use_finger = difficulty == "hard" and rng.random() < 0.45
    if difficulty == "clean" and rng.random() < 0.55:
        # Scan / masă neutră — card mare, fundal discret
        from ro_id_synth.textures import _paper_scan

        bg = _paper_scan(out_h, out_w, rng)
        x = (out_w - rw) // 2
        y = (out_h - rh) // 2
        use_shadow = False

    scene = paste_card(bg, resized, (x, y), rng, shadow=use_shadow, feather=1 if difficulty == "clean" else 2)
    if use_finger:
        scene = add_finger_hold(scene, rng)
    if difficulty == "medium" and rng.random() < 0.25:
        scene = add_camera_glare(scene, rng)
    if difficulty == "hard" and rng.random() < 0.35:
        scene = add_camera_glare(scene, rng)

    params = SceneParams(difficulty=difficulty, card_width_frac=rw / out_w, background_kind=bg_kind)
    return scene, params


def apply_camera_effects(img: np.ndarray, params: SceneParams, rng: np.random.Generator) -> np.ndarray:
    """Efecte cameră — doar după plasarea cardului."""
    out = img
    d = params.difficulty

    if d == "clean":
        if rng.random() < 0.15:
            out = _mild_perspective(out, rng, 0.003)
        if rng.random() < 0.20:
            out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.15, 0.35)))
        if rng.random() < 0.50:
            q = int(rng.integers(88, 96))
            ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if ok:
                out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if rng.random() < 0.30:
            out = add_indoor_lighting(out, rng)
        return out

    if d == "medium":
        if rng.random() < 0.55:
            out = _mild_perspective(out, rng, float(rng.uniform(0.008, 0.018)))
        if rng.random() < 0.70:
            out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.35, 0.75)))
        if rng.random() < 0.85:
            q = int(rng.integers(72, 88))
            ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if ok:
                out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if rng.random() < 0.50:
            beta = float(rng.uniform(0.94, 1.06))
            out = np.clip(out.astype(np.float32) * beta, 0, 255).astype(np.uint8)
        return add_indoor_lighting(out, rng)

    # hard — dar textul rămâne lizibil
    if rng.random() < 0.65:
        out = _mild_perspective(out, rng, float(rng.uniform(0.015, 0.035)))
    if rng.random() < 0.55:
        out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.6, 1.2)))
    if rng.random() < 0.35:
        out = _motion_blur(out, rng)
    if rng.random() < 0.90:
        q = int(rng.integers(62, 82))
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if rng.random() < 0.45:
        beta = float(rng.uniform(0.82, 0.96))
        out = np.clip(out.astype(np.float32) * beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.30:
        out = _partial_crop(out, rng, max_pct=0.06)
    return out


def apply_realistic_scenarios(card: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    scene, params = place_card_on_background(card, rng)
    return apply_camera_effects(scene, params, rng)
