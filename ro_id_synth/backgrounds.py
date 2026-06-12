"""Fundaluri și distribuție scară card — delegă texturi la textures.py."""

from __future__ import annotations

import numpy as np

from ro_id_synth.textures import sample_realworld_background

__all__ = ["pick_card_scale_tier", "sample_realworld_background"]


def pick_card_scale_tier(rng: np.random.Generator) -> tuple[str, float]:
    """70% large, 20% medium, 10% small."""
    roll = float(rng.random())
    if roll < 0.70:
        return "large", float(rng.uniform(0.72, 0.94))
    if roll < 0.90:
        return "medium", float(rng.uniform(0.46, 0.66))
    return "small", float(rng.uniform(0.28, 0.42))
