"""Pipeline complet: template → înlocuire → randare → scenă foto."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ro_id_synth.field_replace import replace_all_fields
from ro_id_synth.records import SyntheticIdRecord, generate_record
from ro_id_synth.scenarios import apply_camera_effects, place_card_on_background
from ro_id_synth.template_config import TemplateSpec
from ro_id_synth.text_render import render_text_on_card


def render_card_from_template(
    spec: TemplateSpec,
    record: SyntheticIdRecord,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returnează (original, după_inpaint, card_cu_text).
    """
    original = spec.load_image()
    cleaned = replace_all_fields(original, spec)
    card = render_text_on_card(cleaned, original, spec, record, rng)
    return original, cleaned, card


def render_final_sample(
    spec: TemplateSpec,
    record: SyntheticIdRecord,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Card editat → fundal → efecte cameră. Returnează (imagine_finală, card, transcription)."""
    _orig, _cleaned, card = render_card_from_template(spec, record, rng)
    scene, params = place_card_on_background(card, rng)
    final = apply_camera_effects(scene, params, rng)
    return final, card, record.to_transcription()


def write_debug_triplet(
    spec: TemplateSpec,
    record: SyntheticIdRecord,
    rng: np.random.Generator,
    debug_dir: Path,
) -> None:
    """Phase 6 — original / replaced / final."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    original, cleaned, card = render_card_from_template(spec, record, rng)
    scene, params = place_card_on_background(card, rng)
    final = apply_camera_effects(scene, params, rng)

    for name, img in (
        ("original.jpg", original),
        ("replaced_fields.jpg", cleaned),
        ("final_sample.jpg", final),
    ):
        path = debug_dir / name
        cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 94])[1].tofile(str(path))


def generate_record_for_template(seed: int) -> SyntheticIdRecord:
    return generate_record(seed=seed)
