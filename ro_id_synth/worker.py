"""Worker — generator template-based cu filtru calitate."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ro_id_synth.pipeline import render_card_from_template, render_final_sample
from ro_id_synth.quality_filter import MAX_RETRIES, validate_sample
from ro_id_synth.records import generate_record
from ro_id_synth.template_config import TemplateSpec, load_template_spec
from ro_id_synth.template_registry import load_generation_mix

_TEMPLATE_SPEC: TemplateSpec | None = None
_TEMPLATE_SPECS: dict[str, TemplateSpec] = {}
_TEMPLATE_WEIGHTS: list[tuple[str, float]] = []


def init_pool(config_path: str, base_dir: str) -> None:
    global _TEMPLATE_SPEC, _TEMPLATE_SPECS, _TEMPLATE_WEIGHTS
    base = Path(base_dir)
    cfg = Path(config_path)
    _TEMPLATE_SPEC = load_template_spec(cfg, base)
    _TEMPLATE_SPECS = {}
    _TEMPLATE_WEIGHTS = []


def init_pool_multi(mix_path: str, base_dir: str) -> None:
    global _TEMPLATE_SPEC, _TEMPLATE_SPECS, _TEMPLATE_WEIGHTS
    base = Path(base_dir)
    specs, weights = load_generation_mix(Path(mix_path), base)
    _TEMPLATE_SPEC = None
    _TEMPLATE_SPECS = specs
    _TEMPLATE_WEIGHTS = weights


def _spec() -> TemplateSpec:
    if _TEMPLATE_SPEC is None:
        raise RuntimeError("init_pool() nu a fost apelat")
    return _TEMPLATE_SPEC


def _pick_spec(rng: np.random.Generator) -> TemplateSpec:
    if not _TEMPLATE_SPECS:
        return _spec()
    keys = [k for k, _ in _TEMPLATE_WEIGHTS]
    probs = np.array([w for _, w in _TEMPLATE_WEIGHTS], dtype=np.float64)
    probs /= probs.sum()
    key = keys[int(rng.choice(len(keys), p=probs))]
    return _TEMPLATE_SPECS[key]


def generate_one_image(
    global_idx: int,
    base_seed: int,
    rng: np.random.Generator | None = None,
    *,
    skip_quality: bool = False,
) -> tuple[np.ndarray, str, np.ndarray, str]:
    if rng is None:
        rng = np.random.default_rng(base_seed + global_idx)
    spec = _pick_spec(rng)

    for attempt in range(MAX_RETRIES if not skip_quality else 1):
        record = generate_record(seed=base_seed + global_idx + attempt * 9973)
        final, card, transcription = render_final_sample(spec, record, rng)

        if skip_quality:
            payload = json.dumps({"transcription": transcription}, ensure_ascii=False)
            return final, payload, card, spec.key

        passed, _score, _reason = validate_sample(final, card, record, _legacy_cfg_from_spec(spec))
        if passed:
            payload = json.dumps({"transcription": transcription}, ensure_ascii=False)
            return final, payload, card, spec.key

    payload = json.dumps({"transcription": transcription}, ensure_ascii=False)
    return final, payload, card, spec.key


def _legacy_cfg_from_spec(spec: TemplateSpec):
    """Adaptor pentru quality_filter care așteaptă TemplateConfig."""
    from ro_id_synth.roi import FieldRoi, TemplateConfig

    fields = {
        k: FieldRoi(
            x=f.x / spec.width,
            y=f.y / spec.height,
            w=f.width / spec.width,
            h=f.height / spec.height,
            ink=f.ink if f.ink in ("red", "auto", "dark") else "auto",
            align=f.align,
        )
        for k, f in spec.fields.items()
    }
    return TemplateConfig(
        key=spec.key,
        path=spec.image_path,
        auto_rotate=spec.auto_rotate,
        kind=spec.kind,
        fields=fields,
    )


def _process_batch(args: tuple) -> dict[str, int]:
    (
        batch_index,
        batch_size,
        start_index,
        images_dir,
        labels_dir,
        base_seed,
        debug_dir,
        _config_path,
        _base_dir,
    ) = args

    rng = np.random.default_rng(base_seed + batch_index)
    labels_path = Path(labels_dir) / f"part_{batch_index:05d}.txt"
    images_path = Path(images_dir)
    debug_path = Path(debug_dir) if debug_dir else None

    lines: list[str] = []
    for i in range(batch_size):
        global_idx = start_index + i
        img, payload, _card, _tpl_key = generate_one_image(global_idx, base_seed, rng)

        fname = f"{global_idx:07d}.jpg"
        cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tofile(str(images_path / fname))
        lines.append(f"images/{fname}\t{payload}")

    labels_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if debug_path:
        end_idx = start_index + batch_size
        from ro_id_synth.debug_grid import build_debug_grid_at

        for milestone in range(200, end_idx + 1, 200):
            if start_index < milestone <= end_idx:
                grid_idx = milestone // 200
                build_debug_grid_at(
                    _config_path,
                    _base_dir,
                    debug_path / f"grid_{grid_idx:04d}.jpg",
                    count=100,
                    seed=base_seed + grid_idx,
                )

    return {"kept": batch_size, "batch_index": batch_index}
