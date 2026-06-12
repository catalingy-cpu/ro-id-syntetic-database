"""Încărcare multi-template: JSON calibrat (classic_reference) + YAML specimen."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from ro_id_synth.roi import TemplateConfig, load_templates
from ro_id_synth.template_config import FieldRect, TemplateSpec, load_template_spec


def _load_bgr(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Template negăsit: {path}")
    return img


def _orient_image(img: np.ndarray, auto_rotate: bool) -> np.ndarray:
    if auto_rotate and img.shape[0] > img.shape[1]:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def template_config_to_spec(cfg: TemplateConfig) -> TemplateSpec:
    img = _orient_image(_load_bgr(cfg.path), cfg.auto_rotate)
    h, w = img.shape[:2]
    fields: dict[str, FieldRect] = {}
    for name, roi in cfg.fields.items():
        x1, y1, x2, y2 = roi.to_pixels(w, h)
        fields[name] = FieldRect(
            key=name,
            x=x1,
            y=y1,
            width=max(1, x2 - x1),
            height=max(1, y2 - y1),
            ink=roi.ink,
            align=roi.align,
        )
    return TemplateSpec(
        key=cfg.key,
        image_path=cfg.path,
        width=w,
        height=h,
        kind=cfg.kind,
        fields=fields,
        auto_rotate=cfg.auto_rotate,
    )


def load_yaml_specs(base_dir: Path, keys: list[str] | None = None) -> dict[str, TemplateSpec]:
    yaml_path = base_dir / "config" / "templates.yaml"
    all_cfgs = {cfg.key: cfg for cfg in load_templates(yaml_path, base_dir)}
    wanted = keys if keys is not None else list(all_cfgs.keys())
    out: dict[str, TemplateSpec] = {}
    for key in wanted:
        if key not in all_cfgs:
            raise KeyError(f"Template YAML lipsă: {key}")
        out[key] = template_config_to_spec(all_cfgs[key])
    return out


def load_generation_mix(mix_path: Path, base_dir: Path) -> tuple[dict[str, TemplateSpec], list[tuple[str, float]]]:
    raw = json.loads(mix_path.read_text(encoding="utf-8"))
    entries: dict[str, dict] = raw["templates"]
    weights_raw: dict[str, float] = raw["weights"]

    yaml_cache: dict[str, TemplateSpec] = {}
    specs: dict[str, TemplateSpec] = {}

    for key, entry in entries.items():
        if entry["type"] == "json":
            rel = Path(entry["path"])
            spec = load_template_spec((base_dir / rel).resolve(), base_dir)
            specs[key] = replace(spec, key=key)
        elif entry["type"] == "yaml":
            yaml_key = entry.get("key", key)
            if yaml_key not in yaml_cache:
                yaml_cache.update(load_yaml_specs(base_dir, [yaml_key]))
            specs[key] = yaml_cache[yaml_key]
        else:
            raise ValueError(f"Tip template necunoscut: {entry['type']}")

    weights: list[tuple[str, float]] = []
    for key, w in weights_raw.items():
        if key not in specs:
            raise KeyError(f"Weight pentru template necunoscut: {key}")
        if w > 0:
            weights.append((key, float(w)))
    if not weights:
        raise ValueError("generation_mix.json: cel puțin un weight > 0")

    missing = [str(s.image_path) for s in specs.values() if not s.image_path.is_file()]
    if missing:
        raise FileNotFoundError("Template-uri lipsă:\n" + "\n".join(missing))

    return specs, weights
