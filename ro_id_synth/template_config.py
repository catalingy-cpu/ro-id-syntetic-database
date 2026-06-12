"""Phase 1 — încărcare template + câmpuri editabile din JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class FieldRect:
    key: str
    x: int
    y: int
    width: int
    height: int
    ink: str = "dark"
    align: str = "left"

    def to_pixels(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.width, self.y + self.height


@dataclass
class TemplateSpec:
    key: str
    image_path: Path
    width: int
    height: int
    kind: str
    fields: dict[str, FieldRect]

    def load_image(self) -> np.ndarray:
        img = cv2.imdecode(np.fromfile(str(self.image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Template negăsit: {self.image_path}")
        return img


def _parse_field(key: str, spec: dict[str, Any]) -> FieldRect:
    return FieldRect(
        key=key,
        x=int(spec["x"]),
        y=int(spec["y"]),
        width=max(1, int(spec["width"])),
        height=max(1, int(spec["height"])),
        ink=str(spec.get("ink", "dark")),
        align=str(spec.get("align", "left")),
    )


def load_template_spec(config_path: Path, base_dir: Path | None = None) -> TemplateSpec:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    base = base_dir or config_path.parent.parent
    rel = Path(raw["template_image"])
    img_path = (base / rel).resolve()
    if not img_path.is_file():
        img_path = (config_path.parent / rel).resolve()

    fields = {k: _parse_field(k, v) for k, v in raw["fields"].items()}
    return TemplateSpec(
        key=config_path.stem,
        image_path=img_path,
        width=int(raw.get("width", 0)),
        height=int(raw.get("height", 0)),
        kind=str(raw.get("kind", "classic")),
        fields=fields,
    )


def save_fields_overlay(
    spec: TemplateSpec,
    output_path: Path,
    *,
    color: tuple[int, int, int] = (0, 200, 80),
) -> Path:
    """Desenează ROI-uri pe template — util la calibrare."""
    img = spec.load_image().copy()
    for field in spec.fields.values():
        x1, y1, x2, y2 = field.to_pixels()
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            field.key,
            (x1, max(y1 - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tofile(str(output_path))
    return output_path
