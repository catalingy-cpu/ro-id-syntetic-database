from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FieldRoi:
    x: float
    y: float
    w: float
    h: float
    font_ratio: float = 0.68
    align: str = "left"
    ink: str = "auto"

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        x1 = int(self.x * width)
        y1 = int(self.y * height)
        x2 = int(min(width, (self.x + self.w) * width))
        y2 = int(min(height, (self.y + self.h) * height))
        return max(0, x1), max(0, y1), max(2, x2), max(2, y2)


@dataclass
class TemplateConfig:
    key: str
    path: Path
    auto_rotate: bool
    kind: str
    fields: dict[str, FieldRoi]


def _parse_field(name: str, spec: dict[str, Any]) -> FieldRoi:
    return FieldRoi(
        x=float(spec["x"]),
        y=float(spec["y"]),
        w=float(spec["w"]),
        h=float(spec["h"]),
        font_ratio=float(spec.get("font_ratio", 0.68)),
        align=str(spec.get("align", "left")),
        ink=str(spec.get("ink", "auto")),
    )


def load_templates(config_path: Path, base_dir: Path) -> list[TemplateConfig]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out: list[TemplateConfig] = []
    for key, item in raw["templates"].items():
        rel = Path(item["path"])
        path = (base_dir / rel).resolve()
        if not path.is_file():
            path = (config_path.parent / rel).resolve()
        fields = {name: _parse_field(name, spec) for name, spec in item["fields"].items()}
        out.append(
            TemplateConfig(
                key=key,
                path=path,
                auto_rotate=bool(item.get("auto_rotate", False)),
                kind=str(item.get("kind", "classic")),
                fields=fields,
            )
        )
    return out


def load_template_image(cfg: TemplateConfig):
    import cv2
    import numpy as np

    img = cv2.imdecode(np.fromfile(str(cfg.path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Template negăsit: {cfg.path}")
    h, w = img.shape[:2]
    if cfg.auto_rotate and h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img
