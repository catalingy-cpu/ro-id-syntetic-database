"""Rezolvare fonturi — Windows (Arial) și Linux/Colab (Liberation/DejaVu)."""

from __future__ import annotations

import os
import platform
from pathlib import Path

from PIL import ImageFont

_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _linux_font_paths() -> list[str]:
    paths: list[str] = []
    roots = (
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        str(Path.home() / ".fonts"),
        str(Path.home() / ".local/share/fonts"),
    )
    names = (
        "LiberationSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
        "arialbd.ttf",
        "Arial Bold.ttf",
        "calibrib.ttf",
        "calibri.ttf",
    )
    for root in roots:
        base = Path(root)
        if not base.is_dir():
            continue
        for name in names:
            for hit in base.rglob(name):
                paths.append(str(hit))
    return paths


def font_candidates(kind: str) -> list[str]:
    win = os.environ.get("WINDIR", r"C:\Windows")
    fonts_dir = os.path.join(win, "Fonts")
    if kind == "classic":
        names = ["arialbd.ttf", "Arialbd.ttf", "calibrib.ttf", "consolab.ttf"]
    else:
        names = ["calibrib.ttf", "arialbd.ttf", "arialn.ttf", "calibri.ttf"]
    if platform.system() == "Windows":
        return [os.path.join(fonts_dir, n) for n in names] + names
    return _linux_font_paths() + names + ["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"]


def resolve_font(size: int, kind: str = "classic") -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    key = (kind, size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    for path in font_candidates(kind):
        try:
            font = ImageFont.truetype(path, size=size)
            _FONT_CACHE[key] = font
            return font
        except OSError:
            continue
    font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font
