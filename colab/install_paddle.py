#!/usr/bin/env python3
"""Instalare paddlepaddle-gpu pe Colab — detectează CUDA și încearcă mai multe surse."""

from __future__ import annotations

import re
import subprocess
import sys

# Ordinea contează: Colab (2025+) folosește de obicei CUDA 12.x recent
_WHEEL_INDEX = [
    ("12.9", "https://www.paddlepaddle.org.cn/packages/stable/cu129/"),
    ("12.6", "https://www.paddlepaddle.org.cn/packages/stable/cu126/"),
    ("12.3", "https://www.paddlepaddle.org.cn/packages/stable/cu123/"),
    ("11.8", "https://www.paddlepaddle.org.cn/packages/stable/cu118/"),
]

_VERSIONS = ("3.3.0", "3.2.2", "3.2.1", "3.2.0")


def _cuda_version() -> str | None:
    r = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    m = re.search(r"CUDA Version:\s*(\d+\.\d+)", r.stdout or "")
    return m.group(1) if m else None


def _ordered_indices(cuda: str | None) -> list[str]:
    if not cuda:
        return [url for _, url in _WHEEL_INDEX]
    major_minor = cuda
    matched: list[str] = []
    for ver, url in _WHEEL_INDEX:
        if major_minor.startswith(ver[:3]) or ver.startswith(major_minor[:3]):
            matched.append(url)
    for _, url in _WHEEL_INDEX:
        if url not in matched:
            matched.append(url)
    return matched


def install_paddle(python: str | None = None) -> str:
    py = python or sys.executable
    cuda = _cuda_version()
    print(f"CUDA (driver): {cuda or 'necunoscut'}")
    if cuda is None:
        print("Atenție: rulează Runtime → Change runtime type → GPU")

    last_err = ""
    for index_url in _ordered_indices(cuda):
        for version in _VERSIONS:
            cmd = [
                py,
                "-m",
                "pip",
                "install",
                "-U",
                f"paddlepaddle-gpu=={version}",
                "-i",
                index_url,
            ]
            print(">", " ".join(cmd))
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                print(f"Instalat paddlepaddle-gpu=={version} ({index_url})")
                return version
            last_err = (r.stderr or r.stdout)[-800:]

    raise SystemExit(
        "paddlepaddle-gpu nu s-a putut instala.\n"
        "Verifică Runtime → GPU, apoi re-rulează celula.\n"
        f"Ultima eroare:\n{last_err}"
    )


if __name__ == "__main__":
    install_paddle()
