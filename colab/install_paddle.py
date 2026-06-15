#!/usr/bin/env python3
"""Instalare paddlepaddle-gpu + PaddleOCR stack pe Colab."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_WHEEL_INDEX = [
    ("13.0", "https://www.paddlepaddle.org.cn/packages/stable/cu130/"),
    ("12.9", "https://www.paddlepaddle.org.cn/packages/stable/cu129/"),
    ("12.6", "https://www.paddlepaddle.org.cn/packages/stable/cu126/"),
    ("11.8", "https://www.paddlepaddle.org.cn/packages/stable/cu118/"),
]
_VERSIONS = ("3.3.0", "3.2.2", "3.2.0")


def _cuda_version() -> str | None:
    r = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    m = re.search(r"CUDA Version:\s*(\d+\.\d+)", r.stdout or "")
    return m.group(1) if m else None


def _ordered_indices(cuda: str | None) -> list[str]:
    if not cuda:
        return [u for _, u in _WHEEL_INDEX]
    matched = [u for v, u in _WHEEL_INDEX if cuda.startswith(v.split(".")[0]) or v.startswith(cuda)]
    for _, u in _WHEEL_INDEX:
        if u not in matched:
            matched.append(u)
    return matched


def _pip(py: str, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = [py, "-m", "pip", *args]
    print(">", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def _install_paddle_gpu(py: str) -> tuple[str, str]:
    cuda = _cuda_version()
    print(f"CUDA (driver): {cuda or 'necunoscut'}")
    last = ""
    for index_url in _ordered_indices(cuda):
        for version in _VERSIONS:
            r = _pip(py, "install", "-U", f"paddlepaddle-gpu=={version}", "-i", index_url, check=False)
            if r.returncode == 0:
                print(f"OK paddlepaddle-gpu=={version}")
                return version, index_url
            last = (r.stderr or r.stdout)[-600:]
    raise SystemExit(f"paddlepaddle-gpu eșuat.\n{last}")


def _pin_paddle_gpu(py: str, version: str, index_url: str) -> None:
    _pip(py, "install", "-U", "--force-reinstall", f"paddlepaddle-gpu=={version}", "-i", index_url)


def _check_import(py: str, code: str, label: str) -> None:
    r = subprocess.run([py, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"--- {label} ---")
        print(r.stderr or r.stdout)
        raise SystemExit(f"{label} eșuat")
    print((r.stdout or "").strip())


def install_training_stack(py: str, repo_root: Path | None = None) -> None:
    """
  1. paddlepaddle-gpu
  2. paddlex (cu dependențe — altfel importul pică)
  3. paddleocr (--no-deps)
  4. requirements-train.txt
  5. re-pin GPU (paddlex poate aduce paddle CPU)
    """
    version, index_url = _install_paddle_gpu(py)

    _pip(py, "install", "-U", "paddlex")
    _pin_paddle_gpu(py, version, index_url)

    _pip(py, "install", "paddleocr>=3.0.0,<4.0.0", "--no-deps")
    _pin_paddle_gpu(py, version, index_url)

    train_req = (repo_root or Path(".")) / "requirements-train.txt"
    if train_req.is_file():
        _pip(py, "install", "-r", str(train_req))
    _pin_paddle_gpu(py, version, index_url)

    _check_import(py, "import paddle; print('Paddle', paddle.__version__, 'CUDA', paddle.is_compiled_with_cuda())", "paddle")
    _check_import(py, "import paddlex; print('PaddleX OK')", "paddlex")


def install_paddle(python: str | None = None) -> str:
    version, _ = _install_paddle_gpu(python or sys.executable)
    return version


if __name__ == "__main__":
    install_training_stack(sys.executable)
