#!/usr/bin/env python3
"""Instalare paddlepaddle-gpu + PaddleOCR stack pe Colab (fără a înlocui GPU cu CPU)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Colab 2025+: driver CUDA 13.x — încearcă cu130, apoi cu126
_WHEEL_INDEX = [
    ("13.0", "https://www.paddlepaddle.org.cn/packages/stable/cu130/"),
    ("12.9", "https://www.paddlepaddle.org.cn/packages/stable/cu129/"),
    ("12.6", "https://www.paddlepaddle.org.cn/packages/stable/cu126/"),
    ("12.3", "https://www.paddlepaddle.org.cn/packages/stable/cu123/"),
    ("11.8", "https://www.paddlepaddle.org.cn/packages/stable/cu118/"),
]

_VERSIONS = ("3.3.0", "3.2.2", "3.2.0")


def _cuda_version() -> str | None:
    r = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    m = re.search(r"CUDA Version:\s*(\d+\.\d+)", r.stdout or "")
    return m.group(1) if m else None


def _ordered_indices(cuda: str | None) -> list[str]:
    if not cuda:
        return [url for _, url in _WHEEL_INDEX]
    matched: list[str] = []
    for ver, url in _WHEEL_INDEX:
        if cuda.startswith(ver) or ver.startswith(cuda.split(".")[0]):
            matched.append(url)
    for _, url in _WHEEL_INDEX:
        if url not in matched:
            matched.append(url)
    return matched


def _pip(py: str, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = [py, "-m", "pip", *args]
    print(">", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def install_paddle_gpu(py: str) -> tuple[str, str]:
    cuda = _cuda_version()
    print(f"CUDA (driver): {cuda or 'necunoscut'}")
    if cuda is None:
        print("Atenție: Runtime → Change runtime type → GPU")

    last_err = ""
    for index_url in _ordered_indices(cuda):
        for version in _VERSIONS:
            r = _pip(py, "install", "-U", f"paddlepaddle-gpu=={version}", "-i", index_url, check=False)
            if r.returncode == 0:
                print(f"OK paddlepaddle-gpu=={version} ({index_url})")
                return version, index_url
            last_err = (r.stderr or r.stdout)[-800:]

    raise SystemExit(f"paddlepaddle-gpu eșuat.\n{last_err}")


def install_training_stack(py: str, repo_root: Path | None = None) -> None:
    """GPU paddle → paddlex/paddleocr (--no-deps) → deps train → re-pin GPU."""
    version, index_url = install_paddle_gpu(py)

    # Fără --no-deps, pip instalează paddlepaddle (CPU) și strică GPU-ul.
    _pip(py, "install", "paddlex", "paddleocr>=3.0.0,<4.0.0", "--no-deps")

    train_req = (repo_root or Path(".")) / "requirements-train.txt"
    if train_req.is_file():
        _pip(py, "install", "-r", str(train_req))

    _pip(py, "install", "-U", f"paddlepaddle-gpu=={version}", "-i", index_url)

    verify = subprocess.run(
        [py, "-c", "import paddle, paddlex; print('Paddle', paddle.__version__, 'CUDA', paddle.is_compiled_with_cuda())"],
        capture_output=True,
        text=True,
    )
    if verify.returncode != 0:
        print(verify.stderr or verify.stdout)
        raise SystemExit("import paddle/paddlex eșuat după instalare")
    print(verify.stdout.strip())


def install_paddle(python: str | None = None) -> str:
    py = python or sys.executable
    version, _ = install_paddle_gpu(py)
    return version


if __name__ == "__main__":
    install_training_stack(sys.executable)
