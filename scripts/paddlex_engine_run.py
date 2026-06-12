#!/usr/bin/env python3
"""Wrapper pentru `paddlex.engine.Engine` (apelat din train_and_deploy cu -c / -o)."""
from __future__ import annotations

import os

# Trebuie setat înainte de import paddle/paddlex (propagat și la tools/train.py).
for _key, _val in {
    "FLAGS_use_mkldnn": "0",
    "FLAGS_enable_mkldnn": "0",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "MKL_DEBUG_CPU_TYPE": "5",
    "KMP_DUPLICATE_LIB_OK": "TRUE",
}.items():
    os.environ.setdefault(_key, _val)

import shutil
from pathlib import Path
import warnings

import paddlex
from paddlex.engine import Engine

if __name__ == "__main__":
    # Unele instalații PaddleX nu au directoarele de config pentru repo_apis,
    # dar registrele așteaptă fișierele în repo_apis/PaddleOCR_api/configs/.
    # Copiem din paddlex/configs/modules/text_recognition/ în locația așteptată.
    try:
        paddlex_root = Path(paddlex.__file__).resolve().parent
        src = paddlex_root / "configs" / "modules" / "text_recognition"
        dst = paddlex_root / "repo_apis" / "PaddleOCR_api" / "configs"
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.glob("*.yaml"):
                shutil.copy2(str(f), str(dst / f.name))
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Nu pot pregăti config-urile PaddleOCR_api: {exc}", RuntimeWarning, stacklevel=1)

    # Asigură popularea registry-ului de modele pentru train/eval/export text recognition.
    # În unele instalații PaddleX, registry-ul rămâne gol până când aceste module sunt importate.
    try:
        import paddlex.repo_apis.PaddleOCR_api.text_rec  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Nu pot preîncărca PaddleOCR_api.text_rec: {exc}. Continui cu Engine().",
            RuntimeWarning,
            stacklevel=1,
        )
    Engine().run()
