#!/usr/bin/env python3
"""
Împachetează modelul antrenat pentru descărcare (Colab sau local).

Implicit: exports/frc_ci_rec/ (repo standalone) sau PADDLE_OCR_ROOT/models/frc_ci_rec/.
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_model_dir(repo: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    ocr_root = os.getenv("PADDLE_OCR_ROOT", "").strip()
    if ocr_root:
        return Path(ocr_root).resolve() / "models" / "frc_ci_rec"
    sibling = repo.parent / "paddle-ocr" / "models" / "frc_ci_rec"
    if sibling.is_dir():
        return sibling.resolve()
    return (repo / "exports" / "frc_ci_rec").resolve()


def package_model_export(
    *,
    model_dir: Path,
    zip_path: Path,
    archive_root: str = "frc_ci_rec",
) -> Path:
    inference = model_dir / "inference"
    if not inference.is_dir():
        raise SystemExit(
            f"Lipsește folderul inference: {inference}\n"
            "Rulează mai întâi scripts/train_and_deploy.py."
        )
    has_weights = any(inference.glob("*.pdiparams")) or (inference / "inference.pdiparams").is_file()
    has_yml = (inference / "inference.yml").is_file()
    if not has_weights and not has_yml:
        raise SystemExit(f"Inference incomplet în {inference}")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.is_file():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(model_dir.rglob("*")):
            if not file_path.is_file():
                continue
            arcname = Path(archive_root) / file_path.relative_to(model_dir)
            zf.write(file_path, arcname.as_posix())

    return zip_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Creează model_export.zip din modelul frc_ci_rec")
    p.add_argument("--repo-root", type=Path, default=repo_root())
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    model_dir = resolve_model_dir(repo, args.model_dir)
    zip_path = (args.output or (repo / "colab" / "model_export.zip")).resolve()
    out = package_model_export(model_dir=model_dir, zip_path=zip_path)
    print(f"Creat: {out}")


if __name__ == "__main__":
    main()
