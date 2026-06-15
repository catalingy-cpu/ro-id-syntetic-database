#!/usr/bin/env python3
"""
Antrenare PaddleX (rec latin PP-OCRv5) + deploy automat în services/paddle-ocr.

Folosește mediul Python din paddle-ocr (.venv) — același stack ca inferența.

Exemplu (noaptea, după generate.py):
  python scripts/train_and_deploy.py --dataset dataset --device gpu:0

Doar deploy (model deja antrenat):
  python scripts/train_and_deploy.py --skip-train --export-dir training_output/best_accuracy
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Antrenare + deploy model CI în paddle-ocr")
    p.add_argument("--dataset", type=str, default="dataset", help="Folder cu labels/train.txt și images/")
    p.add_argument("--output", type=str, default="training_output", help="Output antrenare PaddleX")
    p.add_argument("--device", type=str, default=None, help="cpu | gpu:0 — implicit din PADDLE_DEVICE")
    p.add_argument("--epochs", type=int, default=None, help="Suprascrie Train.epochs_iters")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--skip-train", action="store_true", help="Sari antrenarea, doar export+deploy")
    p.add_argument("--skip-export", action="store_true", help="Deploy din inference existent")
    p.add_argument("--export-dir", type=str, default=None, help="ex. training_output/best_accuracy")
    p.add_argument(
        "--paddle-python",
        type=str,
        default=None,
        help="Python din services/paddle-ocr/.venv (autodetect)",
    )
    p.add_argument(
        "--model-name",
        type=str,
        default="frc_ci_rec",
        help="Subfolder în paddle-ocr/models/",
    )
    p.add_argument(
        "--install-paddleocr-repo",
        action="store_true",
        help="Forțează `paddlex --install PaddleOCR` (git clone; de obicei nu e necesar)",
    )
    p.add_argument(
        "--skip-check-dataset",
        action="store_true",
        help="Sari verificarea PaddleX check_dataset (nu necesită matplotlib)",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Config rapid: fara RecConAug/RecAug, batch mai mare, fara ext_data extra (recomandat CPU)",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (implicit: 0 pe Windows/fast, 4 pe GPU)",
    )
    p.add_argument(
        "--paddle-ocr-root",
        type=str,
        default=None,
        help="Rădăcină paddle-ocr pentru venv/deploy (sau env PADDLE_OCR_ROOT)",
    )
    p.add_argument(
        "--skip-deploy",
        action="store_true",
        help="Nu scrie în paddle-ocr; export în exports/<model-name>/ (repo standalone)",
    )
    return p.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def frchub_paddle_ocr_sibling() -> Path:
    return repo_root().parent / "paddle-ocr"


def is_frchub_monorepo_layout() -> bool:
    return (frchub_paddle_ocr_sibling() / "requirements.txt").is_file()


def paddle_ocr_root(cli_root: str | None = None) -> Path:
    if cli_root:
        return Path(cli_root).resolve()
    env = os.getenv("PADDLE_OCR_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    if is_frchub_monorepo_layout():
        return frchub_paddle_ocr_sibling().resolve()
    return repo_root().resolve()


def venv_python_in(base: Path) -> Path | None:
    win = base / ".venv" / "Scripts" / "python.exe"
    if win.is_file():
        return win
    unix = base / ".venv" / "bin" / "python"
    if unix.is_file():
        return unix
    return None


def default_paddle_python(cli_root: str | None = None) -> Path:
    bases: list[Path] = []
    if is_frchub_monorepo_layout():
        bases.append(frchub_paddle_ocr_sibling())
    bases.append(paddle_ocr_root(cli_root))
    bases.append(repo_root())
    seen: set[Path] = set()
    for base in bases:
        resolved = base.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        found = venv_python_in(resolved)
        if found is not None:
            return found
    base = paddle_ocr_root(cli_root) / ".venv"
    return base / "Scripts" / "python.exe" if sys.platform == "win32" else base / "bin" / "python"


def run_cmd(cmd: list[str], env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    print(">", " ".join(cmd))
    merged = os.environ.copy()
    merged.update(paddle_runtime_env())
    if env:
        merged.update(env)
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None, env=merged)


def paddle_runtime_env() -> dict[str, str]:
    """Stabilitate Paddle CPU pe Windows (MKL / AMD / subprocess train.py)."""
    return {
        "FLAGS_use_mkldnn": "0",
        "FLAGS_enable_mkldnn": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        # MKL pe procesoare AMD (ex. Vega 8) — evită „Intel MKL ERROR: vsAdd”.
        "MKL_DEBUG_CPU_TYPE": "5",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
    }


def run_paddlex_engine(
    paddle_py: Path,
    config_path: Path,
    overrides: list[str],
    cwd: Path,
    env: dict[str, str] | None = None,
) -> None:
    runner = repo_root() / "scripts" / "paddlex_engine_run.py"
    cmd = [str(paddle_py), str(runner), "-c", str(config_path)]
    for o in overrides:
        cmd.extend(["-o", o])
    run_cmd(cmd, cwd=cwd, env=env)


def paddlex_package_root(paddle_py: Path) -> Path:
    proc = subprocess.run(
        [
            str(paddle_py),
            "-c",
            "import paddlex; from pathlib import Path; print(Path(paddlex.__file__).resolve().parent)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(proc.stdout.strip())


def paddlex_repos_dir(paddle_py: Path) -> Path:
    return paddlex_package_root(paddle_py) / "repo_manager" / "repos"


def bundled_paddleocr_api(paddle_py: Path) -> Path:
    return paddlex_package_root(paddle_py) / "repo_apis" / "PaddleOCR_api"


def paddleocr_repo_dir(paddle_py: Path) -> Path:
    return paddlex_repos_dir(paddle_py) / "PaddleOCR"


def paddleocr_train_config(paddle_py: Path, *, fast: bool = False) -> Path:
    """Config YAML PaddleOCR pentru antrenare rec."""
    if fast:
        fast_cfg = repo_root() / "config" / "ocr_train_ci_fast.yml"
        if fast_cfg.is_file():
            return fast_cfg.resolve()
    repo = paddleocr_repo_dir(paddle_py)
    candidates = (
        repo / "configs/rec/PP-OCRv5/multi_language/latin_PP-OCRv5_mobile_rec.yml",
        repo / "configs/rec/PP-OCRv5/latin_PP-OCRv5_mobile_rec.yml",
        repo / "configs/rec/latin_PP-OCRv5_mobile_rec.yml",
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise SystemExit(
        "Nu găsesc config antrenare PaddleOCR (latin_PP-OCRv5_mobile_rec.yml).\n"
        f"Verifică repo: {repo}"
    )


def is_gpu_device(device: str) -> bool:
    return "gpu" in device.lower()


def default_batch_size(device: str, *, fast: bool) -> int:
    if fast:
        return 128 if is_gpu_device(device) else 8
    return 64 if is_gpu_device(device) else 8


def default_num_workers(device: str, *, fast: bool) -> int:
    if fast or sys.platform == "win32":
        return 0 if not is_gpu_device(device) else 2
    return 4 if is_gpu_device(device) else 0


def apply_fast_train_patches(ocr_repo: Path) -> None:
    """Evita get_ext_data() (2 imagini extra/sample) — mare bottleneck pe CPU."""
    path = ocr_repo / "ppocr" / "data" / "simple_dataset.py"
    if not path.is_file():
        return
    marker = "# frchub-fast-skip-ext-data"
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    old = '            data["ext_data"] = self.get_ext_data()'
    new = f'            data["ext_data"] = []  {marker}'
    if old not in text:
        print(f"Atentie: nu pot aplica patch fast in {path}")
        return
    path.write_text(text.replace(old, new), encoding="utf-8")
    print(f"Patch fast train aplicat: {path.name}")


def materialize_ocr_config(
    src: Path,
    dest: Path,
    *,
    num_workers: int,
    batch_size: int | None = None,
) -> Path:
    """Copie config OCR cu num_workers potrivit platformei."""
    text = src.read_text(encoding="utf-8")
    text = re.sub(r"num_workers:\s*\d+", f"num_workers: {num_workers}", text)
    text = re.sub(r"use_visualdl:\s*true", "use_visualdl: false", text, flags=re.I)
    if batch_size is not None:
        text = re.sub(r"first_bs:\s*&bs\s*\d+", f"first_bs: &bs {batch_size}", text)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return dest


def count_train_lines(paddlex_data: Path) -> int:
    train_txt = paddlex_data / "train.txt"
    if not train_txt.is_file():
        return 0
    return sum(
        1
        for line in train_txt.read_text(encoding="utf-8").splitlines()
        if line.strip() and "\t" in line
    )


def print_train_plan(
    paddlex_data: Path,
    device: str,
    epochs: int,
    batch_size: int,
    *,
    fast: bool,
) -> None:
    n = count_train_lines(paddlex_data)
    iters = max(1, (n + batch_size - 1) // batch_size) if n else 0
    mode = "FAST" if fast else "standard"
    print(f"Plan antrenare [{mode}]: {n} sample train, batch={batch_size}, ~{iters} iter/epoca, {epochs} epoci")
    if not is_gpu_device(device):
        print(
            "Atentie: rulezi pe CPU. Foloseste -Device gpu:0 daca ai NVIDIA; "
            "pe CPU foloseste -Fast -Epochs 3 pentru test."
        )
    elif fast:
        print("Fast: fara augmentari grele (RecConAug/RecAug), batch marit.")


def _paddlex_cli(paddle_py: Path) -> list[str]:
    win = paddle_ocr_root() / ".venv" / "Scripts" / "paddlex.exe"
    if win.is_file():
        return [str(win)]
    unix = paddle_ocr_root() / ".venv" / "bin" / "paddlex"
    if unix.is_file():
        return [str(unix)]
    which = shutil.which("paddlex")
    if which:
        return [which]
    return [str(paddle_py), "-m", "paddlex"]


def _clone_paddleocr_repo(repo_dir: Path) -> None:
    if (repo_dir / "tools" / "train.py").is_file():
        return
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Clone PaddleOCR → {repo_dir}")
    run_cmd(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/PaddlePaddle/PaddleOCR.git",
            str(repo_dir),
        ],
    )


def ensure_paddlex_runtime(paddle_py: Path, cwd: Path, *, force_repo_install: bool = False) -> Path:
    """Pregătește runtime PaddleX pentru antrenare rec și întoarce calea repo PaddleOCR."""
    repos = paddlex_repos_dir(paddle_py)
    repos.mkdir(parents=True, exist_ok=True)
    repo_dir = paddleocr_repo_dir(paddle_py)
    tools_train = repo_dir / "tools" / "train.py"
    if tools_train.is_file():
        return repo_dir

    if bundled_paddleocr_api(paddle_py).is_dir() and not force_repo_install:
        print(
            "PaddleOCR_api există în PaddleX, dar pentru train/export este necesar "
            "repo-ul PaddleOCR (tools/train.py). Încerc instalare automată..."
        )

    if repo_dir.is_dir() and tools_train.is_file():
        return repo_dir

    print("Instalare repo PaddleOCR pentru antrenare (git clone / paddlex)...")
    try:
        run_cmd([*_paddlex_cli(paddle_py), "--install", "PaddleOCR", "-y", "--no_deps"], cwd=cwd)
    except subprocess.CalledProcessError:
        print("paddlex --install a eșuat; încerc git clone direct...")
        try:
            _clone_paddleocr_repo(repo_dir)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                "Nu pot instala repo PaddleOCR pentru train/export.\n"
                f"Detalii: {exc}"
            ) from exc
    if not tools_train.is_file():
        raise SystemExit(
            f"Instalarea s-a încheiat, dar lipsește {tools_train}. "
            "Rulează manual `paddlex --install PaddleOCR -y --no_deps` sau verifică rețeaua."
        )
    return repo_dir


def _venv_can_import(paddle_py: Path, module: str) -> bool:
    proc = subprocess.run(
        [str(paddle_py), "-c", f"import {module}"],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def ensure_paddlex_package(paddle_py: Path) -> None:
    if _venv_can_import(paddle_py, "paddlex"):
        return
    print("Instalare paddlex (necesar pentru antrenare)...")
    run_cmd([str(paddle_py), "-m", "pip", "install", "paddlex"])


def ensure_training_deps(paddle_py: Path) -> None:
    """matplotlib, PaddleOCR train deps etc. în venv-ul paddle-ocr."""
    ver = subprocess.run(
        [str(paddle_py), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if ver >= "3.13":
        print(
            "Atentie: Python 3.13 + Paddle CPU pe Windows poate crăpa la antrenare (MKL/numpy). "
            "Recomandat: recreează venv paddle-ocr cu Python 3.11 (py -3.11 -m venv .venv)."
        )
    train_modules = (
        "matplotlib",
        "cv2",
        "skimage",
        "albumentations",
        "albucore",
        "shapely",
        "pyclipper",
        "lmdb",
        "yaml",
        "tqdm",
        "rapidfuzz",
    )
    if all(_venv_can_import(paddle_py, m) for m in train_modules):
        return
    root = repo_root()
    req_files: list[Path] = []
    ocr_req = paddleocr_repo_dir(paddle_py) / "requirements.txt"
    local_req = root / "requirements-train.txt"
    if ocr_req.is_file():
        req_files.append(ocr_req)
    if local_req.is_file():
        req_files.append(local_req)
    if not req_files:
        raise SystemExit("Lipsesc dependențe antrenare și requirements-train.txt")
    print("Instalare dependențe antrenare (PaddleOCR + requirements-train.txt)...")
    for req in req_files:
        run_cmd([str(paddle_py), "-m", "pip", "install", "-r", str(req)])


def resolve_trained_pdparams(output_dir: Path) -> Path | None:
    """Calea către .pdparams după antrenare (best_accuracy sau iter_epoch_*)."""
    canonical = output_dir / "best_accuracy" / "best_accuracy.pdparams"
    if canonical.is_file():
        return canonical

    train_result = output_dir / "train_result.json"
    if train_result.is_file():
        try:
            data = json.loads(train_result.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        models = data.get("models") if isinstance(data, dict) else None
        if isinstance(models, dict):
            for key in ("best", "last_1", "last_2", "last_3"):
                entry = models.get(key)
                if not isinstance(entry, dict):
                    continue
                rel = entry.get("pdparams")
                if not rel:
                    continue
                candidate = output_dir / str(rel).replace("\\", "/")
                if candidate.is_file():
                    return candidate

    found = sorted(
        output_dir.rglob("*.pdparams"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return found[0] if found else None


def ensure_config_beside_weights(pdparams: Path, output_dir: Path) -> Path:
    """
    PaddleOCR export_model.py caută config.yaml în același folder cu weights.
    Fără el, folosește un stub PaddleX fără PostProcess → KeyError.
    """
    weight_dir = pdparams.parent
    dest = weight_dir / "config.yaml"
    if dest.is_file():
        return dest

    for src_name in ("config.yaml", "config.yml"):
        src = output_dir / src_name
        if src.is_file():
            shutil.copy2(src, dest)
            print(f"Copiat {src.name} -> {dest}")
            return dest

    runtime = output_dir / "ocr_train_runtime.yml"
    if runtime.is_file():
        shutil.copy2(runtime, dest)
        print(f"Copiat ocr_train_runtime.yml -> {dest}")
        return dest

    raise SystemExit(
        f"Lipsește config antrenare în {output_dir} (config.yaml). "
        "Rulează antrenarea completă înainte de export."
    )


def inference_dir_is_deployable(path: Path) -> bool:
    if not path.is_dir():
        return False
    if any(path.glob("*.pdiparams")) or (path / "inference.pdiparams").is_file():
        return True
    # PIR / JSON-only (unele rulari Paddle 3) — acceptat dacă există ambele
    return (path / "inference.yml").is_file() and (path / "inference.json").is_file()


def find_training_inference(output_dir: Path) -> Path | None:
    """Inference produs de tools/train.py (save_inference la fiecare epocă)."""
    ordered = [
        output_dir / "latest" / "inference",
        output_dir / "best_accuracy" / "inference",
        output_dir / "export" / "inference",
    ]
    train_result = output_dir / "train_result.json"
    if train_result.is_file():
        try:
            data = json.loads(train_result.read_text(encoding="utf-8"))
            models = data.get("models") if isinstance(data, dict) else {}
            if isinstance(models, dict):
                for key in ("best", "last_1", "last_2", "last_3"):
                    entry = models.get(key)
                    if not isinstance(entry, dict):
                        continue
                    rel = entry.get("inference_config") or entry.get("pdmodel")
                    if not rel:
                        continue
                    inf = output_dir / Path(str(rel).replace("\\", "/"))
                    if inf.is_file():
                        inf = inf.parent
                    if inference_dir_is_deployable(inf):
                        ordered.insert(0, inf)
        except json.JSONDecodeError:
            pass

    for candidate in ordered:
        if inference_dir_is_deployable(candidate):
            return candidate

    for candidate in sorted(
        output_dir.rglob("inference"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        if inference_dir_is_deployable(candidate):
            return candidate
    return None


def run_paddleocr_export_model(
    paddle_py: Path,
    ocr_repo: Path,
    train_config: Path,
    pdparams: Path,
    save_dir: Path,
    env: dict[str, str] | None,
) -> None:
    """Export direct cu config complet (PostProcess), fără stub PaddleX."""
    save_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(paddle_py),
        str(ocr_repo / "tools" / "export_model.py"),
        "-c",
        str(train_config),
        "-o",
        f"Global.pretrained_model={pdparams}",
        f"Global.save_inference_dir={save_dir}",
    ]
    run_cmd(cmd, cwd=ocr_repo, env=env)


def find_inference_dir(output: Path, export_dir: Path | None) -> Path:
    if export_dir:
        base = Path(export_dir).resolve()
        if (base / "inference").is_dir():
            return base / "inference"
        if (base / "inference" / "inference.yml").is_file() or (base / "inference.json").is_file():
            return base / "inference"
        if base.name == "inference" or (base / "inference.pdiparams").exists():
            return base
        raise SystemExit(f"Nu găsesc inference în {base}")

    candidates = [
        output / "best_accuracy" / "inference",
        output / "best_accuracy",
        output / "export" / "inference",
    ]
    for c in candidates:
        if c.is_dir() and any(c.glob("inference.*")) or (c / "inference.yml").is_file():
            return c
    for c in output.rglob("inference"):
        if c.is_dir() and (list(c.glob("*.pdiparams")) or (c / "inference.yml").is_file()):
            return c
    raise SystemExit(f"Nu găsesc folder inference sub {output}. Rulează export sau --export-dir.")


def _copy_inference_export(inference_src: Path, dest_root: Path) -> Path:
    deployed = dest_root / "inference"
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(inference_src, deployed)
    return deployed


def deploy_to_paddle_ocr(inference_src: Path, model_name: str, *, ocr_root: Path) -> Path:
    dest_root = ocr_root / "models" / model_name
    deployed = _copy_inference_export(inference_src, dest_root)

    env_path = ocr_root / ".env"
    rel_model = f"models/{model_name}/inference"
    rec_name = _inference_model_name_from_yml(deployed)
    _patch_env(env_path, rel_model, rec_name)
    print(f"Model deployat: {deployed}")
    print(f".env actualizat: PADDLE_REC_MODEL_DIR={rel_model}")
    print(f".env actualizat: PADDLE_REC_MODEL_NAME={rec_name}")
    return deployed


def deploy_to_exports(inference_src: Path, model_name: str) -> Path:
    dest_root = repo_root() / "exports" / model_name
    deployed = _copy_inference_export(inference_src, dest_root)
    print(f"Model exportat (standalone): {deployed}")
    print("Copiază în FRCHub: services/paddle-ocr/models/ sau folosește colab/package_model_export.py")
    return deployed


def _inference_model_name_from_yml(inference_dir: Path) -> str:
    yml = inference_dir / "inference.yml"
    if yml.is_file():
        for line in yml.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if stripped.startswith("model_name:"):
                name = stripped.split(":", 1)[1].strip()
                if name:
                    return name
    return "latin_PP-OCRv5_mobile_rec"


def _patch_env(env_path: Path, model_dir_rel: str, rec_model_name: str) -> None:
    lines: list[str] = []
    if env_path.is_file():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    updates = {
        "PADDLE_REC_MODEL_DIR": model_dir_rel,
        "PADDLE_REC_MODEL_NAME": rec_model_name,
    }
    out: list[str] = []
    found: dict[str, bool] = {k: False for k in updates}
    for line in lines:
        replaced = False
        for key, value in updates.items():
            if line.startswith(f"{key}=") or line.startswith(f"{key} "):
                out.append(f"{key}={value}")
                found[key] = True
                replaced = True
                break
        if not replaced:
            out.append(line)
    for key, value in updates.items():
        if not found[key]:
            out.append(f"{key}={value}")
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    root = repo_root()
    ds = Path(args.dataset)
    dataset_root = ds.resolve() if ds.is_absolute() else (root / ds).resolve()
    output_dir = (root / args.output).resolve()
    ocr_root = paddle_ocr_root(args.paddle_ocr_root)
    paddle_py = Path(args.paddle_python) if args.paddle_python else default_paddle_python(args.paddle_ocr_root)
    deploy_to_service = is_frchub_monorepo_layout() and not args.skip_deploy

    if not paddle_py.is_file():
        hint = (
            "FRCHub: cd services/paddle-ocr ; .\\run.ps1\n"
            "Repo standalone: python -m venv .venv ; pip install -r requirements-paddle-train.txt"
        )
        raise SystemExit(f"Python negăsit: {paddle_py}\n{hint}")

    if deploy_to_service:
        print(f"Deploy țintă: {ocr_root / 'models' / args.model_name}")
    else:
        print(f"Export standalone: {root / 'exports' / args.model_name}")

    device = args.device or os.getenv("PADDLE_DEVICE", "cpu")

    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_enable_mkldnn", "0")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    prepare_script = root / "scripts" / "prepare_paddlex_dataset.py"
    paddlex_data = dataset_root / "paddlex"
    if not args.skip_train:
        if not (dataset_root / "labels" / "train.txt").is_file():
            raise SystemExit(f"Lipsește {dataset_root / 'labels' / 'train.txt'} — rulează generate.py")
        ensure_paddlex_package(paddle_py)
        run_cmd([sys.executable, str(prepare_script), "--dataset", str(dataset_root)], cwd=root)
        ocr_repo = ensure_paddlex_runtime(
            paddle_py,
            ocr_root,
            force_repo_install=args.install_paddleocr_repo,
        )
        ensure_training_deps(paddle_py)

        config_tpl = root / "config" / "paddlex_train_ci.yaml"
        run_config = output_dir / "run_config.yaml"
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.fast:
            apply_fast_train_patches(ocr_repo)
        ocr_src = paddleocr_train_config(paddle_py, fast=args.fast)
        batch_size = (
            args.batch_size
            if args.batch_size is not None
            else default_batch_size(device, fast=args.fast)
        )
        num_workers = (
            args.num_workers
            if args.num_workers is not None
            else default_num_workers(device, fast=args.fast)
        )
        train_epochs = args.epochs if args.epochs is not None else 15
        print_train_plan(paddlex_data, device, train_epochs, batch_size, fast=args.fast)
        ocr_train_cfg = materialize_ocr_config(
            ocr_src,
            output_dir / "ocr_train_runtime.yml",
            num_workers=num_workers,
            batch_size=batch_size,
        )
        text = config_tpl.read_text(encoding="utf-8")
        text = text.replace("PLACEHOLDER_DATASET", str(paddlex_data).replace("\\", "/"))
        text = text.replace("PLACEHOLDER_OUTPUT", str(output_dir).replace("\\", "/"))
        text = text.replace("PLACEHOLDER_OCR_TRAIN_CONFIG", str(ocr_train_cfg).replace("\\", "/"))
        run_config.write_text(text, encoding="utf-8")

        overrides = [
            f"Global.dataset_dir={paddlex_data}",
            f"Global.device={device}",
            f"Global.output={output_dir}",
        ]
        if not args.skip_check_dataset:
            check_overrides = ["Global.mode=check_dataset", *overrides]
            print("=== Verificare dataset ===")
            # Matplotlib deep_analyse încearcă GUI backend (Tk) pe Windows;
            # forțăm backend non-GUI ca să nu depindă de Tcl/Tk (Laragon etc.).
            run_paddlex_engine(
                paddle_py,
                run_config,
                check_overrides,
                ocr_root,
                env={
                    "MPLBACKEND": "Agg",
                    "PADDLE_PDX_PADDLEOCR_PATH": str(ocr_repo),
                },
            )
        else:
            print("=== Verificare dataset: sărit (--skip-check-dataset) ===")

        train_overrides = [
            "Global.mode=train",
            f"Train.basic_config_path={ocr_train_cfg}",
            *overrides,
        ]
        train_overrides.append(f"Train.epochs_iters={train_epochs}")
        train_overrides.append(f"Train.batch_size={batch_size}")
        if args.fast:
            train_overrides.append("Train.eval_interval=5")
        print("=== Antrenare (GPU recomandat; pe CPU foloseste -Fast) ===")
        run_paddlex_engine(
            paddle_py,
            run_config,
            train_overrides,
            ocr_root,
            env={"PADDLE_PDX_PADDLEOCR_PATH": str(ocr_repo)},
        )

    if not args.skip_export:
        existing_inf = find_training_inference(output_dir)
        pdparams = resolve_trained_pdparams(output_dir)
        export_inf_dir = output_dir / "best_accuracy" / "inference"

        if existing_inf and inference_dir_is_deployable(existing_inf):
            print(f"=== Inference de la antrenare (sărit export PaddleX): {existing_inf} ===")
        elif pdparams is None:
            raise SystemExit(
                f"Nu găsesc .pdparams sub {output_dir}. "
                "Rulează antrenarea sau folosește --skip-export --export-dir <folder inference>."
            )
        else:
            train_config = ensure_config_beside_weights(pdparams, output_dir)
            repo_dir = paddleocr_repo_dir(paddle_py)
            export_env = {"PADDLE_PDX_PADDLEOCR_PATH": str(repo_dir)} if repo_dir.is_dir() else None

            print("=== Export inference ===")
            print(f"  weights: {pdparams}")
            print(f"  config:  {train_config}")

            if (repo_dir / "tools" / "export_model.py").is_file():
                try:
                    run_paddleocr_export_model(
                        paddle_py,
                        repo_dir,
                        train_config,
                        pdparams,
                        export_inf_dir,
                        export_env,
                    )
                except subprocess.CalledProcessError:
                    print("Export direct PaddleOCR eșuat; încerc PaddleX export...")
                    export_overrides = [
                        "Global.mode=export",
                        f"Global.dataset_dir={paddlex_data if paddlex_data.is_dir() else dataset_root}",
                        f"Global.device={device}",
                        f"Global.output={output_dir}",
                        f"Export.weight_path={pdparams}",
                    ]
                    run_paddlex_engine(
                        paddle_py,
                        output_dir / "run_config.yaml",
                        export_overrides,
                        ocr_root,
                        env=export_env,
                    )
            else:
                export_overrides = [
                    "Global.mode=export",
                    f"Global.dataset_dir={paddlex_data if paddlex_data.is_dir() else dataset_root}",
                    f"Global.device={device}",
                    f"Global.output={output_dir}",
                    f"Export.weight_path={pdparams}",
                ]
                run_paddlex_engine(
                    paddle_py,
                    output_dir / "run_config.yaml",
                    export_overrides,
                    ocr_root,
                    env=export_env,
                )

    export_dir = Path(args.export_dir).resolve() if args.export_dir else None
    inference = find_inference_dir(output_dir, export_dir)
    if deploy_to_service:
        deploy_to_paddle_ocr(inference, args.model_name, ocr_root=ocr_root)
        print("Gata. Repornește serviciul paddle-ocr (run.ps1 sau Docker).")
    else:
        deploy_to_exports(inference, args.model_name)
        print("Gata. Copiază exports/ în paddle-ocr/models/ sau descarcă model_export.zip din Colab.")


if __name__ == "__main__":
    main()
