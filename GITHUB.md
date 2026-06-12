# Publicare pe GitHub (repo separat de FRCHub)

Acest folder (`services/ro-id-synthetic-dataset` în monorepo) este gândit să devină **repository propriu**, de ex. `github.com/<user>/ro-id`.

## Ce conține repo-ul standalone

- Generator dataset sintetic CI (`generate.py`, `ro_id_synth/`)
- Antrenare PaddleOCR (`scripts/train_and_deploy.py`)
- Google Colab (`colab/`)
- **Nu** include FRCHub, Laravel sau `services/paddle-ocr` (doar legătură la deploy)

## Pași — repo nou pe GitHub

### 1. Creează repo gol pe GitHub

Exemplu: `ro-id` (fără README, fără .gitignore — le ai deja aici).

### 2. Copiază conținutul (prima dată)

Din monorepo FRCHub, în PowerShell:

```powershell
cd d:\FRCHub
$dest = "d:\ro-id"   # sau alt folder
New-Item -ItemType Directory -Force -Path $dest
robocopy services\ro-id-synthetic-dataset $dest /E /XD dataset dataset_test dataset_v4 training_output .venv __pycache__ exports
cd $dest
git init
git add .
git commit -m "Initial import: ro-id synthetic dataset + PaddleOCR training"
git branch -M main
git remote add origin https://github.com/<USER>/ro-id.git
git push -u origin main
```

`robocopy` exclude dataset-ul și artefactele mari. Nu urca `dataset/`, `training_output/`, `.venv/`.

### 3. Actualizări ulterioare

Lucrezi în `d:\ro-id` (sau clonezi direct de pe GitHub). În FRCHub poți păstra copia din `services/ro-id-synthetic-dataset` sincronizată manual sau prin submodule (opțional).

## Legătura cu FRCHub (după antrenare)

| Unde antrenezi | Unde ajunge modelul |
|----------------|---------------------|
| **În FRCHub** (`services/ro-id-synthetic-dataset` + `../paddle-ocr`) | Automat în `services/paddle-ocr/models/frc_ci_rec/` |
| **Repo standalone** / Colab | `exports/frc_ci_rec/` sau `model_export.zip` |

Copiere manuală în FRCHub:

```powershell
# din repo standalone
xcopy /E /I exports\frc_ci_rec d:\FRCHub\services\paddle-ocr\models\frc_ci_rec
```

Sau dezarhivezi `model_export.zip` în `services/paddle-ocr/models/`.

## Variabile utile (standalone)

| Variabilă | Rol |
|-----------|-----|
| `PADDLE_OCR_ROOT` | Cale către `services/paddle-ocr` dacă vrei deploy direct acolo |
| `PADDLE_DEVICE` | `cpu` sau `gpu:0` |

Exemplu deploy direct în FRCHub de pe mașină locală:

```powershell
cd d:\ro-id
$env:PADDLE_OCR_ROOT = "d:\FRCHub\services\paddle-ocr"
.\scripts\train_and_deploy.ps1 -Dataset dataset_v4 -Device gpu:0
```

## Google Colab (generare + antrenare)

Notebook-ul rulează **tot** în cloud: `generate.py` → antrenare → `model_export.zip`.

1. Asigură-te că `templates/*.png` și `templates/classic_reference.png` sunt în repo (nu doar local).
2. În notebook: `REPO_URL = "https://github.com/<USER>/ro-id.git"`
3. [Open in Colab](https://colab.research.google.com/github/<USER>/ro-id/blob/main/colab/train_paddleocr.ipynb)
4. `GENERATE_DATASET = True`, `DATASET_DIR` pe Google Drive recomandat.

Detalii: [colab/README.md](colab/README.md)

## Păstrare în monorepo FRCHub

Dacă rămâi și în FRCHub, **nimic nu se strică**: layout-ul cu `../paddle-ocr` este detectat automat. Repo-ul separat e doar o copie publică / de lucru independent.
