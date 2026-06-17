# ro-id pe Google Colab — generare + antrenare

Tot pipeline-ul într-un singur notebook: **imagini sintetice** → **antrenare PaddleOCR** → **`model_export.zip`**.

## Pași rapizi

1. Deschide [train_paddleocr.ipynb](./train_paddleocr.ipynb) în Colab
2. Setează `REPO_URL` (repo GitHub)
3. **Runtime → GPU**
4. Lasă `TRAIN_PROFILE = "smoke"` și **Run All**
5. Descarcă `model_export.zip` → testează pe CI reale în FRCHub
6. Dacă e OK → `TRAIN_PROFILE = "main"` și rulează din nou

## Profiluri antrenare (fail-fast)

| Profil | Ce face | Când |
|--------|---------|------|
| **`smoke`** (implicit) | `classic_reference` only, 1200 img, 3 epoci | Întotdeauna primul run |
| **`main`** | `classic_reference` only, 6000 img, 10 epoci | Doar după ce smoke e OK pe poze reale |

În celula de config:

```python
TRAIN_PROFILE = "smoke"  # sau "main"
FORCE_REFRESH_REPO = True
GENERATE_MULTI = False   # setat automat de profil — NU multi până la calibrare ROI
```

## Configurare importantă

```python
REPO_URL = "https://github.com/USER/ro-id.git"
DATASET_BASE = "/content/drive/MyDrive/ro-id"
MOUNT_GOOGLE_DRIVE = True
WIPE_PADDLEX_BEFORE_TRAIN = True
```

Datasetul merge în subfoldere separate per profil (`dataset_classic_smoke`, `dataset_classic_main`).

### De ce Google Drive?

Spațiul `/content` din Colab **se șterge** când sesiunea expiră. Datasetul generat rămâne pe Drive.

## După antrenare

1. Dezarhivează `model_export.zip` în `FRCHub/services/paddle-ocr/models/`
2. În `services/paddle-ocr/.env`:
   ```env
   PADDLE_REC_MODEL_DIR=models/frc_ci_rec/inference
   PADDLE_REC_MODEL_NAME=latin_PP-OCRv5_mobile_rec
   ```
3. Repornește `paddle-ocr` (`.\run.ps1`)

Dacă modelul e prost, comentează cele 2 linii și revii la model stock.

## Timp estimat

| Profil | Durată orientativă |
|--------|-------------------|
| smoke (gen + train) | 30–90 min |
| main (gen + train) | 2–5 ore |

**OOM pe GPU?** `BATCH_SIZE=16`, `USE_FAST=True`, Restart session, rulează doar celula de antrenare.

## Publicare repo

Vezi [GITHUB.md](../GITHUB.md).
