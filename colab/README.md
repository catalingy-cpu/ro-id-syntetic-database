# ro-id pe Google Colab — generare + antrenare

Tot pipeline-ul într-un singur notebook: **imagini sintetice** → **antrenare PaddleOCR** → **`model_export.zip`**.

## Pași rapizi

1. Deschide [train_paddleocr.ipynb](./train_paddleocr.ipynb) în Colab  
   `https://colab.research.google.com/github/USER/ro-id/blob/main/colab/train_paddleocr.ipynb`
2. Setează `REPO_URL` (repo GitHub `ro-id`)
3. **Runtime → GPU**
4. **Run All**
5. Descarcă `model_export.zip` → copiază în `FRCHub/services/paddle-ocr/models/`

## Configurare importantă

```python
REPO_URL = "https://github.com/USER/ro-id.git"

GENERATE_DATASET = True
GENERATE_MULTI = True                # clasic + electronic + telefon
GENERATE_MIX = "config/generation_mix.json"
GENERATE_COUNT = 5000
SKIP_GENERATION_IF_EXISTS = True

DATASET_DIR = "/content/drive/MyDrive/ro-id/dataset_colab"
MOUNT_GOOGLE_DRIVE = True
```

`classic_reference` (ROI calibrat) rămâne în mix — nu e atins. Pentru un singur template: `GENERATE_MULTI = False`.

### De ce Google Drive?

Spațiul `/content` din Colab **se șterge** când sesiunea expiră. Datasetul generat rămâne pe Drive dacă `DATASET_DIR` pointează acolo.

## Moduri dataset

| Mod | Setări |
|-----|--------|
| **Generează în Colab** (implicit) | `GENERATE_DATASET = True` |
| **Refolosește de pe Drive** | `SKIP_GENERATION_IF_EXISTS = True` + același `DATASET_DIR` |
| **Upload ZIP** | `GENERATE_DATASET = False`, `UPLOAD_DATASET_ZIP = True` |
| **Doar antrenare** (fără regen) | `GENERATE_DATASET = False` + dataset deja la `DATASET_DIR` |

## Timp estimat

| Pas | Durată orientativă |
|-----|-------------------|
| Generare 5k imagini | 30–90 min (CPU Colab) |
| Generare 50k | câteva ore |
| Antrenare 15 epoci (GPU T4) | 1–4 ore |

## Template-uri

Specimen-ele CI (`templates/*.png`) sunt **în git** — Colab le clonează automat. Nu încărca poze reale de buletin.

## Publicare repo

Vezi [GITHUB.md](../GITHUB.md).
