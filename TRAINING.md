# Antrenare PaddleOCR pe dataset sintetic CI

Ghid pentru rulare **noaptea**: ce rulezi unde, ce intră în git și cum folosești modelul în `services/paddle-ocr`.

---

## Start rapid (smoke → Colab → deploy)

```powershell
cd d:\FRCHub\services\ro-id-synthetic-dataset
.\scripts\smoke_pipeline.ps1 -Count 800
```

Creează `dataset_smoke/` + `dataset_smoke.zip` cu **line-crops ROI** (nu benzi orizontale).

1. Încarcă `dataset_smoke.zip` pe Google Drive.
2. Deschide `colab/train_paddleocr.ipynb` → `TRAIN_PROFILE=smoke`, `GENERATE_DATASET=False`, `UPLOAD_DATASET_ZIP=True`.
3. După antrenare: decomentează `PADDLE_REC_MODEL_DIR` în `services/paddle-ocr/.env` și repornește `run.ps1`.
4. Test: `paddle-ocr\scripts\eval_ci_fields.py --dataset dataset_smoke --limit 30`

ROI inferență (`paddle-ocr/app/roi_layout.py`) trebuie să coincidă cu `config/template_fields.json`:
`python scripts/sync_roi_layout.py` (afișează constantele).

---


| Pas | Unde îl rulezi | În git? |
|-----|----------------|---------|
| **1. Generare imagini** (`generate.py`) | PC local, VPS CPU sau **Google Colab** (notebook `colab/`) | **Nu** — folderul `dataset/` e mare (zeci de GB la 50k) |
| **2. Template-uri** (`templates/*.png` specimen) | În git | **Da** |
| **3. Conversie etichete** | Oriunde ai `dataset/` | **Da** — scriptul `scripts/convert_to_paddle_rec.py` |
| **4. Antrenare model** | Mașină cu **GPU** (PC local cu NVIDIA sau VPS cloud) | **Nu** — greutăți `.pdparams` / export (sute MB–GB) |
| **5. Serviciu OCR producție** | Serverul unde rulează Laravel + `paddle-ocr` | **Nu** modelul — îl copiezi manual / SFTP / volume Docker |

**Nu** pui dataset-ul sau modelul antrenat în git. În git rămân doar codul (generator, scripturi, config ROI).

**O dată local + git?** — Poți genera datasetul **o singură dată** pe PC, antrena **o dată** (GPU), apoi **copiezi modelul exportat** pe serverul PaddleOCR. Git nu e pentru dataset/model; e pentru sursă.

**Server vs local:**

- **Generare 50k imagini** — poți lăsa **noaptea pe PC-ul tău** (CPU, 8 workers). Nu ai nevoie de server dedicat dacă ai ~30–50 GB liberi pe disc.
- **Antrenare** — practic **necesită GPU**. Fie PC local cu placă video, fie VPS GPU (Lambda, RunPod, etc.). Pe CPU pur antrenarea poate dura zile/săptămâni — nerecomandat.
- **Producție** — microserviciul `paddle-ocr` rulează pe **același server** ca API-ul (sau VPS separat), cu modelul copiat acolo, nu în repo.

---

## Pas 0 — Pregătire (o singură dată)

```powershell
cd d:\FRCHub\services\ro-id-synthetic-dataset
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Template-urile specimen sunt deja în `templates/` (vezi `templates/README.md`):

- `scan_flat.png` — CI clasic
- `eid_flat.png` — CI electronic (scan)
- `phone_angle.png` — CI electronic (telefon)

---

## Pas 1 — Generare dataset (noaptea, CPU)

Test scurt:

```powershell
.\.venv\Scripts\activate
python generate.py --count 100 --workers 2 --batch-size 50 --output dataset_test
```

Rulare completă (exemplu 50.000 imagini):

```powershell
python generate.py --count 50000 --workers 8 --batch-size 200 --output dataset
```

Durată orientativă: ore–zeci de ore pe CPU (depinde de CPU și `--workers`).

Rezultat:

```
dataset/
  images/0000000.jpg ...
  labels/train.txt
```

**Nu comita** `dataset/` — e deja în `.gitignore`.

---

## Pas 2 — Conversie etichete pentru Paddle (rec)

```powershell
python scripts/convert_to_paddle_rec.py --dataset dataset --val-ratio 0.05
```

Creează:

```
dataset/paddle_rec/train_list.txt   # images/xxx.jpg<TAB>NUME: ...
dataset/paddle_rec/val_list.txt
```

Format listă: cale relativă la `dataset/` + tab + transcrierea (blocul NUME/PRENUME/CNP/…).

---

## Pas 3 — Antrenare + deploy automat (recomandat)

Scriptul face tot lanțul: pregătire dataset PaddleX → antrenare → export → copiere în `services/paddle-ocr/models/` → actualizare `.env`.

**Rulează din `services/ro-id-synthetic-dataset`**, folosind Python-ul din `services/paddle-ocr/.venv` (același PaddleX ca la inferență).

### Pregătire (o dată)

```powershell
cd d:\FRCHub\services\paddle-ocr
.\run.ps1
# Ctrl+C după ce a descărcat modelele; sau lasă pornit în alt terminal
```

Antrenarea folosește `PaddleOCR_api` din pachetul PaddleX — **nu** e nevoie de `paddlex --install PaddleOCR` (git clone). Doar dacă vrei explicit: `.\scripts\train_and_deploy.ps1 -InstallPaddleocrRepo`.

`check_dataset` cere **matplotlib** în venv-ul `paddle-ocr` — scriptul îl instalează automat, sau manual:

```powershell
..\paddle-ocr\.venv\Scripts\pip install -r requirements-train.txt
```

### Noaptea — după `generate.py`

```powershell
cd d:\FRCHub\services\ro-id-synthetic-dataset

# GPU (recomandat):
.\scripts\train_and_deploy.ps1 -Dataset dataset -Device gpu:0 -Epochs 15

# CPU / laptop fara GPU — profil rapid (batch mic pe CPU AMD/Intel):
.\scripts\train_and_deploy.ps1 -Dataset dataset -Device cpu -Epochs 3 -Fast -BatchSize 8 -SkipCheckDataset

# Batch explicit (mai multe imagini per pas = epoca mai scurta):
.\scripts\train_and_deploy.ps1 -Dataset dataset -Device gpu:0 -Fast -BatchSize 128 -Epochs 15
```

**De ce „20 zile / epoca”?** Configul implicit PaddleOCR foloseste `RecConAug` + `RecAug` (distorsionari pe imagini mari) si incarca **2 imagini extra** la fiecare sample — pe CPU e extrem de lent. `-Fast` elimina asta si foloseste **line-crops** din `prepare_paddlex_dataset.py`.

Sau:

```powershell
..\paddle-ocr\.venv\Scripts\python.exe scripts\train_and_deploy.py --dataset dataset --device gpu:0 --epochs 15
```

La final:

- Model în `services/paddle-ocr/models/frc_ci_rec/inference/`
- `services/paddle-ocr/.env` primește `PADDLE_REC_MODEL_DIR=models/frc_ci_rec/inference`
- **Repornești** `paddle-ocr` (`.\run.ps1`)

### Doar deploy (model deja antrenat)

```powershell
.\scripts\train_and_deploy.ps1 -SkipTrain -ExportDir training_output\best_accuracy
```

### Ce face scriptul

1. `prepare_paddlex_dataset.py` — `dataset/paddlex/` cu `train.txt`, `val.txt`, `dict.txt`, junction `images/`
2. PaddleX `check_dataset` → `train` → `export`
3. Copiază `inference/` în `paddle-ocr/models/frc_ci_rec/`
4. Patch `.env` — fără copiere manuală

Config antrenare: `config/paddlex_train_ci.yaml` (epoci, batch, lr).

---

## Pas 4 — Deploy pe server (fără git pentru model)

Pe serverul de producție:

1. **Nu** urca `dataset/` sau template-uri reale.
2. Copiază doar folderul model exportat (ex. `rsync`, SCP, volume Docker montat).
3. `backend/.env`: `OCR_DRIVER=paddle`, `PADDLE_OCR_URL=...`
4. `services/paddle-ocr/.env`: `PADDLE_DEVICE=gpu` dacă serverul are GPU.

---

## Google Colab (generare + antrenare)

Tot pipeline-ul poate rula în cloud, fără PC:

1. Publică repo-ul `ro-id` pe GitHub (vezi [GITHUB.md](GITHUB.md)) — include `templates/*.png` specimen și `templates/classic_reference.png`.
2. Deschide [colab/train_paddleocr.ipynb](colab/train_paddleocr.ipynb) → **Runtime → GPU** → **Run All**.
3. Setează `REPO_URL` și, recomandat, `DATASET_DIR` pe Google Drive (datasetul rămâne după ce expiră sesiunea Colab).

```python
GENERATE_DATASET = True
GENERATE_COUNT = 5000
DATASET_DIR = "/content/drive/MyDrive/ro-id/dataset_colab"
MOUNT_GOOGLE_DRIVE = True
SKIP_GENERATION_IF_EXISTS = True   # a doua rulare: doar antrenare
```

Detalii, timpi estimați și moduri (upload ZIP, doar antrenare): [colab/README.md](colab/README.md).

---

## Checklist rulare noaptea (PC local)

1. [x] Template-uri specimen în `templates/` (din git)
2. [ ] `python generate.py --count 50000 ...` (lasă terminalul deschis / `nohup`)
3. [ ] Dimineața: `python scripts/convert_to_paddle_rec.py --dataset dataset`
4. [ ] Dacă ai GPU: pornește antrenarea Paddle pe `train_list.txt`
5. [ ] După export: copiază modelul → repornește `paddle-ocr`
6. [ ] **Nu** `git add dataset/` sau `.pdparams`

---

## Întrebări frecvente

**Pot rula totul pe serverul Laravel?**  
Generarea da (CPU), dacă ai spațiu. Antrenarea doar dacă serverul are GPU. Altfel generezi local, copiezi doar modelul finit pe server.

**Salvez modelul în git?**  
Nu. E prea mare și se schimbă des. Versionezi doar codul; modelul = artefact de deploy.

**Am nevoie de antrenare ca să folosesc app-ul?**  
Nu. App-ul merge cu modelul mobil pre-antrenat `latin_PP-OCRv5_mobile_rec` + mod ROI. Antrenarea e opțională pentru acuratețe mai bună pe buletine reale.

**Ce fac dacă nu am GPU?**  
Poți genera datasetul noaptea pe CPU; pentru antrenare închiriază un VPS GPU câteva ore sau sari antrenarea și rămâi la modelul implicit + parserul din plugin.

**Export: `KeyError: 'PostProcess'` sau „config.yaml … is not exist, use default instead”?**  
PaddleOCR export cere `config.yaml` **în același folder** cu `best_accuracy.pdparams` (cu secțiunea `PostProcess`). Antrenarea scrie `training_output/config.yaml`; scriptul `train_and_deploy.py` îl copiază automat lângă weights înainte de export. Dacă ai rulat doar export manual:

```powershell
copy training_output\config.yaml training_output\best_accuracy\config.yaml
.\scripts\train_and_deploy.ps1 -SkipTrain
```

Dacă antrenarea a produs deja `latest/inference` sau `iter_epoch_*\inference`, deploy-ul poate sări exportul PaddleX.

**Antrenare CPU: `Intel MKL ERROR: vsAdd` / exit `3221225477`?**  
Crash MKL la primul pas de antrenare — frecvent pe **AMD (ex. Vega 8)** sau **Python 3.13 + numpy 2.x**. Scripturile setează acum `MKL_DEBUG_CPU_TYPE=5` și dezactivează oneDNN. Reîncearcă:

```powershell
.\scripts\train_and_deploy.ps1 -Dataset dataset -Device cpu -Fast -BatchSize 8 -Epochs 3 -SkipCheckDataset
```

Dacă persistă: recreează venv-ul `services/paddle-ocr` cu **Python 3.11** (`py -3.11 -m venv .venv`), reinstalează paddle/paddlex, apoi rulează din nou antrenarea.

---

Vezi și: [README.md](README.md) (generator), [services/paddle-ocr/README.md](../paddle-ocr/README.md) (inferență).
