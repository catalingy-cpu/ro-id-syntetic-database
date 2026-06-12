# Generator dataset CI — template-based

Editează o **imagine template** de CI (layout real), înlocuiește textul cu date **100% sintetice**, apoi simulează poze de telefon pentru antrenare OCR.

**Nu** construiește documentul de la zero. **Nu** păstrează date reale în output.

## Arhitectură (7 faze)

| Fază | Modul | Descriere |
|------|--------|-----------|
| 1 | `tools/analyze_template.py` + `config/template_fields.json` | Coordonate câmpuri editabile |
| 2 | `ro_id_synth/field_replace.py` | Inpaint doar pixeli text |
| 3 | `ro_id_synth/text_style.py` + `text_render.py` | Font/culoare/baseline estimate ±5% |
| 4 | `ro_id_synth/records.py` | Identități fictive Faker + CNP valid |
| 5 | `ro_id_synth/scenarios.py` | Fundal + cameră (80/15/5 clean/med/hard) |
| 6 | `debug/` | `original.jpg`, `replaced_fields.jpg`, `final_sample.jpg`, `grid_XXXX.jpg` |
| 7 | `ro_id_synth/quality_filter.py` | OCR ≥95% sau euristică |

## Setup

```powershell
cd services\ro-id-synthetic-dataset
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Plasează template-ul HD în `templates/classic_reference.png` (local, gitignored dacă e poză reală).

## Calibrare câmpuri (Faza 1)

Coordonatele sunt în **pixelii template-ului** (`1024×725`). Dacă ai o poză de referință cu cutii portocalii pe aceeași rezoluție:

```powershell
python tools/import_orange_reference.py path\to\referinta_portocalie.png
python tools/analyze_template.py --preview
# → debug/fields_overlay.jpg (verde) trebuie să acopere valorile, nu etichetele
```

Ajustare manuală:

```powershell
python tools/analyze_template.py --set surname --x 358 --y 183 --width 287 --height 54 --preview
```

## Generare

Triplet debug + analiză:

```powershell
python generate.py --analyze
```

Grid 100 sample-uri:

```powershell
python generate.py --debug-grid --output dataset_test
```

Producție (contact sheet la fiecare 200 imagini):

```powershell
python generate.py --count 10000 --workers 4 --output dataset
```

## Filtru OCR (Faza 7)

```powershell
$env:SYNTH_OCR_URL = "http://localhost:8765"
python generate.py --count 5000 --output dataset
```

## Distribuție foto

- **80%** clean — card 60–95% lățime, text lizibil
- **15%** medium — blur/perspective/JPEG ușoare
- **5%** hard — perspective mai puternică, lumină slabă, crop ușor

## Antrenare PaddleOCR

Vezi [TRAINING.md](TRAINING.md).
