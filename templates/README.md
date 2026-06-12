# Template-uri CI (SPECIMEN — în git)

Imagini **oficiale specimen** (fără date personale reale), pot fi versionate și folosite pe orice mașină fără copiere manuală.

| Fișier | Descriere |
|--------|-----------|
| `scan_flat.png` | CI clasic, scan plat mic (legacy) |
| `classic_popescu.png` | CI clasic specimen POPESCU (600×423, HD) |
| `classic_cenuse.png` | CI clasic specimen vechi CENUSE (600×346) |
| `eid_flat.png` | CI electronic, scan plat (specimen MANOLE) |
| `eid_specimen_v2.png` | CI electronic specimen v2 (MANOLE, calitate mai bună) |
| `phone_angle.png` | CI electronic, poză telefon (specimen MANOLE) |
| `eid_phone_hand.png` | CI electronic, poză cu mână (specimen MANOLE) |

Generatorul **nu** copiază textul din poze — șterge ROI-urile și desenează date **fictive** (Faker + CNP algoritmic).

Coordonate ROI: `config/templates.yaml`. Dacă textul fictiv nu se aliniază pe un template, ajustează valorile normalizate (0–1) acolo.

**Nu** înlocui aceste fișiere cu poze reale de buletin — nu le comite și nu le urca pe server.

