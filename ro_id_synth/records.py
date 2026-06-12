"""Înregistrări fictive (Faker ro_RO) — fără date din template-uri."""

from __future__ import annotations

import random
import string
from dataclasses import dataclass
from datetime import date, timedelta

from faker import Faker

from ro_id_synth.cnp import generate_cnp, random_birth_date

COUNTY_CODES: dict[str, int] = {
    "Alba": 1,
    "Arad": 2,
    "Argeș": 3,
    "Bacău": 4,
    "Bihor": 5,
    "Bistrița-Năsăud": 6,
    "Botoșani": 7,
    "Brăila": 8,
    "Brașov": 9,
    "București": 40,
    "Buzău": 11,
    "Călărași": 12,
    "Caraș-Severin": 13,
    "Cluj": 14,
    "Constanța": 15,
    "Covasna": 16,
    "Dâmbovița": 17,
    "Dolj": 18,
    "Galați": 19,
    "Giurgiu": 20,
    "Gorj": 21,
    "Harghita": 22,
    "Hunedoara": 23,
    "Ialomița": 24,
    "Iași": 25,
    "Ilfov": 26,
    "Maramureș": 27,
    "Mehedinți": 28,
    "Mureș": 29,
    "Neamț": 30,
    "Olt": 31,
    "Prahova": 32,
    "Sălaj": 33,
    "Satu Mare": 34,
    "Sibiu": 35,
    "Suceava": 36,
    "Teleorman": 37,
    "Timiș": 38,
    "Tulcea": 39,
    "Vaslui": 41,
    "Vâlcea": 42,
    "Vrancea": 43,
}

COUNTY_ABBR: dict[str, str] = {
    "Mureș": "Ms",
    "Cluj": "Cj",
    "București": "B",
    "Timiș": "Tm",
    "Iași": "Is",
    "Constanța": "Ct",
    "Brașov": "Bv",
    "Dolj": "Dj",
    "Prahova": "Ph",
    "Sibiu": "Sb",
}


@dataclass(frozen=True)
class SyntheticIdRecord:
    nume: str
    prenume: str
    cnp: str
    serie: str
    numar: str
    data_nasterii: str
    data_nasterii_display: str
    birth_date_display: str
    issue_date_display: str
    expiry_date_display: str
    validity_display: str
    sex: str
    cetatenie: str
    cetatenie_short: str
    adresa: str
    adresa_line1: str
    adresa_line2: str
    loc_nastere: str
    issued_by: str

    def to_transcription(self) -> str:
        lines = [
            f"NUME: {self.nume}",
            f"PRENUME: {self.prenume}",
            f"CNP: {self.cnp}",
            f"SERIE: {self.serie}",
            f"NUMAR: {self.numar}",
            f"DATA_NASTERII: {self.birth_date_display}",
            f"DATA_EMITERE: {self.issue_date_display}",
            f"DATA_EXPIRARE: {self.expiry_date_display}",
            f"SEX: {self.sex}",
            f"CETATENIE: {self.cetatenie}",
            f"EMISA_DE: {self.issued_by}",
        ]
        if self.adresa_line1:
            lines.append(f"ADRESA: {self.adresa_line1}")
        if self.adresa_line2:
            lines.append(f"ADRESA2: {self.adresa_line2}")
        return "\n".join(lines)


def _upper_name(s: str) -> str:
    return s.upper().replace("Ţ", "Ț").replace("Ş", "Ș").replace("Ă", "Ă")


def _random_serie(rng: random.Random) -> str:
    return "".join(rng.choice(string.ascii_uppercase) for _ in range(2))


def _format_address(county: str, city: str, street: str, nr: int, ap: int | None) -> tuple[str, str, str]:
    abbr = COUNTY_ABBR.get(county, county[:2].title())
    loc = f"Jud.{abbr} Mun.{city}"
    line2 = f"Str.{street} nr.{nr}" + (f" ap.{ap}" if ap else "")
    return loc, loc, line2


def _classic_date(d: date) -> str:
    return d.strftime("%d.%m.%y")


def _classic_date_long(d: date) -> str:
    return d.strftime("%d.%m.%Y")


def _issued_by_label(city: str) -> str:
    short = city.replace("Târgu ", "Tg.").replace("Targu ", "Tg.")
    return f"SPCLEP {short}"


def generate_record(seed: int | None = None) -> SyntheticIdRecord:
    rng = random.Random(seed)
    fake = Faker("ro_RO")
    Faker.seed(seed if seed is not None else rng.randint(0, 2**31 - 1))

    sex = rng.choice(["M", "F"])
    birth = random_birth_date(rng)
    county_name = rng.choice(list(COUNTY_CODES.keys()))
    county_code = COUNTY_CODES[county_name]
    city = fake.city().split(",")[0].strip()
    if len(city) < 3:
        city = rng.choice(["Târgu Mureș", "Cluj-Napoca", "Timișoara", "Iași", "Brașov"])

    nume = _upper_name(fake.last_name())
    prenume = _upper_name(fake.first_name())
    if rng.random() < 0.35:
        prenume = f"{prenume}-{_upper_name(fake.first_name())}"

    cnp = generate_cnp(sex=sex, birth=birth, county_code=county_code, rng=rng)
    serie = _random_serie(rng)
    numar = f"{rng.randint(100000, 999999)}"
    street = fake.street_name().replace("Strada ", "").replace("strada ", "").strip()
    nr = rng.randint(1, 200)
    ap = rng.randint(1, 30) if rng.random() < 0.6 else None
    loc, adresa_line1, adresa_line2 = _format_address(county_name, city, street, nr, ap)
    adresa = f"{adresa_line1} {adresa_line2}".strip()

    min_issue = birth + timedelta(days=6570)
    issue = date.today() - timedelta(days=rng.randint(180, 4500))
    if issue < min_issue:
        issue = min_issue + timedelta(days=rng.randint(30, 900))
    expiry = issue + timedelta(days=rng.randint(3650, 4380))

    validity = f"{_classic_date(issue)}-{_classic_date_long(expiry)}"
    cet = "Română / ROU"
    cet_short = "ROU"

    return SyntheticIdRecord(
        nume=nume,
        prenume=prenume,
        cnp=cnp,
        serie=serie,
        numar=numar,
        data_nasterii=birth.isoformat(),
        data_nasterii_display=birth.strftime("%d %m %Y"),
        birth_date_display=_classic_date(birth),
        issue_date_display=_classic_date(issue),
        expiry_date_display=_classic_date_long(expiry),
        validity_display=validity,
        sex=sex,
        cetatenie=cet,
        cetatenie_short=cet_short,
        adresa=adresa,
        adresa_line1=adresa_line1,
        adresa_line2=adresa_line2,
        loc_nastere=loc,
        issued_by=_issued_by_label(city),
    )
