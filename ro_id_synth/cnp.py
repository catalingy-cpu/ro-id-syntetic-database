"""Generare CNP valid (algoritm cifră de control) — date 100% fictive."""

from __future__ import annotations

import random
from datetime import date, timedelta


def cnp_control_digit(first12: str) -> str:
    weights = [2, 7, 9, 1, 4, 6, 3, 5, 8, 2, 7, 9]
    total = sum(int(d) * w for d, w in zip(first12, weights))
    r = total % 11
    return "1" if r == 10 else str(r)


def sex_century_digit(sex: str, birth: date) -> int:
    y = birth.year
    male = sex.upper() in ("M", "MALE", "B")
    if y >= 2000:
        return 5 if male else 6
    if y >= 1900:
        return 1 if male else 2
    if y >= 1800:
        return 3 if male else 4
    return 7 if male else 8


def generate_cnp(
    *,
    sex: str,
    birth: date,
    county_code: int | None = None,
    rng: random.Random | None = None,
) -> str:
    rng = rng or random.Random()
    cc = county_code if county_code is not None else rng.randint(1, 52)
    s = sex_century_digit(sex, birth)
    yy = birth.year % 100
    mm = birth.month
    dd = birth.day
    serial = rng.randint(1, 999)
    first12 = f"{s}{yy:02d}{mm:02d}{dd:02d}{cc:02d}{serial:03d}"
    return first12 + cnp_control_digit(first12)


def random_birth_date(rng: random.Random, min_age: int = 8, max_age: int = 75) -> date:
    today = date.today()
    start = today - timedelta(days=int(max_age * 365.25))
    end = today - timedelta(days=int(min_age * 365.25))
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, max(1, delta)))
