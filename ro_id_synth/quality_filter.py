"""Validare calitate — fără benzi albe, lizibilitate, similitudine OCR ≥ 95%."""

from __future__ import annotations

import io
import os
import re
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from ro_id_synth.records import SyntheticIdRecord
    from ro_id_synth.roi import TemplateConfig

SIMILARITY_THRESHOLD = float(os.environ.get("SYNTH_OCR_MIN_SIM", "0.95"))
MAX_RETRIES = int(os.environ.get("SYNTH_MAX_RETRIES", "8"))


def _normalize_text(s: str) -> str:
    s = s.upper()
    s = re.sub(r"[^A-Z0-9ĂÂÎȘȚÄÖÜäöüß\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _token_set(s: str) -> set[str]:
    return {t for t in _normalize_text(s).split() if len(t) >= 2}


def _field_similarity(expected: str, ocr_text: str) -> float:
    try:
        from rapidfuzz import fuzz

        return fuzz.partial_ratio(_normalize_text(expected), _normalize_text(ocr_text)) / 100.0
    except ImportError:
        exp = _normalize_text(expected)
        if not exp:
            return 1.0
        return 1.0 if exp in _normalize_text(ocr_text) else 0.0


def _ocr_via_http(img_bgr: np.ndarray) -> str | None:
    url = os.environ.get("SYNTH_OCR_URL", "").rstrip("/")
    if not url:
        return None
    try:
        import requests

        ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return None
        headers: dict[str, str] = {}
        api_key = os.environ.get("SYNTH_OCR_API_KEY", "")
        if api_key:
            headers["X-OCR-API-Key"] = api_key
        resp = requests.post(
            f"{url}/v1/ocr",
            files={"file": ("sample.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
            headers=headers,
            timeout=30,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        return str(data.get("text", "") or "")
    except Exception:
        return None


def _detect_white_boxes(img_bgr: np.ndarray, cfg: TemplateConfig) -> bool:
    """True dacă există benzi albe suspecte în ROI-uri text."""
    h, w = img_bgr.shape[:2]
    for roi in cfg.fields.values():
        x1, y1, x2, y2 = roi.to_pixels(w, h)
        patch = img_bgr[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        bright = gray > 210
        if bright.mean() > 0.55:
            std = float(np.std(gray[bright])) if bright.any() else 999.0
            if std < 8.0:
                return True
    return False


def _sharpness_score(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def _ocr_similarity(record: SyntheticIdRecord, ocr_text: str) -> float:
    checks = [
        record.cnp,
        record.nume,
        record.prenume,
        record.serie + record.numar,
    ]
    scores = [_field_similarity(c, ocr_text) for c in checks if c]
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def _heuristic_ocr_from_card(card_bgr: np.ndarray, record: SyntheticIdRecord, cfg: TemplateConfig) -> float:
    """Fallback fără serviciu OCR: verifică absența benzilor albe + contrast în ROI."""
    if _detect_white_boxes(card_bgr, cfg):
        return 0.0
    h, w = card_bgr.shape[:2]
    contrasts: list[float] = []
    for key in ("cnp", "nume", "prenume"):
        roi = cfg.fields.get(key)
        if roi is None:
            continue
        x1, y1, x2, y2 = roi.to_pixels(w, h)
        patch = card_bgr[y1:y2, x1:x2]
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        contrasts.append(float(gray.std()))
    if not contrasts:
        return 0.5
    avg_contrast = sum(contrasts) / len(contrasts)
    if avg_contrast < 12:
        return 0.4
    return 0.97 if avg_contrast >= 18 else 0.88


def validate_sample(
    img_bgr: np.ndarray,
    card_bgr: np.ndarray,
    record: SyntheticIdRecord,
    cfg: TemplateConfig,
) -> tuple[bool, float, str]:
    """
    Returnează (passed, score, reason).
    Respinge dacă: benzi albe, prea neclar, sau similitudine OCR < prag.
    """
    if _detect_white_boxes(card_bgr, cfg):
        return False, 0.0, "white_box_on_card"

    sharp = _sharpness_score(img_bgr)
    if sharp < 35:
        return False, 0.0, "too_blurry"

    ocr_text = _ocr_via_http(img_bgr)
    if ocr_text:
        score = _ocr_similarity(record, ocr_text)
        if score < SIMILARITY_THRESHOLD:
            return False, score, "ocr_mismatch"
        return True, score, "ocr_ok"

    score = _heuristic_ocr_from_card(card_bgr, record, cfg)
    if score < SIMILARITY_THRESHOLD:
        return False, score, "heuristic_fail"
    return True, score, "heuristic_ok"
