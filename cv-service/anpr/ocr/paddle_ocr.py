"""PaddleOCR-based plate reader (primary PlateOCR implementation)."""
from __future__ import annotations

import logging
import re

import numpy as np

from ..interfaces import PlateOCR

logger = logging.getLogger(__name__)

_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]")


class PaddleOCRPlateReader(PlateOCR):
    """Reads a plate string + confidence from a cropped plate image.

    Only light cleanup is applied (uppercase, strip non-alphanumeric OCR
    noise) — this is deliberately NOT full plate normalization/matching.
    Per docs/DATABASE_SCHEMA.md, fuzzy matching against
    `plate_number_matched` happens in the backend at write time (Phase 3);
    this class only emits the "as read" string (FR-3.1's `plate_number_raw`).

    Known risk: PaddleOCR's Python API changed materially between the 2.x
    line (`PaddleOCR(use_angle_cls=..., use_gpu=...)` + `.ocr(img, cls=True)`)
    and the 3.x line pinned in requirements.txt
    (`PaddleOCR(use_doc_orientation_classify=..., ...)` + `.predict(img)`
    returning pipeline-style results). This wrapper targets 3.x and extracts
    results defensively across a couple of known result shapes so a minor
    point-release schema change fails with a clear log message instead of
    silently returning garbage — if it starts erroring after an upgrade,
    check the current PaddleOCR quickstart docs for the result schema.
    """

    def __init__(
        self,
        lang: str = "en",
        min_confidence: float = 0.30,
        use_textline_orientation: bool = True,
        device: str | None = None,
    ):
        from paddleocr import PaddleOCR

        self.min_confidence = min_confidence

        kwargs = dict(
            lang=lang,
            use_doc_orientation_classify=False,  # full-document preprocessing — irrelevant for small plate crops
            use_doc_unwarping=False,
            use_textline_orientation=use_textline_orientation,
        )
        if device:
            kwargs["device"] = device

        try:
            self._ocr = PaddleOCR(**kwargs)
        except TypeError:
            logger.warning(
                "PaddleOCR() rejected one of kwargs=%s (API mismatch against the pinned version?); "
                "retrying with just `lang`.",
                list(kwargs),
            )
            self._ocr = PaddleOCR(lang=lang)

    def read(self, plate_crop: np.ndarray) -> tuple[str, float]:
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0

        try:
            results = self._ocr.predict(plate_crop)
        except Exception:
            logger.exception("PaddleOCR inference failed on a plate crop")
            return "", 0.0

        texts, scores = self._extract_texts_and_scores(results)
        if not texts:
            return "", 0.0

        # Plates are effectively single-line; concatenate in returned order
        # in case the model split it into multiple text-line detections.
        raw_text = "".join(texts)
        confidence = float(min(scores)) if scores else 0.0  # weakest line caps overall confidence
        cleaned = _NON_ALNUM_RE.sub("", raw_text.upper())

        if not cleaned or confidence < self.min_confidence:
            return "", confidence
        return cleaned, confidence

    @staticmethod
    def _extract_texts_and_scores(results) -> tuple[list[str], list[float]]:
        if not results:
            return [], []
        res = results[0]
        accessors = (
            lambda r: (r["res"]["rec_texts"], r["res"]["rec_scores"]),
            lambda r: (r["rec_texts"], r["rec_scores"]),
            lambda r: (r.rec_texts, r.rec_scores),
        )
        for accessor in accessors:
            try:
                texts, scores = accessor(res)
                return list(texts), list(scores)
            except (KeyError, TypeError, AttributeError):
                continue
        logger.error(
            "Could not extract rec_texts/rec_scores from a PaddleOCR result (unrecognized shape: %s). "
            "PaddleOCR's result schema may have changed — check the pinned version in "
            "requirements.txt against the current PaddleOCR quickstart docs.",
            type(res),
        )
        return [], []
