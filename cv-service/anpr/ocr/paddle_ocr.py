"""PaddleOCR-based plate reader (primary PlateOCR implementation)."""
from __future__ import annotations

import logging
import re

import numpy as np

from ..interfaces import PlateOCR

logger = logging.getLogger(__name__)

_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]")

# Every genuine plate read confirmed on real footage normalizes to letters
# followed by digits, nothing else (e.g. AAL988, BAG9976, BUV711, ARK2363).
# Non-plate text the detector occasionally boxes (signage, "#11 car 0.39"
# style overlay artifacts) doesn't fit this shape — e.g. "ENTRANCE" is pure
# letters, no digits at all. This is a content-based filter *on top of* the
# detector's geometric filters (edge/size/aspect-ratio in
# anpr/detectors/plate_detector.py), not a replacement for them — catches
# false positives whose box geometry looks plate-like but whose text doesn't.
# Tune/replace for plate formats outside 2-4 letters + 2-4 digits. Lives
# here (not in PlateOCR) since it's a acceptance decision, applied by
# whoever calls read() — see PipelineRunner.
DEFAULT_PLATE_PATTERN = re.compile(r"^[A-Z]{2,4}[0-9]{2,4}$")


class PaddleOCRPlateReader(PlateOCR):
    """Reads a plate string + confidence from a cropped plate image.

    Only light, non-judgmental cleanup is applied (uppercase, strip
    non-alphanumeric OCR noise) — this always returns PaddleOCR's raw
    recognized text and confidence, never an accept/reject decision.

    Design note (external review — fixed): an earlier version accepted a
    `min_confidence_override` kwarg not part of the PlateOCR ABC, with
    callers catching TypeError to detect whether a given implementation
    supported it. That's backwards: OCR's job is "what text is here, how
    confident are you," full stop — deciding whether that's good enough is
    contextual (which vehicle class, which threshold sweep is being tested)
    and belongs in the caller (PipelineRunner), not baked into this class.
    Moving it out also makes offline threshold sweeping trivial: change the
    runner's threshold, no need to reconstruct this class. Confidence
    filtering and DEFAULT_PLATE_PATTERN format-checking both now happen in
    PipelineRunner._process_frame.

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
        use_textline_orientation: bool = True,
        device: str | None = None,
    ):
        from paddleocr import PaddleOCR

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
        """Returns (cleaned_text, confidence) — cleaned_text is "" only when
        PaddleOCR found no text at all. Never applies a confidence or
        format threshold; see the module docstring for where that belongs.
        """
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
