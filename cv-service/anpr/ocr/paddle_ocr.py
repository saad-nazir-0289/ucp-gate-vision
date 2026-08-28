"""PaddleOCR-based plate reader (primary PlateOCR implementation)."""
from __future__ import annotations

import logging
import re

import numpy as np

from ..interfaces import PlateOCR, PlateReading

logger = logging.getLogger(__name__)

_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]")

# The plate-format filter. This carries far more weight than its size
# suggests: it is the pipeline's PRIMARY OCR filter, and the only signal
# separating a plate number from the province band OCR glues onto it
# (see PaddleOCRPlateReader.read_candidates). It also rejects non-plate
# signage ("ENTRANCE" — pure letters) and truncated reads ("545" — pure
# digits) on shape alone.
#
# WIDENED from ^[A-Z]{2,4}[0-9]{2,4}$. All 33 plates in
# sample_data/ground_truth.csv fit that narrower shape, but 33 plates from
# two gates is a small sample to hard-code a silent discard rule against:
# anything it doesn't match is dropped with no event and no review flag.
# Older/other local formats, and plates carrying a city or series prefix,
# use separators and different group lengths. The pattern below still
# demands letters-then-digits (which is what does the actual work) but
# stops assuming the exact group counts.
#
# An optional trailing-letter suffix was tried here and REVERTED: it let
# 'ARK2363N' back through — the real ['ARK2363', 'n'] noise-fragment case
# from this project's footage. Candidate ordering would still have picked
# the right one (a joined candidate can never outscore its own best line),
# but there is no evidence local plates carry a trailing letter, so there
# is no reason to spend the safety margin on it.
#
# Configurable per deployment: pass `plate_pattern=` to PipelineRunner, or
# --plate-pattern on the CLI. Set it to None to disable format filtering
# entirely — but read read_candidates() first, because without this the
# province band comes back as part of the plate text.
DEFAULT_PLATE_PATTERN = re.compile(r"^[A-Z]{1,4}[0-9]{1,5}$")


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
        """Returns (cleaned_text, confidence) for the single best candidate —
        "" only when PaddleOCR found no text at all. Never applies a
        confidence or format threshold; see the module docstring for where
        that belongs.

        Prefer `read_candidates()` when the caller has a plate-format
        policy to apply (PipelineRunner does): this method has to collapse
        a multi-line crop down to one answer before the caller can weigh
        in, and the highest-confidence line is not always the plate.
        """
        candidates = self.read_candidates(plate_crop)
        if not candidates:
            return "", 0.0
        best = candidates[0]
        return best.text, best.confidence

    def read_candidates(self, plate_crop: np.ndarray) -> list[PlateReading]:
        """Every plausible reading of this crop, best-confidence first.

        CRITICAL FIX (confirmed against real evidence crops from this
        project's own footage, not theory). The previous implementation did
        `"".join(texts)` over every text line PaddleOCR returned and took
        `min(scores)` as the confidence. Both are wrong for local plates,
        and each one alone was enough to destroy a perfect read:

            rec_texts  = ['BUV 711', 'JUNJAU']
            rec_scores = [0.9988,    0.6661]

        'JUNJAU' is the "PUNJAB" province band across the top of the plate,
        misread. The plate number itself was read perfectly at 0.9988. But
        joining produced 'BUV711JUNJAU' (rejected by the caller's plate-
        format regex) with confidence 0.6661 (rejected by the caller's
        confidence floor). A correct, high-confidence read was thrown away
        twice over. Same story for a stray noise fragment:
        ['ARK2363', 'n'] with scores [0.9909, 0.1747] — a 0.17 speck of
        noise dragged a 0.99 plate read below every threshold.

        So instead of guessing which lines belong to the plate, this
        enumerates the possibilities and lets the caller's format policy
        decide:

        - each individual line on its own, and
        - each contiguous run of 2+ lines joined in VISUAL order (top-to-
          bottom, then left-to-right), for genuinely stacked two-line
          plates — e.g. ['GAA', '545'] must join to 'GAA545', while
          ['BUV 711', 'JUNJAU'] must not join at all.

        Visual ordering matters and is not free: PaddleOCR's returned order
        does not track layout. In the 'BUV 711'/'JUNJAU' crop above the
        province band sits ABOVE the number (y=5 vs y=9) yet came back
        second, and in another crop of the same plate it came back first.
        Joining in returned order would produce '711BUVJUNJAU'-style
        garbage on exactly the plates this is meant to rescue.

        A multi-line candidate's confidence is the min over its own lines
        (all of them have to be right for the joined text to be right) —
        which is what `min(scores)` was reaching for, just applied to the
        lines actually being used rather than to every line in the crop.
        """
        if plate_crop is None or plate_crop.size == 0:
            return []

        try:
            results = self._ocr.predict(plate_crop)
        except Exception:
            logger.exception("PaddleOCR inference failed on a plate crop")
            return []

        lines = self._extract_lines(results)
        if not lines:
            return []

        candidates: list[PlateReading] = []
        seen: set[str] = set()
        for start in range(len(lines)):
            for end in range(start + 1, len(lines) + 1):
                run = lines[start:end]
                text = _NON_ALNUM_RE.sub("", "".join(t for t, _ in run).upper())
                if not text or text in seen:
                    continue
                seen.add(text)
                candidates.append(
                    PlateReading(
                        text=text,
                        confidence=float(min(s for _, s in run)),
                        source_lines=len(run),
                    )
                )

        # Best confidence first; on a tie prefer the candidate built from
        # fewer lines (less chance of having glued on a province band).
        candidates.sort(key=lambda c: (-c.confidence, c.source_lines))
        return candidates

    @classmethod
    def _extract_lines(cls, results) -> list[tuple[str, float]]:
        """(text, score) per OCR line, sorted top-to-bottom then left-to-right.

        Sorting is by the line's own polygon, so a two-line plate joins in
        reading order rather than in whatever order PaddleOCR happened to
        return (see read_candidates — it varies between crops of the same
        plate). If polygons are unavailable the returned order is kept,
        which is still correct for the common single-line case.
        """
        texts, scores, polys = cls._extract_texts_scores_polys(results)
        if not texts:
            return []

        n = min(len(texts), len(scores))
        lines = [(str(texts[i]), float(scores[i])) for i in range(n)]

        if polys is not None and len(polys) >= n:
            def _top_left(i: int) -> tuple[float, float]:
                try:
                    pts = np.asarray(polys[i], dtype=float).reshape(-1, 2)
                    return (float(pts[:, 1].min()), float(pts[:, 0].min()))  # (y, x)
                except Exception:
                    return (float(i), 0.0)  # unparseable polygon — fall back to returned order

            order = sorted(range(n), key=_top_left)
            lines = [lines[i] for i in order]
        return lines

    @staticmethod
    def _extract_texts_scores_polys(results) -> tuple[list[str], list[float], list | None]:
        if not results:
            return [], [], None
        res = results[0]

        def _poly(container):
            for key in ("rec_polys", "dt_polys", "rec_boxes"):
                try:
                    value = container[key]
                except (KeyError, TypeError, IndexError):
                    value = getattr(container, key, None)
                if value is not None and len(value):
                    return list(value)
            return None

        accessors = (
            lambda r: (r["res"]["rec_texts"], r["res"]["rec_scores"], _poly(r["res"])),
            lambda r: (r["rec_texts"], r["rec_scores"], _poly(r)),
            lambda r: (r.rec_texts, r.rec_scores, _poly(r)),
        )
        for accessor in accessors:
            try:
                texts, scores, polys = accessor(res)
                return list(texts), list(scores), polys
            except (KeyError, TypeError, AttributeError):
                continue
        logger.error(
            "Could not extract rec_texts/rec_scores from a PaddleOCR result (unrecognized shape: %s). "
            "PaddleOCR's result schema may have changed — check the pinned version in "
            "requirements.txt against the current PaddleOCR quickstart docs.",
            type(res),
        )
        return [], [], None
