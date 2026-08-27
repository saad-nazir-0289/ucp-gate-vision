"""Tests for PaddleOCRPlateReader's candidate generation.

No PaddleOCR model is loaded: the reader is built with `__new__` and given
a stub `_ocr` whose `predict()` returns result dicts in the real 3.x schema.
Every fixture below is verbatim real output captured from this project's own
evidence crops (cv-service/output/*_evidence*/), not invented — see
PaddleOCRPlateReader.read_candidates for the capture.
"""
import unittest

import numpy as np

from anpr.ocr.paddle_ocr import DEFAULT_PLATE_PATTERN, PaddleOCRPlateReader


def _poly(x1: int, y1: int, x2: int, y2: int):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int16)


class _StubPaddle:
    def __init__(self, texts, scores, polys=None):
        self._result = {"rec_texts": texts, "rec_scores": scores}
        if polys is not None:
            self._result["rec_polys"] = polys

    def predict(self, _img):
        return [self._result]


def _reader(texts, scores, polys=None) -> PaddleOCRPlateReader:
    reader = PaddleOCRPlateReader.__new__(PaddleOCRPlateReader)
    reader._ocr = _StubPaddle(texts, scores, polys)
    return reader


def _crop() -> np.ndarray:
    return np.zeros((30, 60, 3), dtype=np.uint8)


class TestCandidateGeneration(unittest.TestCase):
    def test_province_band_yields_the_bare_plate_as_a_candidate(self):
        """Real crop: the PUNJAB band misread as 'JUNJAU' alongside a
        perfect 0.9988 read of the number."""
        reader = _reader(
            ["BUV 711", "JUNJAU"], [0.9988, 0.6661],
            [_poly(4, 9, 55, 39), _poly(30, 5, 52, 14)],
        )

        candidates = reader.read_candidates(_crop())
        texts = [c.text for c in candidates]

        self.assertIn("BUV711", texts)
        well_formed = [c for c in candidates if DEFAULT_PLATE_PATTERN.match(c.text)]
        self.assertEqual([c.text for c in well_formed], ["BUV711"])
        self.assertAlmostEqual(well_formed[0].confidence, 0.9988, places=4)

    def test_two_line_plate_produces_the_joined_candidate(self):
        """Real crop: ['GAA', '545'] — neither line is a plate alone."""
        reader = _reader(
            ["GAA", "545"], [0.9945, 0.9825],
            [_poly(5, 4, 40, 20), _poly(8, 22, 38, 36)],
        )

        candidates = reader.read_candidates(_crop())
        well_formed = [c for c in candidates if DEFAULT_PLATE_PATTERN.match(c.text)]

        self.assertEqual([c.text for c in well_formed], ["GAA545"])
        self.assertAlmostEqual(well_formed[0].confidence, 0.9825, places=4)
        self.assertEqual(well_formed[0].source_lines, 2)

    def test_joined_candidate_uses_visual_order_not_returned_order(self):
        """PaddleOCR's returned order does not track layout — confirmed on
        two crops of the same plate, where the province band came back
        first in one and second in the other. Joining in returned order
        would produce '545GAA' here."""
        reader = _reader(
            ["545", "GAA"], [0.9825, 0.9945],
            [_poly(8, 22, 38, 36), _poly(5, 4, 40, 20)],  # '545' is the LOWER line
        )

        candidates = reader.read_candidates(_crop())
        texts = [c.text for c in candidates]

        self.assertIn("GAA545", texts)
        self.assertNotIn("545GAA", texts)

    def test_noise_fragment_does_not_cap_the_plate_confidence(self):
        """Real crop: ['ARK2363', 'n'] scores [0.9909, 0.1747]. Under the
        old min(scores) the plate arrived at 0.1747."""
        reader = _reader(
            ["ARK2363", "n"], [0.9909, 0.1747],
            [_poly(3, 6, 78, 30), _poly(60, 32, 66, 40)],
        )

        candidates = reader.read_candidates(_crop())
        by_text = {c.text: c.confidence for c in candidates}

        self.assertAlmostEqual(by_text["ARK2363"], 0.9909, places=4)
        self.assertAlmostEqual(by_text["ARK2363N"], 0.1747, places=4)

    def test_single_line_crop_is_unchanged(self):
        reader = _reader(["BUV711"], [0.9998], [_poly(2, 3, 50, 26)])

        candidates = reader.read_candidates(_crop())

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].text, "BUV711")
        self.assertEqual(candidates[0].source_lines, 1)

    def test_candidates_are_sorted_best_confidence_first(self):
        reader = _reader(
            ["JUNJAU", "BUV 711"], [0.6661, 0.9988],
            [_poly(30, 5, 52, 14), _poly(4, 15, 55, 39)],
        )

        candidates = reader.read_candidates(_crop())

        self.assertEqual(candidates[0].text, "BUV711")
        self.assertEqual([c.confidence for c in candidates], sorted((c.confidence for c in candidates), reverse=True))

    def test_read_returns_the_best_candidate(self):
        reader = _reader(["BUV 711", "JUNJAU"], [0.9988, 0.6661], [_poly(4, 9, 55, 39), _poly(30, 5, 52, 14)])

        text, confidence = reader.read(_crop())

        self.assertEqual(text, "BUV711")
        self.assertAlmostEqual(confidence, 0.9988, places=4)

    def test_missing_polygons_falls_back_to_returned_order(self):
        """A future PaddleOCR result shape without polygons must still
        work for the common single-line case rather than crashing."""
        reader = _reader(["GAA", "545"], [0.9945, 0.9825])

        candidates = reader.read_candidates(_crop())

        self.assertIn("GAA545", [c.text for c in candidates])

    def test_empty_crop_returns_no_candidates(self):
        reader = _reader([], [])

        self.assertEqual(reader.read_candidates(_crop()), [])
        self.assertEqual(reader.read_candidates(np.zeros((0, 0, 3), dtype=np.uint8)), [])


if __name__ == "__main__":
    unittest.main()
