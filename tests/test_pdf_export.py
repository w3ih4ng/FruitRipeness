from pathlib import Path
import tempfile
import unittest

from PIL import Image

from fruitripeness.pdf_export import (
    PREDICTION_PDF_FIELDS,
    export_report_pdf,
)


class PdfExportTests(unittest.TestCase):
    def test_export_writes_pdf_and_never_overwrites(self):
        rows = [
            {
                "source": "sample.png",
                "method": "hsv",
                "predicted_stage": "ripe",
                "score_unripe": 0.1,
                "score_ripe": 0.8,
                "score_overripe": 0.1,
                "processing_ms": 12.3,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "prediction.pdf"
            export_report_pdf(
                target,
                title="Prediction report",
                rows=rows,
                fields=PREDICTION_PDF_FIELDS,
                image=Image.new("RGB", (900, 420), "white"),
                protected=[],
                labels={"hsv": "Method 1 - HSV colour threshold"},
                notes=["Scores are uncalibrated."],
            )
            payload = target.read_bytes()
            self.assertTrue(payload.startswith(b"%PDF-"))
            self.assertGreater(len(payload), 1000)
            with self.assertRaisesRegex(ValueError, "already exists"):
                export_report_pdf(
                    target,
                    title="Prediction report",
                    rows=rows,
                    fields=PREDICTION_PDF_FIELDS,
                    protected=[],
                )

    def test_export_rejects_protected_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            protected = Path(tmp) / "input"
            protected.mkdir()
            with self.assertRaisesRegex(ValueError, "outside"):
                export_report_pdf(
                    protected / "report.pdf",
                    title="Protected",
                    rows=[{"source": "sample.png"}],
                    fields=[("source", "Image")],
                    protected=[protected],
                )


if __name__ == "__main__":
    unittest.main()
