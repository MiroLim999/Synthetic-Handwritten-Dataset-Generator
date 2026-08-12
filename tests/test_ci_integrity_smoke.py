import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from src.dataset_validation import MONTAGE_FILENAME, REPORT_FILENAME, validate_dataset
from src.manifest import MANIFEST_COLUMNS, MANIFEST_SCHEMA_VERSION


class CiIntegritySmokeTests(unittest.TestCase):
    def test_complete_dataset_passes_and_writes_bound_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "dataset_ci"
            rows = []
            for index, split in enumerate(("train", "val", "test"), start=1):
                split_dir = dataset / split
                split_dir.mkdir(parents=True)
                filename = f"syn_{index:06d}.png"
                image = Image.new("RGB", (96, 32), "white")
                ImageDraw.Draw(image).rectangle((12, 10, 82, 21), fill="black")
                image.save(split_dir / filename)
                rows.append(
                    {
                        "filename": filename,
                        "label": f"sample {index}",
                        "split": split,
                        "source": "synthetic",
                        "field_type": "full_name",
                        "font": "CiFont.ttf",
                        "sample_mode": "regular",
                        "writer_id": "",
                        "schema_version": MANIFEST_SCHEMA_VERSION,
                    }
                )

            with (dataset / "manifest.csv").open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=MANIFEST_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)

            report = validate_dataset(dataset, expected_count=3)

            self.assertTrue(report.valid, report.errors)
            self.assertEqual(report.statistics["manifest_rows"], 3)
            self.assertEqual(report.statistics["images"], 3)
            self.assertTrue(report.manifest_sha256)
            self.assertTrue(report.images_sha256)
            self.assertTrue((dataset / MONTAGE_FILENAME).is_file())
            persisted = json.loads((dataset / REPORT_FILENAME).read_text("utf-8"))
            self.assertEqual(persisted["manifest_sha256"], report.manifest_sha256)
            self.assertEqual(persisted["images_sha256"], report.images_sha256)


if __name__ == "__main__":
    unittest.main()
