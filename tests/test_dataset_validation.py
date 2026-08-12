import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from src.dataset_validation import (
    MANIFEST_COLUMNS,
    MANIFEST_SCHEMA_VERSION,
    MONTAGE_FILENAME,
    REPORT_FILENAME,
    validate_dataset,
)


def _write_ink_image(path: Path, text_line: bool = True):
    image = Image.new("RGB", (100, 40), "white")
    if text_line:
        ImageDraw.Draw(image).rectangle((15, 15, 85, 24), fill="black")
    image.save(path)


class DatasetValidationTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name) / "dataset_001"
        for split in ("train", "val", "test"):
            (self.root / split).mkdir(parents=True)
        self.rows = []
        for index, split in enumerate(("train", "val", "test"), start=1):
            filename = f"syn_{index:06d}.png"
            _write_ink_image(self.root / split / filename)
            self.rows.append(
                {
                    "filename": filename,
                    "label": f"Label {index}",
                    "split": split,
                    "source": "synthetic",
                    "field_type": "full_name",
                    "font": "Test.ttf",
                    "sample_mode": "regular",
                    "writer_id": "",
                    "schema_version": str(MANIFEST_SCHEMA_VERSION),
                }
            )
        self._write_manifest()

    def tearDown(self):
        self._temporary.cleanup()

    def _write_manifest(self):
        with (self.root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(self.rows)

    def test_valid_dataset_writes_report_and_montage(self):
        report = validate_dataset(self.root, expected_count=3)
        self.assertTrue(report.valid, report.errors)
        self.assertTrue((self.root / REPORT_FILENAME).is_file())
        self.assertTrue((self.root / MONTAGE_FILENAME).is_file())
        saved = json.loads((self.root / REPORT_FILENAME).read_text(encoding="utf-8"))
        self.assertTrue(saved["valid"])
        self.assertEqual(saved["statistics"]["manifest_rows"], 3)
        self.assertEqual(len(saved["images_sha256"]), 64)

    def test_missing_or_orphan_image_fails(self):
        (self.root / "val" / "syn_000002.png").unlink()
        _write_ink_image(self.root / "test" / "orphan.png")
        report = validate_dataset(self.root, write_artifacts=False)
        self.assertFalse(report.valid)
        self.assertTrue(any("missing" in error.lower() for error in report.errors))
        self.assertTrue(any("orphan" in error.lower() for error in report.errors))

    def test_blank_image_and_empty_split_fail(self):
        _write_ink_image(self.root / "test" / "syn_000003.png", text_line=False)
        self.rows = [row for row in self.rows if row["split"] != "val"]
        self._write_manifest()
        (self.root / "val" / "syn_000002.png").unlink()
        report = validate_dataset(self.root, write_artifacts=False)
        self.assertFalse(report.valid)
        self.assertTrue(any("blank" in error.lower() for error in report.errors))
        self.assertTrue(any("split is empty: val" in error.lower() for error in report.errors))

    def test_unsupported_image_mode_fails_with_mode_name(self):
        palette = Image.new("P", (100, 40), color=255)
        palette.putpixel((50, 20), 0)
        palette.save(self.root / "train" / "syn_000001.png")

        report = validate_dataset(self.root, write_artifacts=False)

        self.assertFalse(report.valid)
        self.assertTrue(any("unsupported image mode 'P'" in error for error in report.errors))

    def test_real_row_requires_writer_and_schema(self):
        self.rows[0]["source"] = "real"
        self.rows[0]["schema_version"] = "99"
        self._write_manifest()
        report = validate_dataset(self.root, write_artifacts=False)
        self.assertFalse(report.valid)
        self.assertTrue(any("writer_id" in error for error in report.errors))
        self.assertTrue(any("schema version" in error for error in report.errors))

    def test_synthetic_rows_require_empty_writer_and_generation_metadata(self):
        self.rows[0]["writer_id"] = "synthetic-writer-001"
        self.rows[0]["field_type"] = ""
        self.rows[0]["font"] = " "
        self.rows[0]["sample_mode"] = ""
        self._write_manifest()

        report = validate_dataset(self.root, write_artifacts=False)

        self.assertFalse(report.valid)
        errors = "\n".join(report.errors)
        self.assertIn("synthetic sample must have empty writer_id", errors)
        self.assertIn("synthetic sample requires non-empty", errors)
        self.assertIn("field_type", errors)
        self.assertIn("font", errors)
        self.assertIn("sample_mode", errors)

    def test_real_writer_cannot_cross_splits(self):
        for row in self.rows[:2]:
            row["source"] = "real"
            row["writer_id"] = "writer-001"
        self._write_manifest()

        report = validate_dataset(self.root, write_artifacts=False)

        self.assertFalse(report.valid)
        self.assertTrue(
            any(
                "writer-001" in error
                and "multiple splits" in error
                and "train" in error
                and "val" in error
                for error in report.errors
            )
        )

    def test_manifest_header_must_match_exact_columns_and_order(self):
        reversed_columns = tuple(reversed(MANIFEST_COLUMNS))
        with (self.root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=reversed_columns)
            writer.writeheader()
            writer.writerows(self.rows)

        report = validate_dataset(self.root, write_artifacts=False)

        self.assertFalse(report.valid)
        self.assertTrue(
            any(
                "columns must be exactly" in error and "in this order" in error
                for error in report.errors
            )
        )

    def test_ragged_rows_return_actionable_errors_instead_of_crashing(self):
        with (self.root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(MANIFEST_COLUMNS)
            writer.writerow([self.rows[0][name] for name in MANIFEST_COLUMNS[:-1]])
            writer.writerow([self.rows[1][name] for name in MANIFEST_COLUMNS] + ["extra"])
            writer.writerow([self.rows[2][name] for name in MANIFEST_COLUMNS])

        report = validate_dataset(self.root, write_artifacts=False)

        self.assertFalse(report.valid)
        errors = "\n".join(report.errors)
        self.assertIn("malformed CSV row has 8 columns; expected exactly 9", errors)
        self.assertIn("malformed CSV row has 10 columns; expected exactly 9", errors)
        self.assertIn("missing or extra commas/quotes", errors)

    def test_broken_csv_quoting_returns_an_actionable_error(self):
        with (self.root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            handle.write(",".join(MANIFEST_COLUMNS) + "\n")
            handle.write('"unterminated')

        report = validate_dataset(self.root, write_artifacts=False)

        self.assertFalse(report.valid)
        self.assertTrue(
            any(
                "Could not parse manifest.csv" in error and "quoting and delimiters" in error
                for error in report.errors
            )
        )

    def test_conflicting_duplicate_manifest_rows_are_identified(self):
        duplicate = dict(self.rows[0])
        duplicate["label"] = "Conflicting label"
        self.rows.append(duplicate)
        self._write_manifest()

        report = validate_dataset(self.root, write_artifacts=False)

        self.assertFalse(report.valid)
        self.assertTrue(
            any(
                "conflicting duplicate manifest rows" in error
                and "differing columns: label" in error
                for error in report.errors
            )
        )

    def test_duplicate_filename_across_splits_fails(self):
        old = self.root / "val" / self.rows[1]["filename"]
        self.rows[1]["filename"] = self.rows[0]["filename"]
        old.rename(self.root / "val" / self.rows[1]["filename"])
        self._write_manifest()
        report = validate_dataset(self.root, write_artifacts=False)
        self.assertFalse(report.valid)
        self.assertTrue(any("duplicate filename" in error.lower() for error in report.errors))


if __name__ == "__main__":
    unittest.main()
