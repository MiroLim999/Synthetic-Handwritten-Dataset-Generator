"""Packaging accepts the format IDs src.fields emits for non-date fields."""

import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
from src.generate_synthetic import generate, zip_dataset
from src.handoff_validation import BASE_NON_DATE_FORMAT_IDS


class HandoffFormatIdTests(unittest.TestCase):
    def test_dataset_with_base_non_date_test_rows_packages(self):
        with (tempfile.TemporaryDirectory() as temporary,
              mock.patch.object(config, "DATASETS_DIR", Path(temporary) / "datasets"),
              contextlib.redirect_stdout(io.StringIO())):
            out_dir = generate(
                30, dataset="format_ids", seed=3, show_bar=False, workers=1,
                custom_field_weights={"citizenship": 1, "full_name": 1, "age": 1})
            with (out_dir / "evaluation-annotations.csv").open(encoding="utf-8") as handle:
                format_ids = {row["format_id"] for row in csv.DictReader(handle)}
            self.assertIn("default", format_ids)
            self.assertTrue(zip_dataset(out_dir).is_file())

    def test_allowlist_covers_every_non_date_field(self):
        non_date = set(config.SUPPORTED_FIELD_TYPES) - {"date_numeric", "date_written"}
        self.assertEqual(set(BASE_NON_DATE_FORMAT_IDS), non_date)


if __name__ == "__main__":
    unittest.main()
