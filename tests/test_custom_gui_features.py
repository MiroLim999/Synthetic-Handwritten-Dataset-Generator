"""Tests for new GUI features: custom field weights, custom split fractions, presets, and preflight estimates."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
from src.generate_synthetic import generate
from gui import PRESET_DIR


class CustomGuiFeaturesTests(unittest.TestCase):
    def test_generate_with_custom_field_weights_and_split_fractions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "datasets"
            custom_weights = {"date_written": 10, "full_name": 5}
            custom_fractions = (0.70, 0.15, 0.15)

            with mock.patch.object(config, "DATASETS_DIR", root):
                out_dir = generate(
                    count=20,
                    dataset="custom_run",
                    seed=42,
                    custom_field_weights=custom_weights,
                    custom_split_fractions=custom_fractions,
                    show_bar=False,
                )

            manifest_path = out_dir / "manifest.csv"
            self.assertTrue(manifest_path.is_file())

            with manifest_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 20)

            # Confirm only requested field types were generated
            field_types = {row["field_type"] for row in rows}
            self.assertTrue(field_types.issubset({"date_written", "full_name"}))

            # Confirm custom split allocation
            split_counts = {"train": 0, "val": 0, "test": 0}
            for row in rows:
                split_counts[row["split"]] += 1
            self.assertEqual(split_counts, {"train": 14, "val": 3, "test": 3})

    def test_default_presets_exist_and_are_valid_json(self):
        self.assertTrue(PRESET_DIR.is_dir())
        json_files = list(PRESET_DIR.glob("*.json"))
        self.assertGreaterEqual(len(json_files), 3)

        for preset_file in json_files:
            data = json.loads(preset_file.read_text(encoding="utf-8"))
            self.assertIn("name", data)
            self.assertIn("fade", data)
            self.assertIn("noise", data)

    def test_short_label_degradation_safeguard_never_crashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "datasets"
            with mock.patch.object(config, "DATASETS_DIR", root):
                out_dir = generate(
                    count=10,
                    dataset="short_label_run",
                    seed=43,
                    sample_mode="semi_broken_mixed",
                    font_style="cursive",
                    cursive_group="Loose informal",
                    custom_field_weights={"age": 10},
                    semi_broken_params={"erode_prob": 1.0, "gap_prob": 1.0, "gap_count": 30},
                    show_bar=False,
                )
            self.assertTrue((out_dir / "manifest.csv").is_file())


if __name__ == "__main__":
    unittest.main()
