"""Security and all-or-nothing preflight tests for real-data merging."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
from PIL import Image, ImageDraw
from src import build_splits
from src.manifest import MANIFEST_COLUMNS, MANIFEST_SCHEMA_VERSION, manifest_bytes


class BuildSplitsSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.real_dir = self.root / "real"
        self.real_images = self.real_dir / "images"
        self.real_images.mkdir(parents=True)

        self.datasets_dir = self.root / "datasets"
        self.out_dir = self.datasets_dir / "dataset_001"
        for split in ("train", "val", "test"):
            (self.out_dir / split).mkdir(parents=True)

        # Existing synthetic data and metadata files make it possible to
        # prove rejected input leaves every destination byte unchanged.
        synthetic_rows = []
        for index, split in enumerate(("train", "val", "test"), start=1):
            filename = f"syn_{index:06d}.png"
            self._write_ink_image(self.out_dir / split / filename, shade=index)
            synthetic_rows.append({
                "filename": filename,
                "label": f"Existing {index}",
                "split": split,
                "source": "synthetic",
                "field_type": "name",
                "font": "Test.ttf",
                "sample_mode": "regular",
                "writer_id": "",
                "schema_version": MANIFEST_SCHEMA_VERSION,
            })
        (self.out_dir / "manifest.csv").write_bytes(manifest_bytes(synthetic_rows))
        with (self.out_dir / "labels.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(("filename", "label", "split"))
            writer.writerows(
                (row["filename"], row["label"], row["split"])
                for row in synthetic_rows
            )

        patches = (
            mock.patch.object(config, "REAL_DIR", self.real_dir),
            mock.patch.object(config, "DATASETS_DIR", self.datasets_dir),
            mock.patch.object(config, "SPLIT_NAMES", ("train", "val", "test")),
            mock.patch.object(config, "REAL_TRAIN_FRAC", 0.6),
            mock.patch.object(config, "REAL_VAL_FRAC", 0.2),
            mock.patch.object(config, "RANDOM_SEED", 42),
            mock.patch.object(
                config, "resolve_dataset_dir", side_effect=lambda _name: self.out_dir
            ),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _write_labels(self, rows: list[tuple[str, str, str]]) -> None:
        with (self.real_dir / "labels.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["filename", "label", "writer_id"])
            writer.writerows(rows)

    @staticmethod
    def _write_ink_image(path: Path, *, shade: int = 0) -> None:
        image = Image.new("RGB", (96, 36), "white")
        ImageDraw.Draw(image).rectangle((12, 12, 84, 23), fill=(shade,) * 3)
        image.save(path, "PNG")

    def _add_source(self, filename: str, content: bytes = b"real-image") -> None:
        self._write_ink_image(self.real_images / filename, shade=sum(content) % 100)

    def _dataset_snapshot(self) -> tuple[tuple[str, str, bytes | None], ...]:
        snapshot: list[tuple[str, str, bytes | None]] = []
        for path in sorted(self.out_dir.rglob("*"), key=lambda item: str(item)):
            relative = path.relative_to(self.out_dir).as_posix()
            if path.is_dir():
                snapshot.append((relative, "directory", None))
            else:
                snapshot.append((relative, "file", path.read_bytes()))
        return tuple(snapshot)

    def test_valid_explicit_writer_csv_is_merged(self) -> None:
        rows = [
            ("P001_name.png", "Maria Santos", "P001"),
            ("P002_date.png", "January 2 1945", "P002"),
            ("P003_place.png", "Manila", "P003"),
            ("P004_age.png", "Thirty Two", "P004"),
            ("P005_status.png", "Single", "P005"),
        ]
        for index, (filename, _label, _writer_id) in enumerate(rows):
            self._add_source(filename, f"image-{index}".encode("ascii"))
        self._write_labels(rows)

        result = build_splits.build("dataset_001")

        self.assertEqual(result.out_dir, self.out_dir)
        self.assertEqual(result.copied, len(rows))
        with (self.out_dir / "manifest.csv").open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            self.assertEqual(tuple(reader.fieldnames or ()), MANIFEST_COLUMNS)
            manifest = [row for row in reader if row["source"] == "real"]
        self.assertEqual(len(manifest), len(rows))
        self.assertEqual({row["filename"] for row in manifest}, {row[0] for row in rows})
        for row in manifest:
            copied = self.out_dir / row["split"] / row["filename"]
            self.assertTrue(copied.is_file())
            self.assertEqual(
                copied.read_bytes(), (self.real_images / row["filename"]).read_bytes()
            )

        with (self.out_dir / "labels.csv").open(newline="", encoding="utf-8") as file:
            labels = list(csv.DictReader(file))
        self.assertEqual(len(labels), len(rows) + 3)  # existing synthetic + real

    def test_unsafe_filenames_are_rejected_without_destination_changes(self) -> None:
        unsafe_names = (
            ".",
            "..",
            "../outside.png",
            "..\\outside.png",
            "nested/image.png",
            "nested\\image.png",
            "/tmp/outside.png",
            "C:\\Temp\\outside.png",
            "C:drive-relative.png",
            "\\\\server\\share\\outside.png",
        )
        before = self._dataset_snapshot()
        for filename in unsafe_names:
            with self.subTest(filename=filename):
                self._write_labels([(filename, "malicious", "P999")])
                with self.assertRaises(ValueError):
                    build_splits.build("dataset_001")
                self.assertEqual(self._dataset_snapshot(), before)

    def test_duplicate_and_conflicting_filenames_abort_before_copying(self) -> None:
        self._add_source("P001_name.png")
        before = self._dataset_snapshot()

        cases = (
            (
                [
                    ("P001_name.png", "Same", "P001"),
                    ("P001_name.png", "Same", "P001"),
                ],
                "Duplicate real-data filename",
            ),
            (
                [
                    ("P001_name.png", "First", "P001"),
                    ("P001_name.png", "Second", "P001"),
                ],
                "Conflicting labels",
            ),
            (
                [
                    ("P001_name.png", "Same", "P001"),
                    ("p001_NAME.PNG", "Same", "P001"),
                ],
                "Duplicate real-data filename",
            ),
        )
        for rows, message in cases:
            with self.subTest(rows=rows):
                self._write_labels(rows)
                with self.assertRaisesRegex(ValueError, message):
                    build_splits.build("dataset_001")
                self.assertEqual(self._dataset_snapshot(), before)

    def test_late_missing_source_cannot_leave_an_earlier_copy(self) -> None:
        self._add_source("P001_valid.png", b"would-have-been-copied")
        self._write_labels(
            [
                ("P001_valid.png", "Valid", "P001"),
                ("P002_missing.png", "Missing", "P002"),
                ("P003_valid.png", "Also Valid", "P003"),
            ]
        )
        self._add_source("P003_valid.png")
        before = self._dataset_snapshot()

        with self.assertRaises(FileNotFoundError):
            build_splits.build("dataset_001")

        self.assertEqual(self._dataset_snapshot(), before)
        for split in ("train", "val", "test"):
            self.assertFalse((self.out_dir / split / "P001_valid.png").exists())

    def test_source_directory_is_rejected_as_a_missing_image(self) -> None:
        (self.real_images / "P001_not_an_image.png").mkdir()
        self._add_source("P002_valid.png")
        self._add_source("P003_valid.png")
        self._write_labels([
            ("P001_not_an_image.png", "Invalid", "P001"),
            ("P002_valid.png", "Valid", "P002"),
            ("P003_valid.png", "Valid", "P003"),
        ])
        before = self._dataset_snapshot()

        with self.assertRaises(FileNotFoundError):
            build_splits.build("dataset_001")

        self.assertEqual(self._dataset_snapshot(), before)

    def test_collision_in_any_split_aborts_without_changes(self) -> None:
        filename = "P999_name.png"
        self._add_source(filename, b"new-real-image")
        # The writer's assigned split is deliberately irrelevant: any existing
        # filename in any split must block the merge globally.
        (self.out_dir / "val" / filename).write_bytes(b"existing-image")
        self._add_source("P998_name.png")
        self._add_source("P997_name.png")
        self._write_labels([
            (filename, "New Label", "P999"),
            ("P998_name.png", "Other", "P998"),
            ("P997_name.png", "Other", "P997"),
        ])
        before = self._dataset_snapshot()

        with self.assertRaises(FileExistsError):
            build_splits.build("dataset_001")

        self.assertEqual(self._dataset_snapshot(), before)

    def test_case_only_destination_collision_is_detected_portably(self) -> None:
        self._add_source("P999_name.png", b"new-real-image")
        (self.out_dir / "test" / "p999_NAME.PNG").write_bytes(b"existing-image")
        self._add_source("P998_name.png")
        self._add_source("P997_name.png")
        self._write_labels([
            ("P999_name.png", "New Label", "P999"),
            ("P998_name.png", "Other", "P998"),
            ("P997_name.png", "Other", "P997"),
        ])
        before = self._dataset_snapshot()

        with self.assertRaises(FileExistsError):
            build_splits.build("dataset_001")

        self.assertEqual(self._dataset_snapshot(), before)

    def test_malformed_csv_row_aborts_before_destination_changes(self) -> None:
        (self.real_dir / "labels.csv").write_text(
            "filename,label\nP001_only_one_column\n", encoding="utf-8"
        )
        before = self._dataset_snapshot()

        with self.assertRaisesRegex(ValueError, "missing required columns: writer_id"):
            build_splits.build("dataset_001")

        self.assertEqual(self._dataset_snapshot(), before)


if __name__ == "__main__":
    unittest.main()
