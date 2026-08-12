"""Unified-manifest and transactional real-merge regression tests."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import config
from PIL import Image, ImageDraw
from src import build_splits
from src.manifest import MANIFEST_COLUMNS, MANIFEST_SCHEMA_VERSION, manifest_bytes


def _write_ink_image(path: Path, *, shade: int = 0) -> None:
    image = Image.new("RGB", (96, 36), "white")
    ImageDraw.Draw(image).rectangle((12, 12, 84, 23), fill=(shade,) * 3)
    image.save(path, "PNG")


class RealMergeIntegrationTests(unittest.TestCase):
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

        synthetic_rows = []
        for index, split in enumerate(("train", "val", "test"), start=1):
            filename = f"syn_{index:06d}.png"
            _write_ink_image(self.out_dir / split / filename, shade=index)
            synthetic_rows.append({
                "filename": filename,
                "label": f"Synthetic {index}",
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
            mock.patch.object(config, "REAL_TEST_FRAC", 0.2),
            mock.patch.object(config, "RANDOM_SEED", 42),
            mock.patch.object(
                config, "resolve_dataset_dir", side_effect=lambda _name: self.out_dir
            ),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _set_inputs(self, rows: list[tuple[str, str, str]]) -> None:
        with (self.real_dir / "labels.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["filename", "label", "writer_id"])
            writer.writerows(rows)
        for index, (filename, _label, _writer_id) in enumerate(rows):
            image = self.real_images / filename
            if not image.exists():
                _write_ink_image(image, shade=(index + 10) % 100)

    def _manifest_rows(self) -> list[dict[str, str]]:
        with (self.out_dir / "manifest.csv").open(
            newline="", encoding="utf-8"
        ) as file:
            reader = csv.DictReader(file)
            self.assertEqual(tuple(reader.fieldnames or ()), MANIFEST_COLUMNS)
            return list(reader)

    def _real_rows(self) -> list[dict[str, str]]:
        return [row for row in self._manifest_rows() if row["source"] == "real"]

    def _dataset_files(self) -> dict[str, tuple[bytes, int]]:
        result: dict[str, tuple[bytes, int]] = {}
        for path in self.out_dir.rglob("*"):
            if path.is_file():
                result[path.relative_to(self.out_dir).as_posix()] = (
                    path.read_bytes(),
                    path.stat().st_mtime_ns,
                )
        return result

    @staticmethod
    def _rows(count: int) -> list[tuple[str, str, str]]:
        return [
            (f"sample_{index:03d}.png", f"Label {index}", f"W{index:03d}")
            for index in range(1, count + 1)
        ]

    def test_repeated_merge_is_byte_and_timestamp_identical(self) -> None:
        rows = self._rows(8)
        self._set_inputs(rows)
        first = build_splits.build("dataset_001")
        self.assertEqual((first.copied, first.unchanged, first.removed), (8, 0, 0))
        before = self._dataset_files()
        time.sleep(0.02)

        second = build_splits.build("dataset_001")

        self.assertEqual((second.copied, second.unchanged, second.removed), (0, 8, 0))
        self.assertEqual(self._dataset_files(), before)

    def test_merge_writes_passing_report_and_preserves_run_metadata(self) -> None:
        original_metadata = {
            "metadata_schema_version": 1,
            "seed": 1234,
            "custom_generator_value": {"keep": True},
            "real_merge": None,
        }
        (self.out_dir / "run-metadata.json").write_text(
            json.dumps(original_metadata), encoding="utf-8"
        )
        rows = self._rows(10)
        self._set_inputs(rows)

        result = build_splits.build("dataset_001")

        report = json.loads(
            (self.out_dir / "dataset-validation.json").read_text(encoding="utf-8")
        )
        metadata = json.loads(
            (self.out_dir / "run-metadata.json").read_text(encoding="utf-8")
        )
        self.assertTrue(report["valid"])
        self.assertEqual(metadata["seed"], 1234)
        self.assertEqual(metadata["custom_generator_value"], {"keep": True})
        self.assertEqual(metadata["manifest_sha256"], report["manifest_sha256"])
        self.assertEqual(metadata["images_sha256"], report["images_sha256"])
        self.assertEqual(metadata["real_merge"]["writer_count"], 10)
        self.assertEqual(
            sum(metadata["real_merge"]["writer_counts_by_split"].values()), 10
        )
        self.assertEqual(
            len(metadata["real_merge"]["writer_split_assignments"]), 10
        )
        self.assertEqual(metadata["real_merge"]["result"]["copied"], result.copied)
        self.assertEqual(len(metadata["real_merge"]["source_csv_sha256"]), 64)
        self.assertIn("merged_at", metadata["real_merge"])

    def test_validation_failure_rolls_back_entire_merge(self) -> None:
        rows = self._rows(3)
        self._set_inputs(rows)
        build_splits.build("dataset_001")
        before = self._dataset_files()

        changed = list(rows)
        changed[0] = (changed[0][0], "Changed Label", changed[0][2])
        blank = Image.new("RGB", (96, 36), "white")
        blank.save(self.real_images / changed[0][0], "PNG")
        self._set_inputs(changed)

        with self.assertRaisesRegex(ValueError, "Dataset validation failed"):
            build_splits.build("dataset_001")

        self.assertEqual(self._dataset_files(), before)
        self.assertFalse(any(self.out_dir.glob(".real-merge-*")))

    def test_run_metadata_serialization_failure_rolls_back_entire_merge(self) -> None:
        rows = self._rows(3)
        self._set_inputs(rows)
        build_splits.build("dataset_001")
        before = self._dataset_files()

        changed = list(rows)
        changed[0] = (changed[0][0], "Changed Label", changed[0][2])
        _write_ink_image(self.real_images / changed[0][0], shade=120)
        self._set_inputs(changed)
        real_json_bytes = build_splits._json_bytes

        def fail_run_metadata(value):
            if "real_merge" in value:
                raise OSError("injected run metadata failure")
            return real_json_bytes(value)

        with mock.patch.object(
            build_splits, "_json_bytes", side_effect=fail_run_metadata
        ):
            with self.assertRaisesRegex(OSError, "injected run metadata failure"):
                build_splits.build("dataset_001")

        self.assertEqual(self._dataset_files(), before)
        self.assertFalse(any(self.out_dir.glob(".real-merge-*")))

    def test_adding_writers_never_moves_existing_or_test_writers(self) -> None:
        initial = self._rows(10)
        self._set_inputs(initial)
        build_splits.build("dataset_001")
        assignments = {row["writer_id"]: row["split"] for row in self._real_rows()}
        held_out = {writer for writer, split in assignments.items() if split == "test"}
        self.assertTrue(held_out)

        expanded = list(reversed(self._rows(15)))
        self._set_inputs(expanded)
        result = build_splits.build("dataset_001")
        after = {row["writer_id"]: row["split"] for row in self._real_rows()}

        self.assertEqual(result.copied, 5)
        self.assertEqual(result.unchanged, 10)
        for writer, split in assignments.items():
            self.assertEqual(after[writer], split)
        self.assertTrue(all(after[writer] == "test" for writer in held_out))

    def test_removed_writer_keeps_assignment_when_later_reintroduced(self) -> None:
        rows = self._rows(10)
        self._set_inputs(rows)
        build_splits.build("dataset_001")
        before = {row["writer_id"]: row["split"] for row in self._real_rows()}
        held_out_writer = next(
            writer for writer, split in before.items() if split == "test"
        )
        held_out_row = next(row for row in rows if row[2] == held_out_writer)

        without_writer = [row for row in rows if row[2] != held_out_writer]
        self._set_inputs(without_writer)
        removed = build_splits.build("dataset_001")
        self.assertEqual(removed.removed, 1)
        self.assertNotIn(held_out_writer, {row["writer_id"] for row in self._real_rows()})

        self._set_inputs(rows)
        restored = build_splits.build("dataset_001")
        after = {row["writer_id"]: row["split"] for row in self._real_rows()}
        self.assertEqual(restored.copied, 1)
        self.assertEqual(after[held_out_writer], "test")
        restored_path = self.out_dir / "test" / held_out_row[0]
        self.assertTrue(restored_path.is_file())

    def test_changed_real_record_is_reconciled_without_duplicates(self) -> None:
        rows = self._rows(3)
        self._set_inputs(rows)
        build_splits.build("dataset_001")
        target_filename = rows[0][0]
        target_writer = rows[0][2]

        changed = list(rows)
        changed[0] = (target_filename, "Corrected Label", target_writer)
        _write_ink_image(self.real_images / target_filename, shade=100)
        expected_image = (self.real_images / target_filename).read_bytes()
        self._set_inputs(changed)
        result = build_splits.build("dataset_001")

        matching = [row for row in self._real_rows() if row["filename"] == target_filename]
        self.assertEqual(result.copied, 1)
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["label"], "Corrected Label")
        destination = self.out_dir / matching[0]["split"] / target_filename
        self.assertEqual(destination.read_bytes(), expected_image)

    def test_commit_failure_rolls_back_images_and_both_metadata_files(self) -> None:
        rows = self._rows(3)
        self._set_inputs(rows)
        build_splits.build("dataset_001")
        before = self._dataset_files()

        changed = list(rows)
        changed[0] = (changed[0][0], "Changed", changed[0][2])
        _write_ink_image(self.real_images / changed[0][0], shade=110)
        self._set_inputs(changed)

        real_replace = os.replace

        def fail_manifest_install(source, destination):
            destination = Path(destination)
            source = Path(source)
            if (
                destination == self.out_dir / "manifest.csv"
                and source.parent.name == "metadata"
            ):
                raise OSError("injected manifest commit failure")
            return real_replace(source, destination)

        with mock.patch.object(build_splits.os, "replace", side_effect=fail_manifest_install):
            with self.assertRaisesRegex(OSError, "injected manifest commit failure"):
                build_splits.build("dataset_001")

        self.assertEqual(self._dataset_files(), before)
        self.assertFalse(any(self.out_dir.glob(".real-merge-*")))

    def test_legacy_real_manifest_is_migrated_once(self) -> None:
        # Simulate the former six-column synthetic manifest plus a separate
        # real_manifest.csv produced by the old merge implementation.
        (self.out_dir / "manifest.csv").write_text(
            "filename,label,split,field_type,font,sample_mode\n"
            "syn_000001.png,Synthetic 1,train,name,Test.ttf,regular\n"
            "syn_000002.png,Synthetic 2,val,name,Test.ttf,regular\n"
            "syn_000003.png,Synthetic 3,test,name,Test.ttf,regular\n",
            encoding="utf-8",
        )
        legacy_name = "legacy_real.png"
        _write_ink_image(self.out_dir / "test" / legacy_name, shade=50)
        (self.out_dir / "real_manifest.csv").write_text(
            "filename,label,split,writer\n"
            f"{legacy_name},Legacy Label,test,WLEGACY\n",
            encoding="utf-8",
        )
        (self.out_dir / "labels.csv").write_text(
            "filename,label,split\n"
            "syn_000001.png,Synthetic 1,train\n"
            "syn_000002.png,Synthetic 2,val\n"
            "syn_000003.png,Synthetic 3,test\n"
            f"{legacy_name},Legacy Label,test\n",
            encoding="utf-8",
        )
        _write_ink_image(self.real_images / legacy_name, shade=50)
        self._set_inputs([(legacy_name, "Legacy Label", "WLEGACY")])

        result = build_splits.build("dataset_001")

        self.assertEqual((result.copied, result.unchanged), (0, 1))
        self.assertFalse((self.out_dir / "real_manifest.csv").exists())
        real_row = self._real_rows()[0]
        self.assertEqual(real_row["writer_id"], "WLEGACY")
        self.assertEqual(real_row["split"], "test")


if __name__ == "__main__":
    unittest.main()
