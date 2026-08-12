import csv
import hashlib
import json
import threading
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest import mock

from PIL import Image, ImageDraw

import config
from src import generate_synthetic as generator
from src.provenance import hash_dataset_images


class GenerateSafetyTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.base = Path(self._temp.name) / "datasets"
        self.base.mkdir()
        self._datasets_patch = mock.patch.object(config, "DATASETS_DIR", self.base)
        self._datasets_patch.start()

    def tearDown(self):
        self._datasets_patch.stop()
        self._temp.cleanup()

    @contextmanager
    def stubbed_pipeline(self, render=None):
        if render is None:
            def render(*_args, **_kwargs):
                image = Image.new("RGB", (48, 24), "white")
                ImageDraw.Draw(image).rectangle((8, 8, 40, 15), fill="black")
                return image, "TestFont.ttf"
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                generator,
                "_preflight_generation",
                return_value=("regular", config.NAMES_DIR),
            ))
            stack.enter_context(mock.patch.object(
                generator,
                "_weighted_field_types",
                side_effect=lambda count, _mode, rng=None: ["age"] * count,
            ))
            stack.enter_context(mock.patch.object(
                generator.fields,
                "make_value_with_format",
                return_value=("42", ""),
            ))
            stack.enter_context(mock.patch.object(generator, "render_text", side_effect=render))
            stack.enter_context(mock.patch.object(
                generator,
                "degrade",
                side_effect=lambda image, damage_profile, **_kwargs: image,
            ))
            yield

    def assert_no_internal_artifacts(self):
        leftovers = [path.name for path in self.base.iterdir() if path.name.startswith(".")]
        self.assertEqual(leftovers, [])

    @contextmanager
    def stubbed_archive_validation(self):
        report = mock.Mock()
        report.raise_for_errors.return_value = None
        with (mock.patch.object(generator, "validate_dataset", return_value=report),
              mock.patch.object(generator, "_require_fresh_validation_artifacts")):
            yield report

    def test_success_is_published_only_after_validation(self):
        with self.stubbed_pipeline():
            out_dir = generator.generate(5, dataset="safe_run", show_bar=False)

        self.assertEqual(out_dir.resolve(), (self.base / "safe_run").resolve())
        with open(out_dir / "manifest.csv", encoding="utf-8", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 5)
        self.assertEqual(sum(1 for _ in out_dir.glob("*/*.png")), 5)
        self.assertTrue((out_dir / "dataset-validation.json").is_file())
        self.assertTrue((out_dir / "review-montage.jpg").is_file())
        self.assertTrue((out_dir / "image-statistics.json").is_file())
        metadata = json.loads((out_dir / "run-metadata.json").read_text("utf-8"))
        self.assertEqual(metadata["seed"], config.RANDOM_SEED)
        self.assertEqual(metadata["row_count"], 5)
        self.assertIn("evaluation_policy", metadata)
        self.assert_no_internal_artifacts()

    @contextmanager
    def deterministic_seeded_pipeline(self):
        def value(_field_type, *, rng, names_dir=None, format_profile="base"):
            return f"Value {rng.randrange(1_000_000):06d}", ""

        def render(_label, *, specific_font, rng, **_kwargs):
            image = Image.new("RGB", (80, 32), "white")
            x = rng.randint(8, 15)
            ImageDraw.Draw(image).rectangle((x, 10, x + 50, 20), fill="black")
            return image, f"{specific_font}.ttf"

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                generator, "_preflight_generation",
                return_value=("regular", config.NAMES_DIR)))
            stack.enter_context(mock.patch.object(
                generator, "fonts_for_style",
                return_value=("A.ttf", "B.ttf", "C.ttf")))
            stack.enter_context(mock.patch.object(
                generator.fields, "make_value_with_format", side_effect=value))
            stack.enter_context(mock.patch.object(
                generator, "render_text", side_effect=render))
            stack.enter_context(mock.patch.object(
                generator, "degrade",
                side_effect=lambda image, **_kwargs: image))
            yield

    def test_run_seed_reproduces_manifest_and_image_hashes(self):
        with self.deterministic_seeded_pipeline():
            first = generator.generate(12, "same_seed_a", seed=2026, show_bar=False)
            second = generator.generate(12, "same_seed_b", seed=2026, show_bar=False)
            different = generator.generate(12, "different_seed", seed=2027, show_bar=False)

        first_manifest = (first / "manifest.csv").read_bytes()
        second_manifest = (second / "manifest.csv").read_bytes()
        different_manifest = (different / "manifest.csv").read_bytes()
        self.assertEqual(first_manifest, second_manifest)
        self.assertNotEqual(first_manifest, different_manifest)
        self.assertEqual(
            hash_dataset_images(first, config.SPLIT_NAMES),
            hash_dataset_images(second, config.SPLIT_NAMES),
        )
        self.assertNotEqual(
            hash_dataset_images(first, config.SPLIT_NAMES),
            hash_dataset_images(different, config.SPLIT_NAMES),
        )

    def test_generation_failure_never_publishes_partial_output(self):
        calls = 0

        def fail_on_second(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected render failure")
            return Image.new("RGB", (24, 12), "white"), "TestFont.ttf"

        with self.stubbed_pipeline(fail_on_second):
            with self.assertRaisesRegex(RuntimeError, "injected render failure"):
                generator.generate(5, dataset="failed_run", show_bar=False)

        self.assertFalse((self.base / "failed_run").exists())
        self.assert_no_internal_artifacts()

    def test_callback_failure_never_publishes_partial_output(self):
        def failing_callback(*_args):
            raise RuntimeError("injected callback failure")

        with self.stubbed_pipeline():
            with self.assertRaisesRegex(RuntimeError, "injected callback failure"):
                generator.generate(
                    5,
                    dataset="callback_failure",
                    show_bar=False,
                    progress_callback=failing_callback,
                )

        self.assertFalse((self.base / "callback_failure").exists())
        self.assert_no_internal_artifacts()

    def test_image_save_failure_never_publishes_partial_output(self):
        with self.stubbed_pipeline():
            with mock.patch.object(Image.Image, "save", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    generator.generate(5, dataset="save_failure", show_bar=False)

        self.assertFalse((self.base / "save_failure").exists())
        self.assert_no_internal_artifacts()

    def test_manifest_write_failure_never_publishes_partial_output(self):
        with self.stubbed_pipeline():
            with mock.patch.object(
                    generator.AtomicCsvWriter, "write",
                    side_effect=OSError("metadata disk full")):
                with self.assertRaisesRegex(OSError, "metadata disk full"):
                    generator.generate(5, dataset="manifest_failure", show_bar=False)

        self.assertFalse((self.base / "manifest_failure").exists())
        self.assert_no_internal_artifacts()

    def test_cancellation_cleans_staging_and_reservation(self):
        cancelled = threading.Event()

        def progress(*_args):
            cancelled.set()

        with self.stubbed_pipeline():
            with self.assertRaises(generator.GenerationCancelled):
                generator.generate(
                    5,
                    dataset="cancelled_run",
                    show_bar=False,
                    progress_callback=progress,
                    cancel_event=cancelled,
                )

        self.assertFalse((self.base / "cancelled_run").exists())
        self.assert_no_internal_artifacts()

    def test_existing_output_is_not_modified(self):
        existing = self.base / "existing"
        existing.mkdir()
        sentinel = existing / "keep.txt"
        sentinel.write_text("unchanged", encoding="utf-8")

        with self.stubbed_pipeline():
            with self.assertRaises(FileExistsError):
                generator.generate(2, dataset="existing", show_bar=False)

        self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged")
        self.assert_no_internal_artifacts()

    def test_zip_refuses_collision_and_preserves_existing_archive(self):
        dataset = self.base / "archive_me"
        (dataset / "train").mkdir(parents=True)
        (dataset / "train" / "sample.txt").write_text("sample", encoding="utf-8")

        with self.stubbed_archive_validation() as validation:
            archive = generator.zip_dataset(dataset)
        validation.raise_for_errors.assert_called_once_with()
        original = archive.read_bytes()
        checksum = archive.with_name(f"{archive.name}.sha256")
        self.assertEqual(
            checksum.read_text(encoding="ascii"),
            f"{hashlib.sha256(original).hexdigest()}  {archive.name}\n",
        )
        with self.assertRaises(FileExistsError):
            with self.stubbed_archive_validation():
                generator.zip_dataset(dataset)

        self.assertEqual(archive.read_bytes(), original)
        self.assertTrue(dataset.exists())
        self.assert_no_internal_artifacts()

    def test_cancelled_packaging_does_not_publish_archive(self):
        dataset = self.base / "cancel_archive"
        (dataset / "train").mkdir(parents=True)
        (dataset / "train" / "sample.txt").write_text("sample", encoding="utf-8")
        event = threading.Event()

        def create_then_cancel(*args, **kwargs):
            result = shutil_make_archive(*args, **kwargs)
            event.set()
            return result

        shutil_make_archive = generator.shutil.make_archive
        with mock.patch.object(generator.shutil, "make_archive", side_effect=create_then_cancel):
            with self.assertRaises(generator.GenerationCancelled):
                with self.stubbed_archive_validation():
                    generator.zip_dataset(dataset, cancel_event=event)

        self.assertTrue(dataset.is_dir())
        self.assertFalse((self.base / "cancel_archive.zip").exists())
        self.assertFalse((self.base / "cancel_archive.zip.sha256").exists())
        self.assert_no_internal_artifacts()

    def test_zip_only_removes_only_the_valid_dataset(self):
        dataset = self.base / "remove_after_zip"
        (dataset / "train").mkdir(parents=True)
        (dataset / "train" / "sample.txt").write_text("sample", encoding="utf-8")
        sibling = self.base / "keep_me"
        sibling.mkdir()

        with self.stubbed_archive_validation():
            archive = generator.zip_dataset(dataset, remove_dir=True)

        self.assertTrue(archive.is_file())
        self.assertFalse(dataset.exists())
        self.assertTrue(sibling.is_dir())
        self.assert_no_internal_artifacts()

    def test_archive_failure_preserves_source_and_never_publishes_zip(self):
        dataset = self.base / "archive_failure"
        (dataset / "train").mkdir(parents=True)
        sentinel = dataset / "train" / "sample.txt"
        sentinel.write_text("sample", encoding="utf-8")

        with mock.patch.object(
                generator.shutil, "make_archive", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                with self.stubbed_archive_validation():
                    generator.zip_dataset(dataset, remove_dir=True)

        self.assertEqual(sentinel.read_text(encoding="utf-8"), "sample")
        self.assertFalse((self.base / "archive_failure.zip").exists())
        self.assert_no_internal_artifacts()

    def test_guarded_remove_rejects_container_and_outside_path(self):
        outside = Path(self._temp.name) / "outside"
        outside.mkdir()

        with self.assertRaises(ValueError):
            generator.guarded_remove_dataset(self.base)
        with self.assertRaises(ValueError):
            generator.guarded_remove_dataset(outside)

        self.assertTrue(self.base.exists())
        self.assertTrue(outside.exists())


if __name__ == "__main__":
    unittest.main()
