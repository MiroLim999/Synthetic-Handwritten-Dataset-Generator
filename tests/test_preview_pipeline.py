"""The studio preview shows exactly what a run writes."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image, ImageDraw

import config
import gui
from gui_studio import comparison_hints, fit_tile, real_scan_paths, tile_badges
from src.generate_synthetic import generate
from src.image_statistics import compare_image_statistics
from src.preview import PreviewTile, preview_samples, retry_statistics
from tests.studio_fakes import studio_app

OPTIONS = {"sample_mode": "semi_broken_mixed", "edge_clipping": "moderate",
           "augmentation_profile": "scan_noise", "custom_split_fractions": (0.6, 0.2, 0.2)}


def _pixels(image: Image.Image) -> bytes:
    return image.convert("RGB").tobytes()


class PreviewMatchesRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory()
        root = Path(cls._temporary.name) / "datasets"
        with (mock.patch.object(config, "DATASETS_DIR", root),
              contextlib.redirect_stdout(io.StringIO())):
            cls.threaded = generate(24, dataset="threads", seed=13, show_bar=False,
                                    workers=1, **OPTIONS)
            cls.processes = generate(24, dataset="processes", seed=13, show_bar=False,
                                     workers=2, **OPTIONS)

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    def _assert_match(self, tiles):
        self.assertTrue(tiles)
        for out_dir in (self.threaded, self.processes):
            for tile in tiles:
                self.assertIsNone(tile.error)
                written = Image.open(out_dir / tile.path)
                self.assertEqual(_pixels(written), _pixels(tile.image), tile.path)

    def test_first_tiles_match_the_generated_images(self):
        tiles = preview_samples(OPTIONS, count=24, seed=13, n=6)
        self.assertEqual([tile.index for tile in tiles], list(range(1, 7)))
        self._assert_match(tiles)

    def test_tiles_at_an_offset_match(self):
        tiles = preview_samples(OPTIONS, count=24, seed=13, n=5, offset=9)
        self.assertEqual([tile.index for tile in tiles], list(range(10, 15)))
        self._assert_match(tiles)

    def test_split_filter_is_deterministic_and_matches(self):
        first = preview_samples(OPTIONS, count=24, seed=13, n=3, split="test")
        again = preview_samples(OPTIONS, count=24, seed=13, n=3, split="test")
        self.assertEqual([t.path for t in first], [t.path for t in again])
        self.assertTrue(all(t.split == "test" for t in first))
        self.assertEqual([_pixels(t.image) for t in first], [_pixels(t.image) for t in again])
        self._assert_match(first)
        with self.assertRaises(ValueError):
            preview_samples(OPTIONS, count=24, seed=13, n=3, split="holdout")

    def test_retry_statistics_are_deterministic(self):
        first = retry_statistics(OPTIONS, count=24, seed=13, samples=12)
        second = retry_statistics(OPTIONS, count=24, seed=13, samples=12)
        self.assertEqual(first, second)
        self.assertEqual(first.samples, 12)
        self.assertGreaterEqual(first.mean_attempts, 1.0)
        self.assertIn("would be retried", first.describe())


class PreviewQueueTests(unittest.TestCase):
    def _app(self):
        app = studio_app()
        app.preview_grid = mock.Mock(tile_count=6, offset=0, split=None)
        app._preview_token = 0
        app._preview_after = None
        app._preview_temp_seed = 99
        app._mean_attempts = 1.0
        app.after = mock.Mock(return_value="after#1")
        app.after_cancel = mock.Mock()
        app._update_preflight_estimates = mock.Mock()
        return app

    def test_changes_are_debounced(self):
        app = self._app()
        app._schedule_studio_preview()
        app.after.assert_called_once_with(gui.PREVIEW_DEBOUNCE_MS, app._start_studio_preview)
        app._schedule_studio_preview(restart=True)
        app.after_cancel.assert_called_once_with("after#1")
        self.assertEqual(app.preview_grid.offset, 0)

    def test_stale_results_are_discarded(self):
        app = self._app()
        app._preview_token = 2
        tile = PreviewTile(index=1, path="train/syn_000001.png", field_type="age",
                           split="train")
        app._handle_preview_message(("studio_preview", 1, [tile], ""))
        app.preview_grid.show_tiles.assert_not_called()
        app._handle_preview_message(("studio_preview", 2, [tile], "note"))
        app.preview_grid.show_tiles.assert_called_once_with([tile], "note")
        app._handle_preview_message(("studio_preview_error", 1, "old failure"))
        app.preview_grid.show_error.assert_not_called()
        app._handle_preview_message(("studio_preview_error", 2, "boom"))
        app.preview_grid.show_error.assert_called_once_with("boom")

    def test_worker_reports_only_through_the_queue(self):
        app = self._app()
        app.after = mock.Mock(side_effect=AssertionError("worker called Tk"))
        request = app._preview_request()
        self.assertEqual(request["seed"], 42)
        self.assertEqual(request["note"], "")
        app.seed_var.set("")
        self.assertEqual(app._preview_request()["seed"], 99)
        request["count"], request["n"] = 12, 2
        app._preview_worker(5, request)
        kinds = [app.q.get_nowait() for _ in range(app.q.qsize())]
        self.assertEqual([(m[0], m[1]) for m in kinds],
                         [("studio_preview", 5), ("studio_retry", 5)])
        self.assertEqual(len(kinds[0][2]), 2)

    def test_retry_estimate_scales_the_duration(self):
        app = self._app()
        app._preview_token = 3
        stats = retry_statistics({}, count=12, seed=1, samples=4)
        app._handle_preview_message(("studio_retry", 3, stats))
        self.assertEqual(app._mean_attempts, stats.mean_attempts)
        app._update_preflight_estimates.assert_called_once()


class TileAndCompareTests(unittest.TestCase):
    def test_tiles_are_only_ever_scaled_down(self):
        small = Image.new("RGB", (100, 40))
        self.assertIs(fit_tile(small), small)
        self.assertEqual(fit_tile(Image.new("RGB", (900, 60))).size, (300, 20))
        tile = PreviewTile(index=3, path="train/syn_000003.png", field_type="place",
                           split="train", font="C:/Fonts/segoesc.ttf",
                           cut_sides=("top", "left"), attempts=2)
        self.assertEqual(tile_badges(tile),
                         "place | segoesc\ncut: top+left | kept on attempt 2\n"
                         "train/syn_000003.png")

    def test_comparison_against_a_real_image_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            real = Path(temporary)
            with mock.patch.object(config, "REAL_DIR", real):
                self.assertEqual(real_scan_paths(), [])
                (real / "images").mkdir()
                for index in range(5):
                    image = Image.new("RGB", (160, 50), (230, 225, 210))
                    ImageDraw.Draw(image).text((10, 18), f"Real {index}", fill=(80, 70, 60))
                    image.save(real / "images" / f"real_{index}.png")
                paths = real_scan_paths(limit=3)
                self.assertEqual(len(paths), 3)
                tiles = preview_samples({}, count=12, seed=3, n=6)
                comparison = compare_image_statistics([t.image for t in tiles], paths)
        hints = comparison_hints(comparison)
        self.assertEqual(len(hints), 8)
        self.assertTrue(any("synthetic is" in hint or "similar" in hint for hint in hints))

    def test_hint_wording(self):
        class _Metric:
            def __init__(self, sd):
                self.standardized_difference = sd

        class _Comparison:
            metrics = {"mean_luminance": _Metric(-1.2), "robust_contrast": _Metric(0.05),
                       "ink_coverage": _Metric(None), "edge_density": _Metric(0.8)}

        hints = comparison_hints(_Comparison())
        self.assertIn("\u2193 brightness: synthetic is darker (-1.2 sd)", hints)
        self.assertIn("~ contrast: similar (+0.1 sd)", hints)
        self.assertIn("= ink coverage: no spread to compare", hints)
        self.assertIn("\u2191 edges: synthetic is sharper (+0.8 sd)", hints)


if __name__ == "__main__":
    unittest.main()
