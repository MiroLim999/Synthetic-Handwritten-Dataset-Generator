"""The same recipe gives the same dataset through the GUI, the CLI, and replay."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import config
from src import studio_spec
from src.generate_synthetic import main
from src.studio_recipe import RECIPE_FORMAT, load_recipe, parse_recipe, save_recipe
from tests.studio_fakes import studio_app


def _recipe():
    cutoff = studio_spec.default_section("cutoff")
    cutoff["top"] = {"probability": 0.6, "range": [0.04, 0.09]}
    cutoff["max_ink_loss"] = 0.12
    return parse_recipe({
        "format": RECIPE_FORMAT, "schema_version": 2, "name": "Parity",
        "description": "ranges in every section",
        "augmentation": {
            "fade": {"probability": 0.5, "range": [0.5, 0.9]},
            "rotation": {"probability": 0.9, "range": [1.0, 3.5]},
            "stains": {"probability": 0.5, "count_range": [1, 4],
                       "opacity_range": [20, 40]},
            "jpeg": {"probability": 0.5, "range": [60, 90]},
        },
        "stroke_damage": {
            "gaps": {"probability": 0.8, "count_range": [4, 12], "size_range": [2, 9]},
            "erode": {"probability": 0.4, "range": [0.1, 0.3]},
        },
        "cutoff": cutoff,
        "run": {"sample_mode": "semi_broken_mixed", "split_fractions": [0.7, 0.15, 0.15],
                "field_weights": {"full_name": 3, "place": 2, "character": 1},
                "samples_per_writer": 4},
    })


def _hashes(out_dir: Path) -> tuple[str, str]:
    metadata = json.loads((out_dir / "run-metadata.json").read_text(encoding="utf-8"))
    return metadata["images_sha256"], metadata["manifest_sha256"]


class RecipeParityTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)
        patcher = mock.patch.object(config, "DATASETS_DIR", self.root / "datasets")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.recipe_path = self.root / "parity.json"
        save_recipe(self.recipe_path, _recipe())

    def _cli(self, *args) -> Path:
        with contextlib.redirect_stdout(io.StringIO()):
            return main(["--count", "16", "--seed", "42", "--workers", "1", *args])

    def test_gui_snapshot_and_cli_give_identical_datasets(self):
        app = studio_app()
        app._apply_loaded_recipe("parity", load_recipe(self.recipe_path))
        app.count_var.set("16")
        app.dataset_var.set("gui_run")
        app.workers_var.set("1")
        options = app._snapshot_run_options()
        with contextlib.redirect_stdout(io.StringIO()):
            app._run(*app._worker_args(options, threading.Event()))
        messages = []
        while not app.q.empty():
            messages.append(app.q.get_nowait())
        done = [message for message in messages if message[0] == "done"]
        self.assertTrue(done, messages)
        gui_dir = Path(done[0][2])

        cli_dir = self._cli("--recipe", str(self.recipe_path), "--dataset", "cli_run")
        self.assertEqual(_hashes(gui_dir), _hashes(cli_dir))
        gui_meta = json.loads((gui_dir / "run-metadata.json").read_text("utf-8"))
        self.assertEqual(gui_meta["studio_recipe"]["source_preset"], "parity")
        self.assertFalse(gui_meta["studio_recipe"]["modified"])
        self.assertEqual(gui_meta["effective_selection"]["sample_mode"],
                         "semi_broken_mixed")

    def test_replaying_a_datasets_recipe_reproduces_it(self):
        first = self._cli("--recipe", str(self.recipe_path), "--dataset", "first",
                          "--cutoff", "heavy")
        replay = self._cli("--recipe", str(first / "studio-recipe.json"),
                           "--dataset", "replay")
        self.assertEqual(_hashes(first), _hashes(replay))
        self.assertEqual((first / "studio-recipe.json").read_bytes(),
                         (replay / "studio-recipe.json").read_bytes())
        meta = json.loads((first / "run-metadata.json").read_text("utf-8"))
        self.assertTrue(meta["studio_recipe"]["modified"])  # --cutoff overrode it
        self.assertEqual(meta["effective_selection"]["edge_clipping"], "heavy")

    def test_print_recipe_writes_nothing_and_bad_flags_fail_early(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.assertIsNone(main(["--recipe", str(self.recipe_path), "--print-recipe",
                                    "--samples-per-writer", "9"]))
        printed = json.loads(buffer.getvalue())
        self.assertEqual(printed["run"]["samples_per_writer"], 9)
        self.assertEqual(printed["name"], "Parity")
        self.assertFalse((self.root / "datasets").exists())

        for bad in (["--custom-field-weights", "{bad json"],
                    ["--custom-split-fractions", "0.5", "0.5", "0.5"],
                    ["--workers", "0"],
                    ["--recipe", str(self.root / "missing.json")]):
            with (contextlib.redirect_stderr(io.StringIO()),
                  self.assertRaises(SystemExit)):
                main(["--count", "8", *bad])
        self.assertFalse((self.root / "datasets").exists())


if __name__ == "__main__":
    unittest.main()
