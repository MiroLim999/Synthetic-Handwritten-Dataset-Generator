"""Regression tests for the S1 Developer Studio correctness fixes."""

from __future__ import annotations

import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

import config
import gui
from src.augment import degrade
from src.generate_synthetic import field_weights_for_mode, generate
from src.generation_profiles import StrokeDamageProfile
from src.generation_io import iter_generation_plan
from tests.studio_fakes import studio_app


def _ink_image():
    image = Image.new("RGB", (220, 80), "white")
    ImageDraw.Draw(image).text((20, 25), "Maasin 1948", fill="black")
    return image


def _metadata(out_dir: Path) -> dict:
    return json.loads((out_dir / "run-metadata.json").read_text(encoding="utf-8"))


class RegularModeStrokeDamageTests(unittest.TestCase):
    def test_regular_degrade_ignores_stroke_params(self):
        params = {"contrast_prob": 1.0, "erode_prob": 1.0, "gap_prob": 1.0,
                  "gap_count": 30, "scratch_prob": 1.0}
        for profile in ("historical_scan_v1", "clean", "scan_noise"):
            plain = degrade(_ink_image(), "regular", rng=random.Random(9),
                            np_rng=np.random.default_rng(9),
                            augmentation_profile=profile)
            tuned = degrade(_ink_image(), "regular", rng=random.Random(9),
                            np_rng=np.random.default_rng(9),
                            augmentation_profile=profile,
                            semi_broken_params=params)
            self.assertEqual(plain.tobytes(), tuned.tobytes(), profile)

    def test_snapshot_passes_stroke_damage_only_in_semi_broken_modes(self):
        self.assertIsNone(
            studio_app("regular")._snapshot_run_options().stroke_damage_profile)
        for mode in config.SAMPLE_MODES:
            if not mode.startswith("semi_broken"):
                continue
            profile = studio_app(mode)._snapshot_run_options().stroke_damage_profile
            self.assertIsInstance(profile, StrokeDamageProfile, mode)
            self.assertEqual(profile.gap_count_range, (1, 16))
            self.assertEqual(profile.contrast_probability, 0.8)

    def test_stroke_sliders_disabled_outside_semi_broken_modes(self):
        app = studio_app("regular")
        scales = {name: mock.Mock() for name in gui.STROKE_DAMAGE_SLIDERS}
        app.dev_scales = scales
        app._sync_stroke_damage_state()
        for scale in scales.values():
            scale.config.assert_called_with(state="disabled")
        app.sample_mode_var.set(config.SAMPLE_MODES["semi_broken_words"])
        app._sync_stroke_damage_state()
        for scale in scales.values():
            scale.config.assert_called_with(state="normal")
        # While a job holds the lock, the restore path owns widget states.
        app._run_control_states = [(object(), "normal")]
        for scale in scales.values():
            scale.config.reset_mock()
        app._sync_stroke_damage_state()
        for scale in scales.values():
            scale.config.assert_not_called()


class FieldWeightTests(unittest.TestCase):
    def test_grid_weights_give_the_same_plan_as_the_mode(self):
        split_counts = config.allocate_synthetic_split_counts(200)
        for mode in config.SAMPLE_MODES:
            app = studio_app(mode)
            grid = app._snapshot_run_options().custom_field_weights
            self.assertEqual(set(grid), set(config.SUPPORTED_FIELD_TYPES))
            from_grid = list(iter_generation_plan(
                200, grid, split_counts, random.Random(7)))
            from_mode = list(iter_generation_plan(
                200, field_weights_for_mode(mode), split_counts, random.Random(7)))
            self.assertEqual(from_grid, from_mode, mode)

    def test_regular_grid_has_no_character_or_numeric_weight(self):
        grid = studio_app("regular")._snapshot_run_options().custom_field_weights
        self.assertEqual(grid["character"], 0)
        self.assertEqual(grid["numeric"], 0)

    def test_mode_change_replaces_edits_and_says_so(self):
        app = studio_app("regular")
        app.field_weight_vars["age"].set("99")
        app.sample_mode_var.set(config.SAMPLE_MODES["semi_broken_characters"])
        app._apply_mode_field_weights()
        self.assertEqual(app.field_weight_vars["character"].get(), "1")
        self.assertEqual(app.field_weight_vars["age"].get(), "0")
        self.assertIn("edits were replaced", app.status.config.call_args.kwargs["text"])

        app.status.reset_mock()
        app.sample_mode_var.set(config.SAMPLE_MODES["regular"])
        app._apply_mode_field_weights()
        app.status.config.assert_not_called()

    def test_reset_weights_uses_the_current_mode(self):
        app = studio_app("semi_broken_numerics")
        app.field_weight_vars["numeric"].set("1")
        app._reset_field_weights()
        expected = field_weights_for_mode("semi_broken_numerics")
        self.assertEqual(app.field_weight_vars["numeric"].get(), str(expected["numeric"]))


class PresetLoadingTests(unittest.TestCase):
    def _load(self, app, data):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "candidate.json").write_text(json.dumps(data), encoding="utf-8")
            app.preset_var.set("candidate")
            with (mock.patch.object(gui, "PRESET_DIR", folder),
                  mock.patch.object(gui.messagebox, "showerror") as showerror):
                app._load_selected_preset()
        return showerror

    def test_malformed_preset_leaves_sliders_unchanged_and_responsive(self):
        app = studio_app()
        before = {name: var.get() for name, var in app.dev_sliders.items()}
        showerror = self._load(app, {"tint": 0.3, "fade": "faded", "noise": float("nan")})
        showerror.assert_called_once()
        self.assertIn("fade", showerror.call_args.args[1])
        self.assertEqual({name: var.get() for name, var in app.dev_sliders.items()}, before)
        self.assertFalse(app._slider_updating)

    def test_failure_while_applying_still_releases_slider_lock(self):
        app = studio_app()
        app.dev_labels["tint"].set = mock.Mock(side_effect=RuntimeError("Tk gone"))
        with self.assertRaises(RuntimeError):
            app._apply_preset_values("x", {"tint": 0.2}, [])
        self.assertFalse(app._slider_updating)

    def test_valid_preset_is_clamped_snapped_and_reported(self):
        app = studio_app()
        self._load(app, {"name": "X", "tint": 0.9, "jpeg": 66.4, "crop_top": 3,
                         "mystery": 1})
        self.assertEqual(app.dev_sliders["tint"].get(), gui.SLIDER_SPECS["tint"].high)
        self.assertEqual(app.dev_sliders["jpeg"].get(), 66.0)
        self.assertEqual(app.edge_clipping_var.get(), config.EDGE_CLIPPING_OPTIONS["custom"])
        message = app.status.config.call_args.kwargs["text"]
        self.assertIn("crop_top, jpeg, tint", message)
        self.assertIn("ignored mystery", message)

    def test_preset_without_crops_sets_cutoff_to_none(self):
        app = studio_app()
        app.edge_clipping_var.set(config.EDGE_CLIPPING_OPTIONS["heavy"])
        self._load(app, {"crop_top": 0, "crop_bottom": 0, "crop_left": 0,
                         "crop_right": 0})
        self.assertEqual(app.edge_clipping_var.get(), config.EDGE_CLIPPING_OPTIONS["none"])


class PresetSavingTests(unittest.TestCase):
    def _save(self, app, folder, name, *, overwrite=False):
        with (mock.patch.object(gui, "PRESET_DIR", folder),
              mock.patch.object(gui.simpledialog, "askstring", return_value=name),
              mock.patch.object(gui.messagebox, "askyesno", return_value=overwrite),
              mock.patch.object(gui.messagebox, "showerror") as showerror):
            app._refresh_preset_list = mock.Mock()
            app._save_current_preset()
        return showerror

    def test_unsafe_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "presets"
            folder.mkdir()
            for name in ("a/b", "..\\x", "con", ":x", "   ", ""):
                showerror = self._save(studio_app(), folder, name)
                showerror.assert_called_once()
            self.assertEqual(list(folder.iterdir()), [])

    def test_declined_overwrite_leaves_file_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            existing = folder / "my_preset.json"
            existing.write_text('{"name": "original"}', encoding="utf-8")
            self._save(studio_app(), folder, "My Preset", overwrite=False)
            self.assertEqual(existing.read_text(encoding="utf-8"), '{"name": "original"}')

            app = studio_app()
            app.dev_sliders["tint"].set(0.18319)
            app.dev_sliders["jpeg"].set(66.2)
            self._save(app, folder, "My Preset", overwrite=True)
            saved = json.loads(existing.read_text(encoding="utf-8"))
            self.assertEqual(saved["name"], "My Preset")
            self.assertEqual(saved["tint"], 0.18)
            self.assertEqual(saved["jpeg"], 66)
            self.assertEqual(list(folder.glob(".*.tmp")), [])

    def test_slug_rules(self):
        self.assertEqual(gui.preset_slug("Heavy Ink & Gaps"), "heavy_ink_gaps")
        self.assertEqual(gui.preset_slug("  Mixed-Case 2 "), "mixed-case_2")
        for name in ("COM1", "lpt9", "x" * 65, "&&&"):
            with self.assertRaises(ValueError):
                gui.preset_slug(name)
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(
                gui.preset_path_for("ok", Path(temporary)).parent, Path(temporary))


class SnapAndFormatTests(unittest.TestCase):
    def test_snap_value_edges(self):
        self.assertEqual(gui.snap_value(0.123, 0.01, 0, 1), 0.12)
        self.assertEqual(gui.snap_value(4.46, 0.1, 0, 12), 4.5)
        self.assertEqual(gui.snap_value(66.5, 1, 20, 100), 66.0)  # round-half-even
        self.assertEqual(gui.snap_value(-5, 1, 20, 100), 20.0)
        self.assertEqual(gui.snap_value(1e9, 0.01, 0, 1), 1.0)
        self.assertEqual(gui.snap_value(0.30000000000000004, 0.1, 0, 1), 0.3)
        for bad in ((float("nan"), 0.1), (float("inf"), 0.1), (1, 0), (1, -1)):
            with self.assertRaises(ValueError):
                gui.snap_value(bad[0], bad[1], 0, 10)
        with self.assertRaises(ValueError):
            gui.snap_value(1, 1, 5, 0)

    def test_format_value_edges(self):
        self.assertEqual(gui.format_value(0.3, 0.01), "0.30")
        self.assertEqual(gui.format_value(4.5, 0.1, "°"), "4.5°")
        self.assertEqual(gui.format_value(12.6, 1, "%"), "13%")  # no int() truncation
        self.assertEqual(gui.format_value(-0.0001, 0.01), "0.00")
        self.assertEqual(gui.format_value(75.0, 1), "75")

    def test_every_slider_default_is_on_its_step(self):
        for spec in gui.SLIDER_SPECS.values():
            self.assertEqual(
                gui.snap_value(spec.default, spec.step, spec.low, spec.high),
                spec.default, spec.name)


class ResetScopeTests(unittest.TestCase):
    def test_studio_reset_touches_only_studio_sliders(self):
        app = studio_app()
        app.dev_sliders["contrast"].set(0.1)
        app.dev_sliders["crop_top"].set(5.0)
        app.split_train_var.set("70")
        app.field_weight_vars["age"].set("77")
        app._reset_studio_sliders()
        self.assertEqual(app.dev_sliders["contrast"].get(), 0.8)
        self.assertEqual(app.dev_sliders["crop_top"].get(), 0.0)
        # historical_scan_v1 midpoint, snapped to the 0.01 step.
        self.assertEqual(app.dev_sliders["fade"].get(), 0.68)
        self.assertEqual(app.split_train_var.get(), "70")
        self.assertEqual(app.field_weight_vars["age"].get(), "77")


class ProvenanceAndParityTests(unittest.TestCase):
    def test_gui_defaults_match_cli_defaults_and_record_provenance(self):
        options = studio_app("regular", count=12, seed=42)._snapshot_run_options()
        with (tempfile.TemporaryDirectory() as temporary,
              mock.patch.object(config, "DATASETS_DIR", Path(temporary) / "datasets"),
              contextlib.redirect_stdout(io.StringIO())):
            gui_dir = generate(
                options.count, dataset="gui_run", seed=options.seed,
                names_version=options.names_version,
                sample_mode=options.sample_mode, font_style=options.font_style,
                cursive_group=options.cursive_group,
                specific_font=options.specific_font, show_bar=False,
                augmentation_profile=options.augmentation_profile,
                edge_clipping=options.edge_clipping,
                stroke_damage_profile=options.stroke_damage_profile,
                custom_field_weights=options.custom_field_weights,
                custom_split_fractions=options.custom_split_fractions,
                workers=options.workers,
            )
            cli_dir = generate(12, dataset="cli_run", seed=42, show_bar=False)
            semi_dir = generate(12, dataset="semi_run", seed=42, show_bar=False,
                                sample_mode="semi_broken_mixed")
            gui_meta, cli_meta = _metadata(gui_dir), _metadata(cli_dir)
            semi_meta = _metadata(semi_dir)

        self.assertEqual(gui_meta["images_sha256"], cli_meta["images_sha256"])
        self.assertEqual(gui_meta["manifest_sha256"], cli_meta["manifest_sha256"])

        selection = gui_meta["effective_selection"]
        self.assertIsNone(gui_meta["stroke_damage_profile"])
        self.assertEqual(selection["field_weights"], field_weights_for_mode("regular"))
        self.assertEqual(selection["split_fractions"], [0.8, 0.1, 0.1])
        self.assertEqual(gui_meta["metadata_schema_version"], 1)
        semi_selection = semi_meta["effective_selection"]
        self.assertEqual(semi_meta["stroke_damage_profile"]["gap_probability"],
                         config.BROKEN_INK_GAP_PROB)
        self.assertNotIn("character", selection["field_weights"])
        self.assertIn("character", semi_selection["field_weights"])


if __name__ == "__main__":
    unittest.main()
