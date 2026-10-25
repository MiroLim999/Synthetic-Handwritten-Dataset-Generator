"""Regression tests for the S1 Developer Studio correctness fixes.

The S1 guarantees still hold after the Studio tab rebuild; these tests check
them against the studio state rather than the old single-value sliders.
"""

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
from src import studio_spec
from src.augment import degrade
from src.generate_synthetic import field_weights_for_mode, generate
from src.generation_io import iter_generation_plan
from src.generation_profiles import stroke_damage_from_config
from src.studio_recipe import (RECIPE_FORMAT, StudioState, load_recipe,
                               migrate_v1, parse_recipe)
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
            self.assertEqual(profile, stroke_damage_from_config(), mode)

    def test_stroke_controls_follow_the_mode_and_respect_the_run_lock(self):
        app = studio_app("regular")
        app.studio_panel = mock.Mock()
        app._sync_stroke_damage_state()
        app.studio_panel.set_stroke_enabled.assert_called_with(False)
        app.sample_mode_var.set(config.SAMPLE_MODES["semi_broken_words"])
        app._sync_stroke_damage_state()
        app.studio_panel.set_stroke_enabled.assert_called_with(True)
        # While a job holds the lock, the restore path owns widget states.
        app._run_control_states = [(object(), "normal")]
        app.studio_panel.reset_mock()
        app._sync_stroke_damage_state()
        app.studio_panel.set_stroke_enabled.assert_not_called()


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
    def test_loading_applies_only_the_sections_a_preset_contains(self):
        app = studio_app("regular")
        cutoff = studio_spec.default_section("cutoff")
        cutoff["top"] = {"probability": 1.0, "range": [0.03, 0.03]}
        before = app.studio_state
        app._apply_loaded_recipe("only_cutoff", parse_recipe({
            "format": RECIPE_FORMAT, "schema_version": 2, "name": "Cut",
            "cutoff": cutoff, "run": {"sample_mode": "semi_broken_words"}}))
        self.assertEqual(app.studio_state.augmentation, before.augmentation)
        self.assertEqual(app.studio_state.cutoff_profile().top.fraction_range, (0.03, 0.03))
        self.assertEqual(app._current_sample_mode(), "semi_broken_words")
        self.assertFalse(app.studio_state.dirty)
        self.assertEqual(app.studio_state.source_key, "only_cutoff")
        self.assertIn("cutoff, run", app.status.config.call_args.kwargs["text"])

    def test_old_flat_preset_is_migrated_and_clamped(self):
        app = studio_app()
        app._apply_loaded_recipe("old", migrate_v1(
            {"name": "X", "tint": 0.9, "jpeg": 66.4, "crop_top": 3}))
        options = app._snapshot_run_options()
        self.assertEqual(options.augmentation_profile.paper_tint_alpha_range, (0.5, 0.5))
        self.assertEqual(options.augmentation_profile.jpeg_quality_range, (66, 66))
        self.assertEqual(options.edge_clipping.top.fraction_range, (0.03, 0.03))

    def test_malformed_presets_are_rejected_before_anything_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            bad = Path(temporary) / "bad.json"
            for content in ({"tint": 0.3, "fade": "faded"}, {"tint": 0.2, "mystery": 1},
                            {"fade": float("nan")}):
                bad.write_text(json.dumps(content), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_recipe(bad)

    def test_editing_after_load_marks_the_recipe_custom(self):
        state = StudioState.default().loaded(
            load_recipe(gui.PRESET_DIR / "historical_faded.json"),
            source_name="Historical Faded Scan", source_key="historical_faded")
        self.assertEqual(state.label, "Historical Faded Scan")
        edited = state.with_value(studio_spec.PARAMS_BY_PATH["augmentation.blur.range"],
                                  [0.5, 1.0])
        self.assertEqual(edited.label, "custom (based on Historical Faded Scan)")
        self.assertEqual(edited.dirty_sections, ("augmentation",))


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

    def test_every_spec_default_is_on_its_step(self):
        for spec in studio_spec.PARAMS:
            values = spec.default if spec.kind == "range" else (spec.default,)
            for value in values:
                self.assertEqual(
                    gui.snap_value(value, spec.step, spec.low, spec.high), value, spec.path)


class ResetScopeTests(unittest.TestCase):
    def test_section_reset_returns_to_the_loaded_source_only(self):
        state = StudioState.default()
        blur = studio_spec.PARAMS_BY_PATH["augmentation.blur.range"]
        top = studio_spec.PARAMS_BY_PATH["cutoff.top.probability"]
        edited = state.with_value(blur, [0.5, 1.0]).with_value(top, 0.5)
        reset = edited.reset_section("augmentation", [blur])
        self.assertEqual(reset.value(blur), [0.4, 1.8])
        self.assertEqual(reset.value(top), 0.5)

    def test_reset_all_also_resets_weights_splits_and_studio(self):
        app = studio_app()
        app.split_train_var.set("70")
        app.field_weight_vars["age"].set("77")
        app.count_var.set("5")
        app.studio_state = app.studio_state.with_base_profile("scan_noise")
        app.cursive_row = mock.Mock()
        with mock.patch.object(gui.App, "_on_font_style_change"):
            app._reset_all_controls()
        self.assertEqual(app.split_train_var.get(), "80")
        self.assertEqual(app.field_weight_vars["age"].get(), "5")
        self.assertEqual(app.count_var.get(), str(config.DEFAULT_COUNT))
        self.assertEqual(app.studio_state, StudioState.default())


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
                workers=options.workers, recipe=options.recipe,
                recipe_source=options.recipe_source,
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
        self.assertEqual(gui_meta["studio_recipe"]["source_preset"], "historical_scan_v1")
        self.assertFalse(gui_meta["studio_recipe"]["modified"])
        semi_selection = semi_meta["effective_selection"]
        self.assertEqual(semi_meta["stroke_damage_profile"]["gap_probability"],
                         config.BROKEN_INK_GAP_PROB)
        self.assertNotIn("character", selection["field_weights"])
        self.assertIn("character", semi_selection["field_weights"])


if __name__ == "__main__":
    unittest.main()
