"""Studio state, tab handling by widget identity, and the preset-bar helpers."""

from __future__ import annotations

import shutil
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest import mock

import config
import gui
from gui_studio import cli_command, section_chips
from src import studio_spec
from src.generation_profiles import AUGMENTATION_PROFILES, CUTOFF_PROFILES
from src.studio_recipe import (ReadOnlyPresetError, RecipeError, StudioState,
                               import_preset, load_recipe, parse_recipe,
                               save_user_preset)
from tests.studio_fakes import studio_app


class StudioStateTests(unittest.TestCase):
    def test_state_and_recipe_round_trip(self):
        state = StudioState.default().with_value(
            studio_spec.PARAMS_BY_PATH["cutoff.top.probability"], 0.4)
        recipe = state.to_recipe("Mine", "desc")
        self.assertEqual(recipe.sections, ("augmentation", "stroke_damage", "cutoff"))
        again = StudioState.default().loaded(recipe, source_name="Mine")
        self.assertEqual(again.sections(), state.sections())
        self.assertFalse(again.dirty)
        partial = state.to_recipe("Cut only", sections=("cutoff",),
                                  run={"sample_mode": "regular", "field_weights": None})
        self.assertEqual(partial.sections, ("cutoff", "run"))
        self.assertNotIn("field_weights", partial.run)

    def test_dirty_tracking_and_labels(self):
        state = StudioState.default()
        self.assertEqual(state.label, "historical_scan_v1")
        blur = studio_spec.PARAMS_BY_PATH["augmentation.blur.range"]
        edited = state.with_value(blur, [0.4, 2.0])
        self.assertEqual(edited.dirty_sections, ("augmentation",))
        self.assertEqual(edited.label, "custom (based on historical_scan_v1)")
        self.assertFalse(edited.with_value(blur, [0.4, 1.8]).dirty)
        base = edited.with_base_profile("scan_noise")
        self.assertEqual(base.source_name, "scan_noise")
        self.assertFalse(base.dirty)
        self.assertEqual(base.augmentation_profile(), AUGMENTATION_PROFILES["scan_noise"])

    def test_profiles_and_baseline_recipe(self):
        state = StudioState.default().with_section(
            "cutoff", {**studio_spec.default_section("cutoff"),
                       "top": {"probability": 0.5, "range": [0.03, 0.08]}})
        self.assertTrue(state.cutoff_profile().active)
        baseline = state.baseline_recipe()
        self.assertEqual(baseline.name, "historical_scan_v1")
        self.assertEqual(baseline.cutoff, StudioState.default().cutoff)
        with self.assertRaisesRegex(ValueError, "cutoff.top.range"):
            state.with_section("cutoff", {"top": {"range": [0.5, 0.1]}})

    def test_builtins_are_read_only_and_bad_imports_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            presets = Path(temporary) / "presets"
            shutil.copytree(gui.PRESET_DIR, presets,
                            ignore=shutil.ignore_patterns("user", "README.md"))
            state = StudioState.default()
            with self.assertRaises(ReadOnlyPresetError):
                save_user_preset(state.to_recipe("Clean Minimal"), "Clean Minimal",
                                 builtin_dir=presets)
            bad = Path(temporary) / "bad.json"
            bad.write_text('{"format": "synthetic-generator.studio-recipe", '
                           '"schema_version": 2, "augmentation": {"fog": {}}}',
                           encoding="utf-8")
            with self.assertRaisesRegex(RecipeError, "augmentation.fog"):
                import_preset(bad, builtin_dir=presets)
            self.assertFalse((presets / "user").exists())


class PresetBarHelperTests(unittest.TestCase):
    def test_section_chips(self):
        self.assertEqual(section_chips(("augmentation", "cutoff", "run")),
                         "[aug]  [cutoff]  [run]")
        self.assertEqual(section_chips(()), "")

    def test_cli_command(self):
        root = config.ROOT
        path = root / "resources" / "presets" / "user" / "mine.json"
        options = {"count": "500", "seed": "7", "mode": "semi_broken_words"}
        self.assertEqual(
            cli_command(path, ("augmentation",), options, root),
            'python -m src.generate_synthetic --recipe "resources/presets/user/mine.json" '
            "--count 500 --seed 7 --mode semi_broken_words")
        self.assertNotIn("--mode", cli_command(path, ("augmentation", "run"), options, root))


class TabIdentityTests(unittest.TestCase):
    def test_tab_checks_use_the_tab_widget_not_its_index(self):
        app = studio_app()
        app.tab_generate, app.tab_studio, app.tab_datasets = object(), object(), object()
        widgets = {"gen": app.tab_generate, "studio": app.tab_studio,
                   "data": app.tab_datasets}
        app.nb = mock.Mock()
        app.nametowidget = lambda path: widgets[path]
        app._refresh_datasets = mock.Mock()
        for selected, refreshes in (("gen", 0), ("studio", 0), ("data", 1)):
            app._refresh_datasets.reset_mock()
            app.nb.select.return_value = selected
            app._on_tab_change()
            self.assertEqual(app._refresh_datasets.call_count, refreshes, selected)
            self.assertIs(app._selected_tab(), widgets[selected])

    def test_wheel_targets_the_scroll_area_under_the_pointer_on_the_visible_tab(self):
        app = studio_app()
        app.nb = mock.Mock()
        app._scroll_areas = [(".nb.gen", ".nb.gen.left"), (".nb.studio", ".nb.studio.area")]
        app.nb.select.return_value = ".nb.studio"
        self.assertEqual(app._scroll_target(".nb.studio.area.frame.slider"), ".nb.studio.area")
        self.assertIsNone(app._scroll_target(".nb.gen.left.frame"))
        self.assertIsNone(app._scroll_target(".nb.studio.areaX"))
        self.assertIsNone(app._scroll_target(None))


class StudioSmokeTests(unittest.TestCase):
    """Optional: real widgets, skipped when Tk has no display."""

    def test_load_edit_and_run_snapshot_with_real_widgets(self):
        try:
            app = gui.App()
        except tk.TclError as exc:
            self.skipTest(f"Tk has no display: {exc}")
        try:
            app.update()
            panel = app.studio_panel
            self.assertIn("historical_faded", panel._entries)
            panel.preset_tree.selection_set("historical_faded")
            panel.load_selected()
            self.assertTrue(app.recipe_summary.cget("text").startswith("Historical Faded Scan"))
            panel.pick_cutoff("heavy")
            app.update()
            self.assertTrue(app.studio_state.dirty)
            self.assertIn("*", app.recipe_summary.cget("text"))
            options = app._snapshot_run_options()
            self.assertEqual(options.edge_clipping, CUTOFF_PROFILES["heavy"])
            self.assertEqual(options.recipe_source, "historical_faded")
            app._set_job_active(True)
            self.assertEqual(str(panel.sliders["augmentation.blur.range"].cget("state")),
                             "disabled")
            app._set_job_active(False)
            self.assertEqual(str(panel.sliders["augmentation.blur.range"].cget("state")),
                             "normal")
            self.assertEqual(
                str(panel.sliders["stroke_damage.gaps.probability"].cget("state")),
                "disabled")
        finally:
            app.destroy()

    def test_recipe_loaded_via_panel_matches_load_recipe(self):
        recipe = load_recipe(gui.PRESET_DIR / "heavy_damage.json")
        state = StudioState.default().loaded(recipe, source_name=recipe.name)
        expected = {k: v for k, v in recipe.augmentation.items() if k != "based_on"}
        saved = parse_recipe(state.to_recipe(recipe.name, recipe.description).to_dict())
        self.assertEqual(saved.augmentation, expected)  # based_on is informational


if __name__ == "__main__":
    unittest.main()
