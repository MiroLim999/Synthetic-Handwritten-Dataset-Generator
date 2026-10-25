"""Studio Recipe v2: format, migration, apply rules, and preset storage."""

from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path

from src import studio_spec
from src.generation_profiles import (StrokeDamageProfile,
                                     augmentation_profile_from_spec,
                                     create_custom_augmentation_profile,
                                     cutoff_profile_from_spec,
                                     stroke_damage_from_spec)
from src.studio_recipe import (RECIPE_FORMAT, PresetExistsError, RecipeError,
                               ReadOnlyPresetError, StudioRecipe, apply_recipe,
                               compose_settings, default_settings,
                               delete_user_preset, duplicate_preset,
                               export_preset, find_preset, import_preset,
                               list_presets, load_recipe, migrate_v1,
                               parse_recipe, recipe_sha256, rename_user_preset,
                               resolved_recipe, save_user_preset)

PRESET_DIR = Path(__file__).resolve().parents[1] / "resources" / "presets"

# The bundled presets as they were before S4 (flat, single-value sliders).
OLD_HEAVY_DAMAGE = {
    "name": "Heavy Ink Damage & Gaps", "fade": 0.45, "tint": 0.25, "stain": 0.60,
    "rotate": 6.0, "blur": 1.2, "noise": 25.0, "texture": 7.0, "scanline": 15.0,
    "jpeg": 50.0, "crop_top": 2, "crop_bottom": 2, "crop_left": 2, "crop_right": 2,
    "contrast": 0.50, "gap_prob": 0.95, "gap_count": 24.0, "scratch": 0.85,
    "erode": 0.75,
}
OLD_PROFILE_1 = {
    "name": "Profile 1", "fade": 1.0, "tint": 0.18319327731092439,
    "stain": 0.5819327731092437, "rotate": 6.4789915966386555,
    "blur": 0.19537815126050423, "noise": 5.294117647058823,
    "texture": 4.008403361344538, "scanline": 8.193277310924369,
    "jpeg": 66.21848739495798, "crop_top": 0.0, "crop_bottom": 0.0,
    "crop_left": 0.0, "crop_right": 0.0, "contrast": 1.0,
    "gap_prob": 0.07983193277310924, "gap_count": 1.3865546218487395,
    "scratch": 0.0861344537815126, "erode": 0.6260504201680672,
}


def _recipe(**sections) -> dict:
    return {"format": RECIPE_FORMAT, "schema_version": 2, "name": "T", **sections}


class RecipeFormatTests(unittest.TestCase):
    def test_each_section_round_trips(self):
        sections = {
            "augmentation": studio_spec.default_section("augmentation"),
            "stroke_damage": studio_spec.default_section("stroke_damage"),
            "cutoff": studio_spec.default_section("cutoff"),
            "run": {"sample_mode": "semi_broken_mixed", "split_fractions": [0.7, 0.2, 0.1],
                    "field_weights": {"place": 2, "age": 0}, "samples_per_writer": 8},
        }
        for name, value in sections.items():
            recipe = parse_recipe(_recipe(**{name: value}, description="d"))
            self.assertEqual(recipe.sections, (name,))
            again = parse_recipe(json.loads(json.dumps(recipe.to_dict())))
            self.assertEqual(again, recipe)
            self.assertEqual(recipe_sha256(again), recipe_sha256(recipe))
        full = parse_recipe(_recipe(**sections))
        changed = parse_recipe(_recipe(**{**sections, "run": {"sample_mode": "regular"}}))
        self.assertNotEqual(recipe_sha256(full), recipe_sha256(changed))

    def test_validation_errors_name_the_json_path(self):
        with self.assertRaises(RecipeError) as caught:
            parse_recipe(_recipe(
                augmentation={"blur": {"range": [3, 1]}, "fog": {}},
                stroke_damage={"gaps": {"count_range": [1, 99]}},
                cutoff={"top": {"probability": 2}},
                run={"sample_mode": "nope", "field_weights": {"zz": 1},
                     "split_fractions": [0.5, 0.5, 0.5], "extra": True},
                surprise=1,
            ))
        joined = "\n".join(caught.exception.errors)
        for path in ("augmentation.blur.range", "augmentation.fog",
                     "stroke_damage.gaps.count_range", "cutoff.top.probability",
                     "run.sample_mode", "run.field_weights.zz", "run.split_fractions",
                     "run.extra", "surprise"):
            self.assertIn(path, joined)

    def test_newer_schema_and_bad_headers_are_rejected(self):
        with self.assertRaisesRegex(RecipeError, "newer generator"):
            parse_recipe({"format": RECIPE_FORMAT, "schema_version": 3})
        with self.assertRaisesRegex(RecipeError, "format"):
            parse_recipe({"format": "other", "schema_version": 2})
        with self.assertRaises(RecipeError):
            parse_recipe(["not", "an", "object"])


class MigrationTests(unittest.TestCase):
    def test_every_bundled_preset_loads_as_v2(self):
        paths = sorted(PRESET_DIR.glob("*.json"))
        self.assertGreaterEqual(len(paths), 5)
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["schema_version"], 2, path.name)
            recipe = load_recipe(path)
            self.assertFalse(recipe.migrated_from_v1)
            augmentation_profile_from_spec(recipe.augmentation)
            stroke_damage_from_spec(recipe.stroke_damage)
            cutoff_profile_from_spec(recipe.cutoff)

    def test_migrated_values_reproduce_the_old_profiles_exactly(self):
        for old in (OLD_HEAVY_DAMAGE, OLD_PROFILE_1):
            recipe = migrate_v1(old)
            self.assertTrue(recipe.migrated_from_v1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                expected = create_custom_augmentation_profile(
                    fade_contrast=old["fade"], paper_tint_alpha=old["tint"],
                    stain_prob=old["stain"], rotate_deg=old["rotate"],
                    blur_radius=old["blur"], noise_std=old["noise"],
                    paper_texture_std=old["texture"],
                    scanline_alpha=int(round(old["scanline"])),
                    jpeg_quality=int(round(old["jpeg"])))
            actual = augmentation_profile_from_spec(recipe.augmentation)
            self.assertEqual(replace(actual, profile_id="x"),
                             replace(expected, profile_id="x"))
            legacy_stroke = StrokeDamageProfile.from_legacy_params({
                "contrast_prob": old["contrast"], "gap_prob": old["gap_prob"],
                "gap_count": old["gap_count"], "scratch_prob": old["scratch"],
                "erode_prob": old["erode"]})
            self.assertEqual(stroke_damage_from_spec(recipe.stroke_damage), legacy_stroke)

    def test_crop_percentages_become_ink_relative_fractions(self):
        cutoff = cutoff_profile_from_spec(migrate_v1(OLD_HEAVY_DAMAGE).cutoff)
        for _name, side in cutoff.sides():
            self.assertEqual((side.probability, side.fraction_range), (1.0, (0.02, 0.02)))
        none = cutoff_profile_from_spec(migrate_v1(OLD_PROFILE_1).cutoff)
        self.assertFalse(none.active)

    def test_unknown_or_malformed_flat_values_are_rejected(self):
        with self.assertRaisesRegex(RecipeError, "mystery"):
            migrate_v1({"fade": 0.5, "mystery": 1})
        with self.assertRaisesRegex(RecipeError, "fade"):
            parse_recipe({"fade": "faded"})


class ApplyRuleTests(unittest.TestCase):
    def test_only_present_sections_are_applied(self):
        base = default_settings()
        cutoff = studio_spec.default_section("cutoff")
        cutoff["top"] = {"probability": 0.5, "range": [0.03, 0.08]}
        applied = apply_recipe(base, parse_recipe(_recipe(cutoff=cutoff)))
        self.assertEqual(applied["cutoff"]["top"]["probability"], 0.5)
        for section in ("augmentation", "stroke_damage", "run"):
            self.assertEqual(applied[section], base[section], section)
        self.assertNotIn("based_on", apply_recipe(
            base, parse_recipe(_recipe(cutoff={**cutoff, "based_on": "x"})))["cutoff"])

    def test_run_keys_merge_and_mode_resets_the_field_mix(self):
        base = default_settings()
        base["run"]["field_weights"] = {"place": 1}
        applied = apply_recipe(base, parse_recipe(_recipe(
            run={"sample_mode": "semi_broken_characters"})))
        self.assertEqual(applied["run"]["sample_mode"], "semi_broken_characters")
        self.assertIsNone(applied["run"]["field_weights"])
        self.assertEqual(applied["run"]["samples_per_writer"], 32)

    def test_precedence_defaults_recipe_overrides(self):
        recipe = parse_recipe(_recipe(run={"samples_per_writer": 8,
                                           "sample_mode": "semi_broken_words"}))
        settings = compose_settings(recipe, overrides={"run": {"samples_per_writer": 16}})
        self.assertEqual(settings["run"]["samples_per_writer"], 16)
        self.assertEqual(settings["run"]["sample_mode"], "semi_broken_words")
        self.assertEqual(compose_settings()["run"]["samples_per_writer"], 32)
        with self.assertRaisesRegex(RecipeError, "run.samples_per_writer"):
            compose_settings(overrides={"run": {"samples_per_writer": 0}})

    def test_resolved_recipe_has_every_section(self):
        recipe = resolved_recipe(default_settings(), name="all")
        self.assertEqual(recipe.sections,
                         ("augmentation", "stroke_damage", "cutoff", "run"))
        self.assertNotIn("field_weights", recipe.run)


class PresetStorageTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.builtin = Path(self._temporary.name) / "presets"
        self.builtin.mkdir()
        self.dirs = {"builtin_dir": self.builtin}
        self.recipe = parse_recipe(_recipe(cutoff=studio_spec.default_section("cutoff")))
        (self.builtin / "ledger.json").write_text(
            json.dumps(_recipe(name="Ledger")), encoding="utf-8")
        (self.builtin / "broken.json").write_text("{", encoding="utf-8")

    def test_listing_flags_builtin_and_user_presets(self):
        save_user_preset(self.recipe, "Mine", **self.dirs)
        entries = {entry.key: entry for entry in list_presets(**self.dirs)}
        self.assertEqual(list(entries), ["broken", "ledger", "user/mine"])
        self.assertTrue(entries["ledger"].builtin)
        self.assertFalse(entries["user/mine"].builtin)
        self.assertEqual(entries["user/mine"].sections, ("cutoff",))
        self.assertIsNotNone(entries["broken"].error)

    def test_builtins_cannot_be_overwritten_renamed_or_deleted(self):
        with self.assertRaises(ReadOnlyPresetError):
            save_user_preset(self.recipe, "Ledger", **self.dirs)
        ledger = find_preset("ledger", **self.dirs)
        with self.assertRaises(ReadOnlyPresetError):
            delete_user_preset(ledger)
        with self.assertRaises(ReadOnlyPresetError):
            rename_user_preset(ledger, "x", **self.dirs)
        self.assertEqual(json.loads(ledger.path.read_text(encoding="utf-8"))["name"],
                         "Ledger")

    def test_user_preset_lifecycle(self):
        path = save_user_preset(self.recipe, "Mine", **self.dirs)
        with self.assertRaises(PresetExistsError):
            save_user_preset(self.recipe, "mine", **self.dirs)
        save_user_preset(replace(self.recipe, description="v2"), "Mine",
                         overwrite=True, **self.dirs)
        self.assertEqual(load_recipe(path).description, "v2")

        copy_path = duplicate_preset(find_preset("ledger", **self.dirs), "Ledger copy",
                                     **self.dirs)
        self.assertEqual(load_recipe(copy_path).name, "Ledger copy")
        renamed = rename_user_preset(find_preset("user/mine", **self.dirs), "Renamed",
                                     **self.dirs)
        self.assertFalse(path.exists())
        self.assertTrue(renamed.exists())

        exported = export_preset(find_preset("user/renamed", **self.dirs),
                                 Path(self._temporary.name) / "out.json")
        imported = import_preset(exported, name="Imported", **self.dirs)
        self.assertEqual(load_recipe(imported).description, "v2")
        delete_user_preset(find_preset("user/renamed", **self.dirs))
        self.assertEqual(
            [entry.key for entry in list_presets(**self.dirs) if not entry.builtin],
            ["user/imported", "user/ledger_copy"])

    def test_bad_imports_are_rejected(self):
        bad = Path(self._temporary.name) / "bad.json"
        bad.write_text(json.dumps(_recipe(cutoff={"top": {"probability": 7}})),
                       encoding="utf-8")
        with self.assertRaisesRegex(RecipeError, "cutoff.top.probability"):
            import_preset(bad, **self.dirs)
        self.assertFalse((self.builtin / "user").exists())

    def test_recipe_dataclass_is_frozen(self):
        with self.assertRaises(AttributeError):
            self.recipe.name = "x"  # type: ignore[misc]
        self.assertIsInstance(self.recipe, StudioRecipe)


if __name__ == "__main__":
    unittest.main()
