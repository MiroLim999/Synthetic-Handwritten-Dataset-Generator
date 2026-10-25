"""Spec <-> AugmentationProfile conversion, bounds, and content IDs."""

from __future__ import annotations

import copy
import random
import subprocess
import sys
import unittest
import warnings
from dataclasses import asdict, replace

from PIL import Image, ImageDraw

import config
from src import studio_spec
from src.augment import _rotate
from src.generation_profiles import (AUGMENTATION_PROFILES,
                                     PROFILE_SCHEMA_VERSION,
                                     AugmentationProfile,
                                     augmentation_profile_from_spec,
                                     augmentation_spec_from_legacy_values,
                                     augmentation_spec_from_profile,
                                     create_custom_augmentation_profile)


def _legacy(**values):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return create_custom_augmentation_profile(**values)


class AugmentationSpecRoundTripTests(unittest.TestCase):
    def test_registered_profiles_round_trip_and_keep_their_id(self):
        for profile_id, profile in AUGMENTATION_PROFILES.items():
            spec = augmentation_spec_from_profile(profile)
            rebuilt = augmentation_profile_from_spec(spec)
            self.assertEqual(rebuilt, profile, profile_id)
            self.assertEqual(rebuilt.profile_id, profile_id)
            self.assertEqual(augmentation_spec_from_profile(rebuilt), spec)

    def test_registered_profiles_lie_inside_the_spec_bounds(self):
        for profile_id, profile in AUGMENTATION_PROFILES.items():
            _normalized, errors = studio_spec.validate_section(
                "augmentation", augmentation_spec_from_profile(profile))
            self.assertEqual(errors, [], profile_id)

    def test_bounds_cover_the_old_custom_profile_clamps(self):
        expected = {
            "augmentation.paper_tint.range": (0.0, 0.50),
            "augmentation.rotation.range": (0.0, 15.0),
            "augmentation.blur.range": (0.0, 5.0),
            "augmentation.noise.range": (0.0, 50.0),
            "augmentation.paper_texture.range": (0.0, 20.0),
            "augmentation.scanlines.range": (0, 60),
            "augmentation.jpeg.range": (10, 100),
        }
        for path, (low, high) in expected.items():
            spec = studio_spec.PARAMS_BY_PATH[path]
            self.assertLessEqual(spec.low, low, path)
            self.assertGreaterEqual(spec.high, high, path)

    def test_integer_params_use_step_one_and_are_stored_as_ints(self):
        for path in ("augmentation.scanlines.range", "augmentation.jpeg.range",
                     "augmentation.stains.count_range",
                     "augmentation.stains.opacity_range",
                     "stroke_damage.gaps.count_range",
                     "stroke_damage.scratches.count_range"):
            spec = studio_spec.PARAMS_BY_PATH[path]
            self.assertEqual(spec.step, 1, path)
            normalized, errors = studio_spec.validate_section(
                spec.section,
                {spec.effect: {spec.keys[1]: [float(spec.low), float(spec.low + 1)]}})
            self.assertEqual(errors, [])
            value = studio_spec.get_value(normalized, spec)
            self.assertTrue(all(type(item) is int for item in value), path)

    def test_every_default_is_valid(self):
        for section in studio_spec.SECTIONS:
            _normalized, errors = studio_spec.validate_section(
                section, studio_spec.default_section(section))
            self.assertEqual(errors, [], section)


class ContentIdTests(unittest.TestCase):
    def test_ids_are_stable_across_processes(self):
        spec = augmentation_spec_from_legacy_values(fade_contrast=0.4, noise_std=20)
        local = augmentation_profile_from_spec(spec).profile_id
        code = (
            "from src.generation_profiles import *\n"
            "from src.generation_profiles import augmentation_profile_from_spec, "
            "augmentation_spec_from_legacy_values\n"
            "print(augmentation_profile_from_spec(augmentation_spec_from_legacy_values("
            "fade_contrast=0.4, noise_std=20)).profile_id)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=str(config.ROOT),
            capture_output=True, text=True, check=True,
            env={"PYTHONHASHSEED": "123", **_minimal_env()},
        )
        self.assertEqual(result.stdout.strip(), local)
        self.assertRegex(local, r"^custom_[0-9a-f]{12}$")

    def test_ids_change_whenever_any_value_changes(self):
        base_spec = augmentation_spec_from_profile("historical_scan_v1")
        base = augmentation_profile_from_spec(base_spec)
        seen = {base.profile_id}
        for param in studio_spec.params_for_section("augmentation"):
            spec = copy.deepcopy(base_spec)
            value = studio_spec.get_value(spec, param)
            if param.kind == "range":
                low, high = value
                changed = ([low, high - param.step] if high - param.step >= low
                           else [low, high + param.step])
            else:
                changed = value - param.step if value - param.step >= param.low else value + param.step
            studio_spec.set_value(spec, param, changed)
            profile_id = augmentation_profile_from_spec(spec).profile_id
            self.assertNotIn(profile_id, seen, param.path)
            seen.add(profile_id)

    def test_matching_values_reuse_the_registered_id(self):
        spec = augmentation_spec_from_profile("scan_noise")
        spec["based_on"] = "something else"
        self.assertEqual(augmentation_profile_from_spec(spec).profile_id, "scan_noise")


class LegacyBuilderTests(unittest.TestCase):
    def test_interim_gui_builder_equals_the_old_custom_profile(self):
        cases = [
            {},
            {"fade_contrast": 0.40, "paper_tint_alpha": 0.25, "stain_prob": 0.80,
             "rotate_deg": 8.0, "blur_radius": 1.5, "noise_std": 25.0,
             "paper_texture_std": 8.0, "scanline_alpha": 18, "jpeg_quality": 50},
            {"fade_contrast": 0.99, "paper_tint_alpha": 0.0, "stain_prob": 0.0,
             "rotate_deg": 0.0, "blur_radius": 0.0, "noise_std": 0.0,
             "paper_texture_std": 0.0, "scanline_alpha": 0, "jpeg_quality": 100},
            {"fade_contrast": 0.01, "paper_tint_alpha": 9, "rotate_deg": 99,
             "blur_radius": 9, "noise_std": 99, "paper_texture_std": 99,
             "scanline_alpha": 99, "jpeg_quality": 1},
        ]
        for values in cases:
            new = augmentation_profile_from_spec(
                augmentation_spec_from_legacy_values(**values))
            old = _legacy(**values)
            self.assertEqual(replace(new, profile_id="x"), replace(old, profile_id="x"),
                             values)
            self.assertNotEqual(new.profile_id, "custom_dev_v1")

    def test_old_builder_still_works_and_warns(self):
        with self.assertWarns(DeprecationWarning):
            profile = create_custom_augmentation_profile(rotate_deg=3.0)
        self.assertEqual(profile.profile_id, "custom_dev_v1")
        self.assertEqual(profile.rotate_degrees, 3.0)


class ExtendedProfileTests(unittest.TestCase):
    def test_metadata_versions(self):
        metadata = AUGMENTATION_PROFILES["clean"].to_metadata()
        self.assertEqual(metadata["augmentation_schema_version"], 2)
        self.assertEqual(metadata["profile_schema_version"], PROFILE_SCHEMA_VERSION)
        self.assertEqual(PROFILE_SCHEMA_VERSION, 1)
        for key in ("rotate_min_degrees", "stain_count_range", "stain_opacity_range"):
            self.assertIn(key, metadata)

    def test_new_fields_are_validated(self):
        base = asdict(AUGMENTATION_PROFILES["historical_scan_v1"])
        for change in ({"rotate_min_degrees": 5.0},  # above rotate_degrees (4.5)
                       {"rotate_min_degrees": -1.0},
                       {"stain_count_range": (3, 2)},
                       {"stain_count_range": (1.5, 2)},
                       {"stain_opacity_range": (0, 300)}):
            with self.assertRaises(ValueError, msg=str(change)):
                AugmentationProfile(**{**base, **change})

    def test_minimum_tilt_draws_a_magnitude_then_a_sign(self):
        class _Recorder:
            def rotate(self, angle, **_kwargs):
                self.angle = angle
                return self

        signs = set()
        for seed in range(40):
            expected = random.Random(seed)
            magnitude = expected.uniform(2.0, 4.0)
            sign = -1 if expected.random() < 0.5 else 1
            recorded = _rotate(_Recorder(), rng=random.Random(seed),
                               degrees=4.0, min_degrees=2.0).angle
            self.assertEqual(recorded, sign * magnitude)
            self.assertTrue(2.0 <= abs(recorded) <= 4.0)
            signs.add(sign)
        self.assertEqual(signs, {-1, 1})

    def test_zero_minimum_tilt_keeps_the_single_legacy_draw(self):
        image = Image.new("RGB", (120, 40), "white")
        ImageDraw.Draw(image).rectangle((10, 15, 110, 25), fill="black")
        rng = random.Random(3)
        _rotate(image, rng=rng, degrees=4.5)
        legacy = random.Random(3)
        legacy.uniform(-4.5, 4.5)
        self.assertEqual(rng.random(), legacy.random())

    def test_validation_errors_name_their_json_path(self):
        _normalized, errors = studio_spec.validate_section("augmentation", {
            "blur": {"probability": 1.5, "range": [2.0, 1.0], "radius": 3},
            "fog": {},
            "rotation": {"range": [0, 99]},
        })
        joined = "\n".join(errors)
        for path in ("augmentation.blur.probability", "augmentation.blur.range",
                     "augmentation.blur.radius", "augmentation.fog",
                     "augmentation.rotation.range"):
            self.assertIn(path, joined)
        with self.assertRaisesRegex(ValueError, "augmentation.blur.range"):
            augmentation_profile_from_spec({"blur": {"range": [2.0, 1.0]}})


def _minimal_env():
    import os

    keep = ("PATH", "SYSTEMROOT", "TEMP", "TMP", "HOME", "USERPROFILE")
    return {key: os.environ[key] for key in keep if key in os.environ}


if __name__ == "__main__":
    unittest.main()
