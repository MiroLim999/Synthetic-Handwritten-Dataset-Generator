"""Unit tests for degradation profiles and character cutoff edge clipping."""

import random
import unittest
from PIL import Image, ImageDraw

import config
from src.augment import degrade, degrade_detailed
from src.generation_profiles import (AUGMENTATION_PROFILES, CUTOFF_PROFILES,
                                     CutoffProfile, CutoffSide,
                                     get_augmentation_profile)


class TestDegradationAndClipping(unittest.TestCase):
    def test_config_registries(self):
        self.assertEqual(set(CUTOFF_PROFILES) | {"custom"},
                         set(config.EDGE_CLIPPING_OPTIONS))
        self.assertTrue(len(AUGMENTATION_PROFILES) > 0)
        self.assertIn("historical_scan_v1", config.DEGRADATION_PROFILES)
        self.assertIn("faded_ink", config.DEGRADATION_PROFILES)
        self.assertIn("bleed_through", config.DEGRADATION_PROFILES)
        self.assertIn("scan_noise", config.DEGRADATION_PROFILES)
        self.assertIn("heavy_smudge", config.DEGRADATION_PROFILES)
        self.assertIn("clean", config.DEGRADATION_PROFILES)

        self.assertIn("none", config.EDGE_CLIPPING_OPTIONS)
        self.assertIn("light", config.EDGE_CLIPPING_OPTIONS)
        self.assertIn("moderate", config.EDGE_CLIPPING_OPTIONS)
        self.assertIn("heavy", config.EDGE_CLIPPING_OPTIONS)

    def test_augmentation_profiles_registered(self):
        for profile_id in ("historical_scan_v1", "faded_ink", "bleed_through",
                           "scan_noise", "heavy_smudge", "clean"):
            prof = get_augmentation_profile(profile_id)
            self.assertEqual(prof.profile_id, profile_id)
            self.assertIn("profile_schema_version", prof.to_metadata())

    def test_edge_clipping_dimensions(self):
        always_top = CutoffProfile(
            profile_id="always_top", top=CutoffSide(1.0, (0.20, 0.20)),
            max_ink_loss=0.5)

        # A blank image has no ink, so it is never cut.
        blank = Image.new("RGB", (200, 100), (255, 255, 255))
        result = degrade_detailed(blank, augmentation_profile="clean",
                                  edge_clipping=always_top, rng=random.Random(42))
        self.assertEqual(result.image.size, (200, 100))
        self.assertFalse(result.cutoff.is_cut)

        # With ink, the cut is measured from the ink box, not the canvas:
        # 20% of a 40 px tall ink box removes 8 px of ink plus the padding.
        inked = blank.copy()
        ImageDraw.Draw(inked).rectangle((40, 30, 159, 69), fill="black")
        result = degrade_detailed(inked, augmentation_profile="clean",
                                  edge_clipping=always_top, rng=random.Random(42))
        self.assertEqual(result.image.size, (200, 100 - 30 - 8))
        self.assertEqual(result.cutoff.cut_sides, frozenset({"top"}))
        self.assertAlmostEqual(result.cutoff.top, 0.20)
        self.assertAlmostEqual(result.cutoff.ink_loss, 0.20)

    def test_degrade_all_profiles(self):
        base_img = Image.new("RGB", (180, 60), (240, 240, 240))
        rng = random.Random(123)

        for profile_id in ("historical_scan_v1", "faded_ink", "bleed_through",
                           "scan_noise", "heavy_smudge", "clean"):
            for clipping in ("none", "light", "moderate", "heavy"):
                out = degrade(
                    base_img,
                    augmentation_profile=profile_id,
                    edge_clipping=clipping,
                    rng=rng,
                )
                self.assertIsInstance(out, Image.Image)
                self.assertGreater(out.width, 0)
                self.assertGreater(out.height, 0)


    def test_create_custom_augmentation_profile(self):
        from src.generation_profiles import create_custom_augmentation_profile
        with self.assertWarns(DeprecationWarning):
            custom_prof = create_custom_augmentation_profile(
                fade_contrast=0.40,
                paper_tint_alpha=0.25,
                stain_prob=0.80,
                rotate_deg=8.0,
                blur_radius=1.5,
                noise_std=25.0,
                paper_texture_std=8.0,
                scanline_alpha=18,
                jpeg_quality=50,
            )
        self.assertEqual(custom_prof.profile_id, "custom_dev_v1")
        self.assertEqual(custom_prof.stain_probability, 0.80)
        self.assertEqual(custom_prof.rotate_degrees, 8.0)

        base_img = Image.new("RGB", (180, 60), (240, 240, 240))
        out = degrade(
            base_img,
            augmentation_profile=custom_prof,
            edge_clipping=(0.05, 0.05, 0.05, 0.05),  # legacy tuple -> custom profile
            rng=random.Random(99),
        )
        self.assertIsInstance(out, Image.Image)

    def test_custom_semi_broken_params(self):
        base_img = Image.new("RGB", (200, 100), (255, 255, 255))
        custom_params = {
            "gap_prob": 1.0,
            "gap_count": 20,
            "scratch_prob": 1.0,
            "erode_prob": 1.0,
        }
        out = degrade(
            base_img,
            damage_profile="semi_broken",
            semi_broken_params=custom_params,
            rng=random.Random(42),
        )
        self.assertIsInstance(out, Image.Image)


if __name__ == "__main__":
    unittest.main()
