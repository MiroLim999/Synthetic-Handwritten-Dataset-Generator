"""Unit tests for degradation profiles and character cutoff edge clipping."""

import random
import unittest
from PIL import Image

import config
from src.augment import degrade, _clip_edges, EDGE_CLIPPING_RANGES
from src.generation_profiles import AUGMENTATION_PROFILES, get_augmentation_profile


class TestDegradationAndClipping(unittest.TestCase):
    def test_config_registries(self):
        self.assertTrue(len(EDGE_CLIPPING_RANGES) > 0)
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
        base_img = Image.new("RGB", (200, 100), (255, 255, 255))
        rng = random.Random(42)

        # 'none' should keep exact original dimensions
        clipped_none = _clip_edges(base_img, level="none", rng=rng)
        self.assertEqual(clipped_none.size, (200, 100))

        # 'heavy' clipping should reduce width and height
        clipped_heavy = _clip_edges(base_img, level="heavy", rng=rng)
        self.assertLess(clipped_heavy.width, 200)
        self.assertLess(clipped_heavy.height, 100)

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
            edge_clipping=(0.05, 0.05, 0.05, 0.05),
            rng=random.Random(99),
        )
        self.assertIsInstance(out, Image.Image)

    def test_custom_tuple_edge_clipping(self):
        base_img = Image.new("RGB", (200, 100), (255, 255, 255))
        clipped = _clip_edges(base_img, level=(0.10, 0.10, 0.10, 0.10))
        self.assertEqual(clipped.width, 160)
        self.assertEqual(clipped.height, 80)


if __name__ == "__main__":
    unittest.main()
