import random
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src import render
from src.augment import degrade
from src.generation_profiles import (
    DEFAULT_AUGMENTATION_PROFILE_ID,
    DEFAULT_WRITER_PROFILE_ID,
    PROFILE_SCHEMA_VERSION,
    get_augmentation_profile,
    writer_style_for,
)


def _source_image():
    image = Image.new("RGB", (150, 55), "white")
    ImageDraw.Draw(image).text((10, 10), "Maasin 1948", fill="black")
    return image


class GenerationProfileTests(unittest.TestCase):
    def tearDown(self):
        render.clear_font_cache()

    def test_writer_style_is_stable_and_does_not_consume_global_rng(self):
        random.seed(1)
        first = writer_style_for("writer-004", 2024)
        random.seed(999_999)
        second = writer_style_for("writer-004", 2024)

        self.assertEqual(first, second)
        self.assertNotEqual(first, writer_style_for("writer-005", 2024))
        self.assertNotEqual(first, writer_style_for("writer-004", 2025))
        self.assertEqual(first.profile_id, DEFAULT_WRITER_PROFILE_ID)
        self.assertEqual(
            first.to_metadata()["profile_schema_version"], PROFILE_SCHEMA_VERSION
        )

    def test_unknown_profiles_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "Unknown writer-style profile"):
            writer_style_for("writer", 1, "missing")
        with self.assertRaisesRegex(ValueError, "Unknown augmentation profile"):
            get_augmentation_profile("missing")

    def test_font_objects_are_cached_by_normalized_path_and_size(self):
        fallback = ImageFont.load_default()
        with mock.patch.object(render.ImageFont, "truetype", return_value=fallback) as load:
            first = render.load_font("fake-font.ttf", 42)
            second = render.load_font("fake-font.ttf", 42)
            third = render.load_font("fake-font.ttf", 43)

        self.assertIs(first, second)
        self.assertIsNotNone(third)
        self.assertEqual(load.call_count, 2)
        self.assertEqual(render.font_cache_info().hits, 1)

    def test_writer_render_is_byte_stable_with_explicit_rng(self):
        fallback = ImageFont.load_default()
        style = writer_style_for("writer-11", 77)

        def render_once(global_seed):
            random.seed(global_seed)
            render.clear_font_cache()
            with (mock.patch.object(
                    render, "fonts_for_style",
                    return_value=("one.ttf", "two.ttf", "three.ttf")),
                  mock.patch.object(
                      render.ImageFont, "truetype", return_value=fallback)):
                image, font = render.render_text(
                    "Maria Santos", rng=random.Random(808), writer_style=style
                )
            return image.mode, image.size, image.tobytes(), font

        self.assertEqual(render_once(1), render_once(999_999))

    def test_writer_selects_one_font_inside_the_supplied_split_pool(self):
        fallback = ImageFont.load_default()
        style = writer_style_for("writer-12", 88)
        pool = ("heldout-a.ttf", "heldout-b.ttf")
        render.clear_font_cache()
        with mock.patch.object(render.ImageFont, "truetype", return_value=fallback):
            _, first = render.render_text(
                "First", rng=random.Random(1), writer_style=style, font_pool=pool
            )
            _, second = render.render_text(
                "Second", rng=random.Random(999), writer_style=style, font_pool=pool
            )
        self.assertEqual(first, second)
        self.assertIn(first, {"heldout-a.ttf", "heldout-b.ttf"})

    def test_versioned_augmentation_is_reproducible(self):
        base = get_augmentation_profile(DEFAULT_AUGMENTATION_PROFILE_ID)
        profile = replace(
            base,
            fade_probability=1.0,
            brightness_probability=1.0,
            paper_tint_probability=1.0,
            stain_probability=1.0,
            rotate_probability=1.0,
            blur_probability=1.0,
            noise_probability=1.0,
            paper_texture_probability=1.0,
            scanline_probability=1.0,
            compression_probability=1.0,
        )

        def augmented(global_seed, sample_seed=123):
            random.seed(global_seed)
            np.random.seed(global_seed)
            image = degrade(
                _source_image(),
                rng=random.Random(sample_seed),
                np_rng=np.random.default_rng(sample_seed),
                augmentation_profile=profile,
            )
            return image.mode, image.size, image.tobytes()

        self.assertEqual(augmented(1), augmented(999_999))
        self.assertNotEqual(augmented(1), augmented(1, sample_seed=124))


if __name__ == "__main__":
    unittest.main()
