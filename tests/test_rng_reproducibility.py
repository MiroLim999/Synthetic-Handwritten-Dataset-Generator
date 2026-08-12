import random
import unittest
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
from src import fields, render
from src.augment import degrade


def _field_sequence(seed):
    rng = random.Random(seed)
    labels = (
        fields.make_value(field_type, rng=rng)
        for field_type in sorted(fields.GENERATORS)
        for _ in range(4)
    )
    return "\n".join(labels).encode("utf-8")


def _rendered(seed):
    fallback_font = ImageFont.load_default()
    with (mock.patch.object(
            render, "fonts_for_style",
            return_value=("alpha.ttf", "beta.ttf")),
          mock.patch.object(
              render.ImageFont, "truetype", return_value=fallback_font)):
        image, font_used = render.render_text(
            "Juan Dela Cruz", rng=random.Random(seed))
    return image.mode, image.size, image.tobytes(), font_used


def _source_image():
    image = Image.new("RGB", (160, 60), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 20, 145, 40), fill="black")
    return image


def _degraded(seed):
    # Force every branch, including NumPy-backed noise, so this covers the
    # entire explicit-RNG call graph rather than depending on current defaults.
    with mock.patch.multiple(
            config,
            AUG_FADE_PROB=1.0,
            AUG_BRIGHTNESS_PROB=1.0,
            AUG_PAPER_TINT_PROB=1.0,
            AUG_STAIN_PROB=1.0,
            AUG_ROTATE_PROB=1.0,
            AUG_BLUR_PROB=1.0,
            AUG_NOISE_PROB=1.0,
            BROKEN_CONTRAST_PROB=1.0,
            BROKEN_ERODE_PROB=1.0,
            BROKEN_INK_GAP_PROB=1.0,
            BROKEN_SCRATCH_PROB=1.0):
        image = degrade(
            _source_image(),
            damage_profile="semi_broken",
            rng=random.Random(seed),
            np_rng=np.random.default_rng(seed),
        )
    return image.mode, image.size, image.tobytes()


class RngReproducibilityTests(unittest.TestCase):
    def test_fields_are_isolated_from_global_random_state(self):
        random.seed(11)
        first = _field_sequence(2024)
        random.seed(999_999)
        second = _field_sequence(2024)

        self.assertEqual(first, second)
        self.assertNotEqual(first, _field_sequence(2025))

    def test_render_is_byte_identical_for_the_same_explicit_rng(self):
        random.seed(22)
        first = _rendered(1234)
        random.seed(888_888)
        second = _rendered(1234)

        self.assertEqual(first, second)
        self.assertNotEqual(first, _rendered(1235))

    def test_degradation_is_isolated_from_python_and_numpy_globals(self):
        random.seed(33)
        np.random.seed(33)
        first = _degraded(2024)
        random.seed(777_777)
        np.random.seed(777_777)
        second = _degraded(2024)

        self.assertEqual(first, second)
        self.assertNotEqual(first, _degraded(2025))

    def test_legacy_defaults_remain_callable(self):
        self.assertIsInstance(fields.make_value("age"), str)

        fallback_font = ImageFont.load_default()
        with (mock.patch.object(
                render, "fonts_for_style", return_value=("fallback.ttf",)),
              mock.patch.object(
                  render.ImageFont, "truetype", return_value=fallback_font)):
            rendered, _ = render.render_text("Test")
        self.assertEqual(rendered.mode, "RGB")

        with mock.patch.multiple(
                config,
                AUG_FADE_PROB=0.0,
                AUG_BRIGHTNESS_PROB=0.0,
                AUG_PAPER_TINT_PROB=0.0,
                AUG_STAIN_PROB=0.0,
                AUG_ROTATE_PROB=0.0,
                AUG_BLUR_PROB=0.0,
                AUG_NOISE_PROB=0.0):
            degraded = degrade(_source_image())
        self.assertEqual(degraded.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
