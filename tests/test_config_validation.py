import math
import unittest
from unittest import mock

import config


class ConfigValidationTests(unittest.TestCase):
    def test_default_configuration_is_valid(self):
        self.assertIsNone(config.validate_config())
        self.assertIsNone(config.validate_config(3))

    def test_count_must_be_a_positive_integer(self):
        for count in (0, -1, 1.5, True, "3"):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "count"):
                config.validate_config(count)

    def test_count_has_a_documented_safety_ceiling(self):
        with self.assertRaisesRegex(ValueError, "MAX_GENERATION_COUNT"):
            config.validate_config(config.MAX_GENERATION_COUNT + 1)
        with mock.patch.object(
                config, "LARGE_GENERATION_WARNING_COUNT",
                config.MAX_GENERATION_COUNT + 1):
            with self.assertRaisesRegex(
                    ValueError, "LARGE_GENERATION_WARNING_COUNT"):
                config.validate_config(3)
        with mock.patch.object(config, "MAX_GENERATION_COUNT", "many"):
            with self.assertRaisesRegex(ValueError, "MAX_GENERATION_COUNT"):
                config.validate_config(3)

    def test_count_must_support_all_positive_synthetic_splits(self):
        for count in (1, 2):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "at least 3"):
                config.validate_config(count)

    def test_synthetic_fractions_use_and_validate_all_three_values(self):
        invalid_sets = [
            (0.8, 0.1, 0.11),
            (0.8, 0.3, -0.1),
            (0.8, 0.1, math.nan),
            (0.8, 0.1, math.inf),
            (0.8, 0.1, "0.1"),
        ]
        for fractions in invalid_sets:
            patches = (
                mock.patch.object(config, "SYNTH_TRAIN_FRAC", fractions[0]),
                mock.patch.object(config, "SYNTH_VAL_FRAC", fractions[1]),
                mock.patch.object(config, "SYNTH_TEST_FRAC", fractions[2]),
            )
            with self.subTest(fractions=fractions), patches[0], patches[1], patches[2]:
                with self.assertRaisesRegex(ValueError, "synthetic split"):
                    config.validate_config()

    def test_real_fractions_are_validated_independently(self):
        with mock.patch.object(config, "REAL_TEST_FRAC", 0.3):
            with self.assertRaisesRegex(ValueError, "real split"):
                config.validate_config()

    def test_split_names_are_exact(self):
        with mock.patch.object(config, "SPLIT_NAMES", ("train", "validation", "test")):
            with self.assertRaisesRegex(ValueError, "SPLIT_NAMES"):
                config.validate_config()

    def test_probabilities_must_be_finite_and_bounded(self):
        for value in (-0.01, 1.01, math.nan, math.inf, True):
            with self.subTest(value=value), mock.patch.object(config, "AUG_BLUR_PROB", value):
                with self.assertRaisesRegex(ValueError, "AUG_BLUR_PROB"):
                    config.validate_config()

    def test_ranges_must_be_typed_ordered_and_bounded(self):
        cases = [
            ("FONT_SIZE_RANGE", (0, 10)),
            ("FONT_SIZE_RANGE", (10.0, 20.0)),
            ("INK_DARKNESS_RANGE", (80, 10)),
            ("INK_DARKNESS_RANGE", (0, 256)),
            ("DATE_YEAR_RANGE", (2000, 1920)),
            ("AGE_YEAR_RANGE", (-1, 110)),
            ("AUG_BLUR_RADIUS", (-0.1, 1.0)),
            ("AUG_NOISE_STD", (1, math.inf)),
            ("AUG_BRIGHTNESS_RANGE", (0, 1.0)),
            ("BROKEN_ERODE_BLEND", (0.1, 1.1)),
            ("BROKEN_CONTRAST_RANGE", (0.0, 0.5)),
        ]
        for name, value in cases:
            with self.subTest(name=name, value=value), mock.patch.object(config, name, value):
                with self.assertRaisesRegex(ValueError, name):
                    config.validate_config()

    def test_font_padding_and_rotation_are_positive_or_nonnegative(self):
        with mock.patch.object(config, "IMAGE_PADDING", 0):
            with self.assertRaisesRegex(ValueError, "IMAGE_PADDING"):
                config.validate_config()
        with mock.patch.object(config, "AUG_ROTATE_DEGREES", -0.1):
            with self.assertRaisesRegex(ValueError, "AUG_ROTATE_DEGREES"):
                config.validate_config()

    def test_field_weights_are_recognized_nonnegative_and_not_all_zero(self):
        invalid = (
            {"not_a_field": 1},
            {"age": -1},
            {"age": 0, "sex": 0},
            {"age": math.nan},
            {"age": True},
        )
        for weights in invalid:
            with self.subTest(weights=weights), mock.patch.object(config, "FIELD_WEIGHTS", weights):
                with self.assertRaisesRegex(ValueError, "FIELD_WEIGHTS"):
                    config.validate_config()


class SplitAllocationTests(unittest.TestCase):
    def test_largest_remainder_uses_stable_split_order_for_ties(self):
        self.assertEqual(
            config.allocate_split_counts(10, (0.34, 0.33, 0.33)),
            {"train": 4, "val": 3, "test": 3},
        )
        self.assertEqual(
            config.allocate_split_counts(11, (0.8, 0.1, 0.1)),
            {"train": 9, "val": 1, "test": 1},
        )

    def test_tiny_three_way_allocation_is_nonempty_or_fails_clearly(self):
        self.assertEqual(
            config.allocate_split_counts(3, (0.8, 0.1, 0.1)),
            {"train": 1, "val": 1, "test": 1},
        )
        with self.assertRaisesRegex(ValueError, "at least 3"):
            config.allocate_split_counts(2, (0.8, 0.1, 0.1))

    def test_zero_fraction_does_not_require_an_item(self):
        self.assertEqual(
            config.allocate_split_counts(2, (0.5, 0.5, 0.0)),
            {"train": 1, "val": 1, "test": 0},
        )

    def test_named_allocators_use_their_respective_configured_fractions(self):
        self.assertEqual(
            config.allocate_synthetic_split_counts(10),
            {"train": 8, "val": 1, "test": 1},
        )
        self.assertEqual(
            config.allocate_real_split_counts(5),
            {"train": 3, "val": 1, "test": 1},
        )

    def test_allocation_validates_total_and_fractions(self):
        for total in (0, -1, 2.5, True):
            with self.subTest(total=total), self.assertRaises(ValueError):
                config.allocate_split_counts(total, (0.8, 0.1, 0.1))
        for fractions in ((0.8, 0.2), (0.8, 0.2, 0.1), (1.1, 0.0, -0.1)):
            with self.subTest(fractions=fractions), self.assertRaises(ValueError):
                config.allocate_split_counts(10, fractions)
        with mock.patch.object(config, "SPLIT_NAMES", ("train", "validation", "test")):
            with self.assertRaisesRegex(ValueError, "SPLIT_NAMES"):
                config.allocate_split_counts(10, (0.8, 0.1, 0.1))


if __name__ == "__main__":
    unittest.main()
