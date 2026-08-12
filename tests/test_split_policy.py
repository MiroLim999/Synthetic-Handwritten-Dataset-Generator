import random
import unittest

from PIL import Image

from src.split_policy import (
    BASE_DEGRADATION_PROFILE,
    HELD_OUT_DEGRADATION_PROFILE,
    build_synthetic_evaluation_policy,
    evaluation_annotations,
)
from src.fields import BASE_FORMAT_PROFILE, HELD_OUT_DATE_FORMAT_PROFILE


class FontHoldoutPolicyTests(unittest.TestCase):
    def setUp(self):
        self.fonts = tuple(f"C:/fonts/font_{index:02d}.ttf" for index in range(10))

    def test_test_fonts_are_completely_disjoint_from_train_and_val(self):
        policy = build_synthetic_evaluation_policy(self.fonts, seed=42)
        train = set(policy.fonts_for_split("train"))
        val = set(policy.fonts_for_split("val"))
        test = set(policy.fonts_for_split("test"))

        self.assertTrue(policy.font_holdout_enabled)
        self.assertEqual(train, val)
        self.assertTrue(train)
        self.assertTrue(test)
        self.assertTrue(train.isdisjoint(test))
        self.assertEqual(train | test, set(self.fonts))

    def test_policy_is_deterministic_and_independent_of_font_input_order(self):
        first = build_synthetic_evaluation_policy(self.fonts, seed=2026)
        second = build_synthetic_evaluation_policy(tuple(reversed(self.fonts)), seed=2026)
        self.assertEqual(first, second)
        self.assertEqual(first.to_metadata(), second.to_metadata())

        rng_a = random.Random(99)
        rng_b = random.Random(99)
        chosen_a = [first.choose_font("test", rng_a) for _ in range(20)]
        chosen_b = [second.choose_font("test", rng_b) for _ in range(20)]
        self.assertEqual(chosen_a, chosen_b)

    def test_single_font_disables_only_font_holdout(self):
        policy = build_synthetic_evaluation_policy([self.fonts[0]], seed=42)
        self.assertFalse(policy.font_holdout_enabled)
        self.assertEqual(policy.train_fonts, policy.test_fonts)
        self.assertEqual(
            policy.evaluation_condition_for_split("test"),
            "synthetic_held_out_degradation",
        )

    def test_duplicate_manifest_font_filenames_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "filenames must be unique"):
            build_synthetic_evaluation_policy(
                ["C:/one/Same.ttf", "D:/two/same.TTF"], seed=1
            )

    def test_invalid_policy_inputs_fail_clearly(self):
        for fonts in ([], "one.ttf"):
            with self.subTest(fonts=fonts), self.assertRaises(ValueError):
                build_synthetic_evaluation_policy(fonts, seed=1)
        for fraction in (0, 1, -0.1, float("nan"), True):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                build_synthetic_evaluation_policy(
                    self.fonts, seed=1, test_font_fraction=fraction
                )
        with self.assertRaises(ValueError):
            build_synthetic_evaluation_policy(self.fonts, seed=True)


class DegradationHoldoutTests(unittest.TestCase):
    def setUp(self):
        self.policy = build_synthetic_evaluation_policy(
            ["a.ttf", "b.ttf"], seed=123
        )
        self.image = Image.new("RGB", (80, 24), "white")
        pixels = self.image.load()
        for x in range(0, 80, 2):
            for y in range(24):
                pixels[x, y] = (0, 0, 0)

    def test_only_test_receives_held_out_degradation(self):
        self.assertEqual(
            self.policy.degradation_profile_for_split("train"),
            BASE_DEGRADATION_PROFILE,
        )
        self.assertEqual(
            self.policy.degradation_profile_for_split("val"),
            BASE_DEGRADATION_PROFILE,
        )
        self.assertEqual(
            self.policy.degradation_profile_for_split("test"),
            HELD_OUT_DEGRADATION_PROFILE,
        )
        self.assertIs(
            self.policy.apply_degradation_holdout(
                self.image, "train", sample_key="syn_1.png"
            ),
            self.image,
        )

    def test_test_degradation_is_deterministic_and_content_changing(self):
        first = self.policy.apply_degradation_holdout(
            self.image, "test", sample_key="syn_1.png"
        )
        second = self.policy.apply_degradation_holdout(
            self.image, "test", sample_key="syn_1.png"
        )
        self.assertEqual(first.size, self.image.size)
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertNotEqual(first.tobytes(), self.image.tobytes())


class EvaluationAnnotationTests(unittest.TestCase):
    def test_sidecar_marks_domain_label_seen_and_font_seen(self):
        policy = build_synthetic_evaluation_policy(
            ["train.ttf", "test.ttf"], seed=5
        )
        train_font = policy.train_fonts[0]
        test_font = policy.test_fonts[0]
        rows = [
            {"filename": "one.png", "label": "Maria", "split": "train", "font": train_font},
            {"filename": "two.png", "label": "Maria", "split": "val", "font": train_font},
            {"filename": "three.png", "label": "Jose", "split": "test", "font": test_font},
        ]
        annotations = evaluation_annotations(rows, policy)

        self.assertTrue(annotations[1]["label_seen_in_train"])
        self.assertTrue(annotations[1]["font_seen_in_train"])
        self.assertEqual(
            annotations[1]["evaluation_condition"], "synthetic_in_distribution"
        )
        self.assertFalse(annotations[2]["label_seen_in_train"])
        self.assertFalse(annotations[2]["font_seen_in_train"])
        self.assertEqual(
            annotations[2]["evaluation_condition"],
            "synthetic_unseen_font+held_out_degradation",
        )

    def test_metadata_exposes_policy_and_no_test_font_overlap(self):
        policy = build_synthetic_evaluation_policy(
            ["a.ttf", "b.ttf", "c.ttf"], seed=8
        )
        metadata = policy.to_metadata()
        self.assertEqual(metadata["policy_version"], "1")
        self.assertFalse(metadata["font_holdout"]["test_fonts_seen_in_train"])
        self.assertEqual(
            metadata["evaluation_conditions"]["test"],
            "synthetic_unseen_font+held_out_degradation",
        )


class FormatHoldoutPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = build_synthetic_evaluation_policy(
            ["train.ttf", "test.ttf"], seed=77
        )

    def test_date_format_holdout_is_test_only(self):
        for field_type in ("date_written", "date_numeric"):
            with self.subTest(field_type=field_type):
                self.assertEqual(
                    self.policy.format_profile_for_field("train", field_type),
                    BASE_FORMAT_PROFILE,
                )
                self.assertEqual(
                    self.policy.format_profile_for_field("val", field_type),
                    BASE_FORMAT_PROFILE,
                )
                self.assertEqual(
                    self.policy.format_profile_for_field("test", field_type),
                    HELD_OUT_DATE_FORMAT_PROFILE,
                )
        self.assertEqual(
            self.policy.format_profile_for_field("test", "full_name"),
            BASE_FORMAT_PROFILE,
        )

    def test_format_condition_is_row_specific(self):
        self.assertEqual(
            self.policy.evaluation_condition_for_sample("test", "full_name"),
            "synthetic_unseen_font+held_out_degradation",
        )
        self.assertEqual(
            self.policy.evaluation_condition_for_sample("test", "date_written"),
            "synthetic_unseen_font+held_out_degradation+held_out_format",
        )

    def test_metadata_proves_pattern_sets_are_disjoint(self):
        metadata = self.policy.to_metadata()["format_holdout"]
        self.assertFalse(metadata["empirically_calibrated"])
        for field_type, profiles in metadata["pattern_ids"].items():
            with self.subTest(field_type=field_type):
                self.assertTrue(
                    set(profiles[BASE_FORMAT_PROFILE]).isdisjoint(
                        profiles[HELD_OUT_DATE_FORMAT_PROFILE]
                    )
                )

    def test_annotations_expose_format_profile_id_and_seen_status(self):
        train_font = self.policy.train_fonts[0]
        test_font = self.policy.test_fonts[0]
        rows = [
            {
                "filename": "train.png",
                "label": "03/15/1967",
                "split": "train",
                "field_type": "date_numeric",
                "font": train_font,
                "format_profile": BASE_FORMAT_PROFILE,
                "format_id": "padded_month_day_full_year",
            },
            {
                "filename": "test.png",
                "label": "1967/03/15",
                "split": "test",
                "field_type": "date_numeric",
                "font": test_font,
                "format_profile": HELD_OUT_DATE_FORMAT_PROFILE,
                "format_id": "full_year_month_day",
            },
        ]
        annotations = evaluation_annotations(rows, self.policy)
        self.assertTrue(annotations[0]["format_seen_in_train"])
        self.assertFalse(annotations[1]["format_seen_in_train"])
        self.assertEqual(
            annotations[1]["evaluation_condition"],
            "synthetic_unseen_font+held_out_degradation+held_out_format",
        )
        self.assertEqual(
            annotations[1]["format_profile"], HELD_OUT_DATE_FORMAT_PROFILE
        )
        self.assertEqual(annotations[1]["format_id"], "full_year_month_day")

    def test_annotations_reject_a_profile_that_violates_policy(self):
        with self.assertRaisesRegex(ValueError, "policy requires"):
            evaluation_annotations([
                {
                    "filename": "bad.png",
                    "label": "03/15/1967",
                    "split": "test",
                    "field_type": "date_numeric",
                    "font": self.policy.test_fonts[0],
                    "format_profile": BASE_FORMAT_PROFILE,
                }
            ], self.policy)


if __name__ == "__main__":
    unittest.main()
