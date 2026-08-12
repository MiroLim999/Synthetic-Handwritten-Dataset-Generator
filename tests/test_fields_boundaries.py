import calendar
import random
import unittest
from unittest import mock

import config
from src import fields


class DateFieldTests(unittest.TestCase):
    def test_ordinal_boundaries(self):
        self.assertEqual(fields._day_ordinal(29), "Twenty-Ninth")
        self.assertEqual(fields._day_ordinal(30), "Thirtieth")
        self.assertEqual(fields._day_ordinal(31), "Thirty-First")

    def test_year_boundaries_and_two_thousand_wording(self):
        with mock.patch.object(fields.random, "random", return_value=1.0):
            self.assertEqual(fields._year_in_words(1900), "Nineteen Hundred")
            self.assertEqual(fields._year_in_words(1920), "Nineteen Hundred Twenty")
            self.assertEqual(
                fields._year_in_words(1999), "Nineteen Hundred Ninety-Nine"
            )
            self.assertEqual(fields._year_in_words(2000), "Two Thousand")
            self.assertEqual(fields._year_in_words(2001), "Two Thousand One")
        self.assertNotIn("Twenty Hundred", fields._year_in_words(2000))

    def test_random_date_allows_leap_day_in_2000(self):
        with mock.patch.object(fields.random, "randint", side_effect=(2000, 2, 29)) as randint:
            self.assertEqual(fields._random_date_parts(), (29, 2, 2000))
        self.assertEqual(randint.call_args_list[-1], mock.call(1, 29))

    def test_random_date_uses_28_days_for_non_leap_century(self):
        with mock.patch.object(fields.random, "randint", side_effect=(1900, 2, 28)) as randint:
            self.assertEqual(fields._random_date_parts(), (28, 2, 1900))
        self.assertEqual(randint.call_args_list[-1], mock.call(1, 28))
        self.assertFalse(calendar.isleap(1900))
        self.assertTrue(calendar.isleap(2000))

    def test_written_and_numeric_dates_can_render_31st_and_leap_day(self):
        with mock.patch.object(fields, "_random_date_parts", return_value=(31, 12, 1999)), \
                mock.patch.object(fields.random, "random", return_value=0.0):
            self.assertEqual(fields.date_written(), "31 December 1999")

        with mock.patch.object(fields, "_random_date_parts", return_value=(29, 2, 2000)), \
                mock.patch.object(fields.random, "choice", return_value="/"), \
                mock.patch.object(fields.random, "random", return_value=0.0):
            self.assertEqual(fields.date_numeric(), "02/29/2000")

    def test_sampled_date_parts_are_always_calendar_valid(self):
        state = fields.random.getstate()
        self.addCleanup(fields.random.setstate, state)
        fields.random.seed(20260812)
        with mock.patch.object(config, "DATE_YEAR_RANGE", (1896, 2004)):
            for _ in range(5_000):
                day, month, year = fields._random_date_parts()
                self.assertLessEqual(day, calendar.monthrange(year, month)[1])

    def test_base_and_held_out_date_pattern_ids_are_disjoint(self):
        for field_type in fields.DATE_FORMAT_FIELD_TYPES:
            with self.subTest(field_type=field_type):
                base = set(fields.date_format_pattern_ids(
                    field_type, fields.BASE_FORMAT_PROFILE
                ))
                held_out = set(fields.date_format_pattern_ids(
                    field_type, fields.HELD_OUT_DATE_FORMAT_PROFILE
                ))
                self.assertTrue(base)
                self.assertTrue(held_out)
                self.assertTrue(base.isdisjoint(held_out))

    def test_written_date_profiles_generate_disjoint_structures(self):
        base_values = set()
        base_ids = set()
        held_values = set()
        held_ids = set()
        with mock.patch.object(fields, "_random_date_parts", return_value=(21, 3, 1967)):
            for roll in (0.1, 0.5, 0.9):
                base_rng = mock.Mock()
                base_rng.random.side_effect = [roll, 1.0]
                value, pattern_id = fields._date_written_with_format(
                    rng=base_rng, format_profile=fields.BASE_FORMAT_PROFILE
                )
                base_values.add(value)
                base_ids.add(pattern_id)

                held_rng = mock.Mock()
                held_rng.random.return_value = roll
                value, pattern_id = fields._date_written_with_format(
                    rng=held_rng,
                    format_profile=fields.HELD_OUT_DATE_FORMAT_PROFILE,
                )
                held_values.add(value)
                held_ids.add(pattern_id)

        self.assertTrue(base_values.isdisjoint(held_values))
        self.assertEqual(
            base_ids,
            set(fields.date_format_pattern_ids(
                "date_written", fields.BASE_FORMAT_PROFILE
            )),
        )
        self.assertEqual(
            held_ids,
            set(fields.date_format_pattern_ids(
                "date_written", fields.HELD_OUT_DATE_FORMAT_PROFILE
            )),
        )

    def test_numeric_date_profiles_generate_disjoint_structures(self):
        base_values = set()
        base_ids = set()
        held_values = set()
        held_ids = set()
        with mock.patch.object(fields, "_random_date_parts", return_value=(5, 3, 1967)):
            for roll in (0.1, 0.5, 0.9):
                base_rng = mock.Mock()
                base_rng.choice.return_value = "/"
                base_rng.random.return_value = roll
                value, pattern_id = fields._date_numeric_with_format(
                    rng=base_rng, format_profile=fields.BASE_FORMAT_PROFILE
                )
                base_values.add(value)
                base_ids.add(pattern_id)

                held_rng = mock.Mock()
                held_rng.random.return_value = roll
                value, pattern_id = fields._date_numeric_with_format(
                    rng=held_rng,
                    format_profile=fields.HELD_OUT_DATE_FORMAT_PROFILE,
                )
                held_values.add(value)
                held_ids.add(pattern_id)

        self.assertTrue(base_values.isdisjoint(held_values))
        self.assertEqual(
            base_ids,
            set(fields.date_format_pattern_ids(
                "date_numeric", fields.BASE_FORMAT_PROFILE
            )),
        )
        self.assertEqual(
            held_ids,
            set(fields.date_format_pattern_ids(
                "date_numeric", fields.HELD_OUT_DATE_FORMAT_PROFILE
            )),
        )

    def test_make_value_with_format_is_backward_compatible_and_auditable(self):
        rng_a = random.Random(55)
        rng_b = random.Random(55)
        legacy = fields.make_value("date_numeric", rng=rng_a)
        labelled, pattern_id = fields.make_value_with_format(
            "date_numeric", rng=rng_b
        )
        self.assertEqual(legacy, labelled)
        self.assertIn(
            pattern_id,
            fields.date_format_pattern_ids(
                "date_numeric", fields.BASE_FORMAT_PROFILE
            ),
        )

    def test_held_out_format_is_deterministic_for_an_explicit_seed(self):
        first = fields.make_value_with_format(
            "date_written",
            rng=random.Random(2026),
            format_profile=fields.HELD_OUT_DATE_FORMAT_PROFILE,
        )
        second = fields.make_value_with_format(
            "date_written",
            rng=random.Random(2026),
            format_profile=fields.HELD_OUT_DATE_FORMAT_PROFILE,
        )
        self.assertEqual(first, second)


class AgeFieldTests(unittest.TestCase):
    def test_age_range_includes_newborns_and_centenarians(self):
        self.assertEqual(config.AGE_YEAR_RANGE, (0, 110))
        with mock.patch.object(fields.random, "randint", return_value=0) as randint:
            self.assertEqual(fields.age(), "0")
            randint.assert_called_once_with(0, 110)
        with mock.patch.object(fields.random, "randint", return_value=110):
            self.assertEqual(fields.age(), "110")


class NameVariantTests(unittest.TestCase):
    def _resource_patches(self):
        return (
            mock.patch.object(fields, "_first_names", return_value=("Maria",)),
            mock.patch.object(fields, "_middle_names", return_value=("Angela",)),
            mock.patch.object(fields, "_last_names", return_value=("Doe",)),
        )

    def test_configurable_initial_casing_order_and_suffix_punctuation(self):
        probabilities = {
            "NAME_MIDDLE_PROB": 1.0,
            "NAME_MIDDLE_INITIAL_PROB": 1.0,
            "NAME_SUFFIX_PROB": 1.0,
            "NAME_SUFFIX_COMMA_PROB": 1.0,
            "NAME_SURNAME_FIRST_PROB": 1.0,
            "NAME_UPPERCASE_SURNAME_PROB": 1.0,
        }
        patches = self._resource_patches()
        with patches[0], patches[1], patches[2], mock.patch.multiple(config, **probabilities):
            value = fields.full_name(rng=random.Random(9))
        self.assertRegex(value, r"^DOE, Maria A\., (?:Jr\.|Sr\.|III|II)$")

    def test_name_variants_can_be_disabled(self):
        probabilities = {
            "NAME_MIDDLE_PROB": 0.0,
            "NAME_MIDDLE_INITIAL_PROB": 0.0,
            "NAME_SUFFIX_PROB": 0.0,
            "NAME_SUFFIX_COMMA_PROB": 0.0,
            "NAME_SURNAME_FIRST_PROB": 0.0,
            "NAME_UPPERCASE_SURNAME_PROB": 0.0,
        }
        patches = self._resource_patches()
        with patches[0], patches[1], patches[2], mock.patch.multiple(config, **probabilities):
            self.assertEqual(fields.full_name(rng=random.Random(9)), "Maria Doe")


class ResourceDeduplicationTests(unittest.TestCase):
    def test_known_resource_duplicates_were_removed(self):
        cases = (
            (config.PLACES_FILE, "combado"),
            (config.VOCAB_DIR / "cause_of_death.txt", "cardiac arrest"),
        )
        for path, expected in cases:
            with self.subTest(path=path):
                values = [line.strip().casefold() for line in path.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(values.count(expected), 1)


if __name__ == "__main__":
    unittest.main()
