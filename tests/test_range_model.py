"""RangeModel keeps its invariants under any sequence of edits."""

from __future__ import annotations

import random
import unittest

from gui_widgets import RangeModel, model_for_spec, snap_value
from src import studio_spec


def _on_step(model: RangeModel, value: float) -> bool:
    return value in (model.low, model.high) or snap_value(
        value, model.step, model.low, model.high) == value


class RangeModelTests(unittest.TestCase):
    def test_random_edits_keep_order_bounds_and_step(self):
        rng = random.Random(1)
        for spec in studio_spec.PARAMS:
            model = model_for_spec(spec, spec.default)
            for _ in range(200):
                handle = rng.choice(model.handles)
                operation = rng.randrange(3)
                if operation == 0:
                    model.set_handle(handle, rng.uniform(spec.low - 5, spec.high + 5))
                elif operation == 1:
                    model.nudge(handle, rng.choice((-10, -1, 1, 10)))
                else:
                    value = rng.uniform(spec.low - 5, spec.high + 5)
                    model.set_value(value if model.single
                                    else (value, rng.uniform(spec.low, spec.high)))
                values = [model.handle_value(h) for h in model.handles]
                self.assertEqual(values, sorted(values), spec.path)
                for value in values:
                    self.assertTrue(spec.low <= value <= spec.high, spec.path)
                    self.assertTrue(_on_step(model, value), (spec.path, value))
                if spec.integer:
                    current = model.value if model.single else model.value[0]
                    self.assertIsInstance(current, int, spec.path)

    def test_handles_stop_at_each_other(self):
        model = RangeModel(0, 10, 0.5, (3, 6))
        self.assertTrue(model.set_handle("low", 9))
        self.assertEqual(model.value, (6.0, 6.0))
        self.assertFalse(model.set_handle("high", 2))
        self.assertEqual(model.value, (6.0, 6.0))
        model.set_value((8, 1))  # set_value orders the pair
        self.assertEqual(model.value, (1.0, 8.0))

    def test_nudge_nearest_handle_and_fractions(self):
        model = RangeModel(0, 15, 0.1, (0.0, 4.5))
        self.assertTrue(model.nudge("low", 1))
        self.assertEqual(model.value, (0.1, 4.5))
        self.assertFalse(model.nudge("high", 1000) and model.nudge("high", 1))
        self.assertEqual(model.value[1], 15.0)
        self.assertEqual(model.nearest_handle(0.0), "low")
        self.assertEqual(model.nearest_handle(14.0), "high")
        self.assertEqual(model.value_at(0.5), 7.5)
        self.assertEqual(model.fraction(15.0), 1.0)
        tied = RangeModel(0, 10, 1, (5, 5))
        self.assertEqual(tied.nearest_handle(8), "high")
        self.assertEqual(tied.nearest_handle(2), "low")

    def test_display_and_typed_values(self):
        chance = RangeModel(0, 1, 0.01, 0.6, single=True, scale=100, unit="%")
        self.assertEqual(chance.text(), "60%")
        self.assertEqual(chance.parse("35%"), 0.35)
        tilt = RangeModel(0, 15, 0.1, (0, 4.5), unit="°")
        self.assertEqual(tilt.text(), "0.0° - 4.5°")
        self.assertEqual(tilt.parse("1.5 - 3"), (1.5, 3.0))
        self.assertEqual(tilt.parse("7"), (7.0, 7.0))
        with self.assertRaises(ValueError):
            tilt.parse("abc")
        with self.assertRaises(ValueError):
            chance.parse("1, 2")
        with self.assertRaises(ValueError):
            RangeModel(5, 5, 1, 5)

    def test_models_match_their_specs(self):
        jpeg = model_for_spec(studio_spec.PARAMS_BY_PATH["augmentation.jpeg.range"], [58, 88])
        self.assertEqual(jpeg.value, (58, 88))
        self.assertFalse(jpeg.single)
        top = model_for_spec(studio_spec.PARAMS_BY_PATH["cutoff.top.range"], [0.04, 0.09])
        self.assertEqual(top.text(), "4% - 9%")
        loss = model_for_spec(studio_spec.PARAMS_BY_PATH["cutoff.max_ink_loss"], 0.15)
        self.assertTrue(loss.single)
        self.assertEqual(loss.text(), "15%")


if __name__ == "__main__":
    unittest.main()
