import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from src.image_statistics import (
    calculate_image_statistics,
    compare_image_statistics,
    iter_image_paths,
    summarize_image_statistics,
)


class ImageStatisticsTests(unittest.TestCase):
    def test_calculates_geometry_luminance_ink_and_edges(self):
        image = Image.new("L", (100, 20), "white")
        ImageDraw.Draw(image).rectangle((20, 5, 80, 15), fill="black")
        statistics = calculate_image_statistics(image)

        self.assertEqual(statistics.aspect_ratio, 5.0)
        self.assertGreater(statistics.ink_coverage, 0.0)
        self.assertGreater(statistics.robust_contrast, 0.0)
        self.assertGreater(statistics.edge_density, 0.0)

    def test_summary_streams_paths_and_images(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.png"
            Image.new("RGB", (20, 10), "white").save(path)
            summary = summarize_image_statistics(
                iter((path, Image.new("RGB", (30, 10), "black")))
            )

        self.assertEqual(summary.image_count, 2)
        self.assertEqual(summary.metrics["width"].mean, 25.0)
        self.assertEqual(summary.to_metadata()["image_count"], 2)

    def test_comparison_reports_direction_without_claiming_equivalence(self):
        comparison = compare_image_statistics(
            [Image.new("L", (10, 10), 240), Image.new("L", (10, 10), 220)],
            [Image.new("L", (10, 10), 80), Image.new("L", (10, 10), 100)],
        )
        luminance = comparison.metrics["mean_luminance"]
        self.assertGreater(luminance.mean_difference, 0)
        self.assertIsNotNone(luminance.standardized_difference)
        self.assertIn("do not", comparison.interpretation)

    def test_iter_image_paths_is_filtered_and_stable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "b.PNG").write_bytes(b"not decoded here")
            (root / "a.txt").write_text("ignore", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "a.jpg").write_bytes(b"not decoded here")
            names = [path.relative_to(root).as_posix()
                     for path in iter_image_paths(root)]
        self.assertEqual(names, ["b.PNG", "nested/a.jpg"])

    def test_empty_summary_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "At least one image"):
            summarize_image_statistics([])


if __name__ == "__main__":
    unittest.main()

