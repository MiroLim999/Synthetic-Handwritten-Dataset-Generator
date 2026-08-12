import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryContractTests(unittest.TestCase):
    def test_documentation_names_the_supported_runtime_and_artifacts(self):
        readme = (ROOT / "README.md").read_text("utf-8")
        self.assertIn("Python 3.10 or newer", readme)
        self.assertIn("manifest.csv", readme)
        self.assertIn("dataset-validation.json", readme)
        self.assertIn("run-metadata.json", readme)
        self.assertIn("image-statistics.json", readme)
        self.assertIn("writer_style_v1", readme)
        self.assertIn("historical_scan_v1", readme)
        self.assertIn("--yes", readme)
        self.assertIn(".zip.sha256", readme)
        self.assertIn("real / writer-held-out", readme.casefold())
        self.assertNotIn("ships with **24 handwriting fonts**", readme)
        self.assertNotIn("modern dark theme", readme)

    def test_kaggle_guide_references_the_actual_notebook(self):
        guide = (ROOT / "kaggle-code" / "README.md").read_text("utf-8")
        self.assertIn("trocr-finetuning-code.ipynb", guide)
        self.assertIn("Kaggle", guide)
        self.assertIn("DATA_ROOT", guide)
        self.assertNotIn("train_trocr_colab.ipynb", guide)
        self.assertNotIn("MyDrive/splits.zip", guide)

    def test_runtime_dependencies_exclude_unimplemented_scan_stack(self):
        runtime = (ROOT / "requirements.txt").read_text("utf-8").casefold()
        development = (ROOT / "requirements-dev.txt").read_text("utf-8").casefold()
        self.assertNotIn("opencv", runtime)
        self.assertIn("faker", development)
        self.assertIn("constraints.txt", runtime)

    def test_windows_launcher_has_no_personal_path(self):
        launcher = (ROOT / "Generate Images.bat").read_text("utf-8").casefold()
        self.assertNotIn("c:\\users\\", launcher)
        self.assertIn(".venv", launcher)
        self.assertIn("conda_prefix", launcher)
        self.assertIn("pyw.exe", launcher)

    def test_font_inventory_template_is_well_formed(self):
        font_guide = ROOT / "resources" / "fonts" / "README.md"
        template = ROOT / "resources" / "fonts" / "FONT_MANIFEST.example.csv"
        self.assertTrue(font_guide.is_file())
        self.assertTrue((ROOT / "NOTICE.md").is_file())
        with template.open("r", encoding="utf-8", newline="") as source:
            reader = csv.reader(source)
            self.assertEqual(
                next(reader),
                [
                    "filename",
                    "sha256",
                    "source_url",
                    "license_id",
                    "license_url",
                    "license_file",
                    "installed_at_utc",
                ],
            )
            self.assertEqual(list(reader), [])


if __name__ == "__main__":
    unittest.main()
