import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.generation_resources import (
    InsufficientDiskSpaceError,
    estimate_generation_resources,
    format_bytes,
    preflight_generation_resources,
)


class GenerationResourceTests(unittest.TestCase):
    def test_estimate_accounts_for_source_and_archive_peak(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch(
                "src.generation_resources.shutil.disk_usage",
                return_value=SimpleNamespace(free=10**12)):
            plain = estimate_generation_resources(Path(temp), 100)
            archived = estimate_generation_resources(
                Path(temp), 100, create_archive=True
            )

        self.assertEqual(plain.estimated_archive_bytes, 0)
        self.assertEqual(
            archived.estimated_peak_disk_bytes,
            archived.estimated_dataset_bytes + archived.estimated_archive_bytes,
        )
        self.assertGreater(
            archived.required_free_space_bytes, plain.required_free_space_bytes
        )
        self.assertTrue(archived.enough_free_space)

    def test_large_job_requests_confirmation(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch(
                "src.generation_resources.shutil.disk_usage",
                return_value=SimpleNamespace(free=10**15)):
            estimate = estimate_generation_resources(
                temp, 10, large_job_count=10
            )
        self.assertTrue(estimate.confirmation_recommended)
        self.assertTrue(any("Large job" in warning for warning in estimate.warnings))

    def test_preflight_rejects_low_disk_before_writes(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch(
                "src.generation_resources.shutil.disk_usage",
                return_value=SimpleNamespace(free=0)):
            with self.assertRaises(InsufficientDiskSpaceError) as caught:
                preflight_generation_resources(temp, 1)
        self.assertFalse(caught.exception.estimate.enough_free_space)

    def test_estimate_is_serializable_and_format_is_readable(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch(
                "src.generation_resources.shutil.disk_usage",
                return_value=SimpleNamespace(free=10**12)):
            estimate = estimate_generation_resources(temp, 2)
        self.assertEqual(estimate.to_metadata()["sample_count"], 2)
        self.assertEqual(format_bytes(1024), "1.0 KiB")


if __name__ == "__main__":
    unittest.main()

