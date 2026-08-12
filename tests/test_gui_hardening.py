"""Headless tests for GUI background work, reporting, logging, and portability."""

from __future__ import annotations

import inspect
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import config
import gui


class _Tree:
    def __init__(self):
        self.rows = {}
        self.updated = []
        self._next = 0

    def get_children(self):
        return tuple(self.rows)

    def delete(self, *items):
        for item in items:
            self.rows.pop(item, None)

    def insert(self, _parent, _position, *, text, values):
        self._next += 1
        iid = f"row-{self._next}"
        self.rows[iid] = {"text": text, "values": values}
        return iid

    def exists(self, iid):
        return iid in self.rows

    def item(self, iid, *, values):
        self.rows[iid]["values"] = values
        self.updated.append((iid, values))


class _Widget:
    def __init__(self, maximum=10):
        self.values = {"maximum": maximum}

    def config(self, **values):
        self.values.update(values)

    def __getitem__(self, key):
        return self.values[key]


def _bare_app():
    app = object.__new__(gui.App)
    app.q = queue.Queue()
    return app


class GuiHardeningTests(unittest.TestCase):
    def test_refresh_worker_uses_queue_and_generation_token_drops_stale_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "dataset_001"
            folder.mkdir()
            (folder / "image.png").write_bytes(b"image")
            hidden_staging = root / ".dataset_002.tmp-incomplete"
            hidden_staging.mkdir()
            (hidden_staging / "partial.png").write_bytes(b"partial")
            (root / "dataset_001.zip").write_bytes(b"zip")
            (root / "dataset_001.zip.sha256").write_text("digest\n", encoding="ascii")

            app = _bare_app()
            app.tree = _Tree()
            app._dataset_paths = {}
            app._refresh_generation = 0
            app.after = mock.Mock(side_effect=AssertionError("worker called Tk.after"))
            app.status = _Widget()

            with mock.patch.object(config, "DATASETS_DIR", root):
                app._refresh_datasets()
                app._refresh_worker.join(timeout=5)
            self.assertFalse(app._refresh_worker.is_alive())
            messages = []
            while not app.q.empty():
                messages.append(app.q.get_nowait())
            size_message = next(item for item in messages if item[0] == "dataset_size")
            self.assertEqual(size_message[1], 1)
            self.assertEqual(size_message[-1], "yes")
            self.assertEqual(len(app._dataset_paths), 1)
            self.assertNotIn(hidden_staging, app._dataset_paths.values())
            app.after.assert_not_called()

            iid = size_message[2]
            app._refresh_generation = 2
            app._patch_row(*size_message[1:])
            self.assertEqual(app.tree.updated, [])
            current = (2, iid, "1.0 MB", "2.0 MB", "yes")
            app._patch_row(*current)
            self.assertEqual(app.tree.updated[-1], (iid, current[2:]))

    def test_refresh_requests_are_coalesced_while_one_scan_is_active(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "dataset_001").mkdir()
            entered = threading.Event()
            release = threading.Event()

            app = _bare_app()
            app.tree = _Tree()
            app._dataset_paths = {}
            app._refresh_generation = 0
            app._refresh_worker = None
            app._refresh_pending = False
            app._refresh_finalize_scheduled = False
            app.status = _Widget()
            app.after = mock.Mock()

            def slow_size(_path):
                entered.set()
                release.wait(timeout=5)
                return 0

            app._folder_size_fast = slow_size
            with mock.patch.object(config, "DATASETS_DIR", root):
                app._refresh_datasets()
                self.assertTrue(entered.wait(timeout=5))
                first_worker = app._refresh_worker
                app._refresh_datasets()
                self.assertIs(app._refresh_worker, first_worker)
                self.assertTrue(app._refresh_pending)
                release.set()
                first_worker.join(timeout=5)
                app._finish_refresh_worker()

            self.assertIsNone(app._refresh_worker)
            app.after.assert_called_once_with(0, app._refresh_datasets)

    def test_delete_worker_removes_folder_zip_and_checksum_through_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "datasets"
            folder = root / "dataset_001"
            folder.mkdir(parents=True)
            (folder / "image.png").write_bytes(b"image")
            archive = root / "dataset_001.zip"
            checksum = root / "dataset_001.zip.sha256"
            archive.write_bytes(b"zip")
            checksum.write_text("digest\n", encoding="ascii")

            app = _bare_app()
            app.after = mock.Mock(side_effect=AssertionError("delete worker called Tk"))
            with mock.patch.object(config, "DATASETS_DIR", root):
                app._delete_dataset_worker(folder, archive, checksum)

            messages = []
            while not app.q.empty():
                messages.append(app.q.get_nowait())
            self.assertEqual(messages[-1], (
                "delete_done", "dataset_001", True, True, True,
            ))
            self.assertFalse(folder.exists())
            self.assertFalse(archive.exists())
            self.assertFalse(checksum.exists())
            app.after.assert_not_called()

    def test_delete_worker_rejects_sidecar_target_substitution_before_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "datasets"
            folder = root / "dataset_001"
            folder.mkdir(parents=True)
            (folder / "keep.png").write_bytes(b"keep")
            expected_zip = root / "dataset_001.zip"
            expected_zip.write_bytes(b"keep zip")
            outside = Path(temporary) / "outside.zip"
            outside.write_bytes(b"outside")

            app = _bare_app()
            with (mock.patch.object(config, "DATASETS_DIR", root),
                  mock.patch.object(gui, "_log_error", return_value=root / "gui.log")):
                app._delete_dataset_worker(
                    folder, outside, root / "dataset_001.zip.sha256"
                )

            self.assertTrue(folder.exists())
            self.assertEqual(expected_zip.read_bytes(), b"keep zip")
            self.assertEqual(outside.read_bytes(), b"outside")
            self.assertEqual(app.q.get_nowait()[0], "delete_error")

    def test_generation_error_is_logged_with_traceback_and_queued(self):
        app = _bare_app()
        event = threading.Event()
        with (mock.patch.object(gui, "generate", side_effect=RuntimeError("boom")),
              mock.patch.object(gui, "_log_error", return_value=Path("gui.log")) as log):
            app._run(
                3, None, 42, "name1", "regular", "all", "", "",
                False, False, event,
            )

        context, detail = log.call_args.args
        self.assertEqual(context, "generation worker")
        self.assertIn("RuntimeError: boom", detail)
        self.assertEqual(app.q.get_nowait(), ("error", "boom", "gui.log"))

    def test_completion_reports_structured_merge_and_no_op_warning(self):
        app = _bare_app()
        app.start_time = time.time()
        app.progress = _Widget(maximum=3)
        app.status = _Widget()
        app.timer = _Widget()
        app.open_btn = _Widget()
        app._set_job_active = mock.Mock()
        app._refresh_datasets = mock.Mock()
        payload = {
            "copied": 0,
            "unchanged": 4,
            "removed": 0,
            "skipped": 0,
            "failed": 0,
        }

        with mock.patch.object(gui.messagebox, "showinfo") as showinfo:
            app._finish_job(("done", 3, "dataset_001", 42, payload))

        message = showinfo.call_args.args[1]
        self.assertIn("copied: 0", message)
        self.assertIn("unchanged: 4", message)
        self.assertIn("Warning: real merge made no dataset changes", message)

    def test_log_rotation_preserves_full_details(self):
        with tempfile.TemporaryDirectory() as temporary:
            log_dir = Path(temporary) / "logs"
            log_file = log_dir / "gui.log"
            log_dir.mkdir()
            log_file.write_text("x" * 100, encoding="utf-8")
            with (mock.patch.object(gui, "LOG_DIR", log_dir),
                  mock.patch.object(gui, "LOG_FILE", log_file),
                  mock.patch.object(gui, "MAX_LOG_BYTES", 32)):
                result = gui._log_error("test context", "Traceback\nfull details")

            self.assertEqual(result, log_file)
            self.assertTrue((log_dir / "gui.log.1").is_file())
            content = log_file.read_text(encoding="utf-8")
            self.assertIn("test context", content)
            self.assertIn("Traceback\nfull details", content)

    def test_portable_linux_folder_opening_and_source_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with (mock.patch.object(gui.sys, "platform", "linux"),
                  mock.patch.object(gui.subprocess, "Popen") as popen):
                gui._open_path_portably(folder)
            popen.assert_called_once_with(["xdg-open", str(folder.resolve())])

        source = inspect.getsource(gui.App._build_generate_tab)
        self.assertIn("config.DEFAULT_COUNT", source)
        self.assertNotIn('StringVar(value="20000")', source)
        start_source = inspect.getsource(gui.App.start)
        self.assertIn("LARGE_GENERATION_WARNING_COUNT", start_source)


if __name__ == "__main__":
    unittest.main()
