import queue
import threading
import unittest
from pathlib import Path
from unittest import mock

import gui


class _Worker:
    def __init__(self, alive=True):
        self.alive = alive
        self.joined = False

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        if self.alive:
            raise AssertionError("joined a live worker")
        self.joined = True


def _bare_app():
    app = object.__new__(gui.App)
    app.q = queue.Queue()
    return app


class GuiLifecycleTests(unittest.TestCase):
    def test_worker_uses_snapshots_and_passes_cancel_event(self):
        app = _bare_app()
        event = threading.Event()

        with (mock.patch.object(gui, "generate", return_value=Path("dataset_001"))
              as generate,
              mock.patch.object(gui, "build") as build,
              mock.patch.object(gui, "zip_dataset",
                                return_value=Path("dataset_001.zip")) as zip_dataset):
            app._run(3, None, 42, "name1", "regular", "all", "", "",
                     True, True, event)

        self.assertIs(generate.call_args.kwargs["cancel_event"], event)
        self.assertEqual(generate.call_args.kwargs["seed"], 42)
        self.assertTrue(generate.call_args.kwargs["archive_planned"])
        build.assert_called_once_with("dataset_001")
        zip_dataset.assert_called_once_with(
            Path("dataset_001"), cancel_event=event)
        self.assertEqual(app.q.get_nowait()[0], "status")
        self.assertEqual(
            app.q.get_nowait(), ("merge_result", mock.ANY))
        self.assertEqual(app.q.get_nowait()[0], "status")
        done = app.q.get_nowait()
        self.assertEqual(done[:4], ("done", 3, "dataset_001.zip", 42))

    def test_unknown_exception_is_not_concealed_when_event_is_set(self):
        app = _bare_app()
        event = threading.Event()
        event.set()

        with (mock.patch.object(gui, "generate",
                               side_effect=RuntimeError("merge corruption")),
              mock.patch.object(gui, "_log_error", return_value=Path("gui.log"))):
            app._run(3, None, 42, "name1", "regular", "all", "", "",
                     False, False, event)

        self.assertEqual(
            app.q.get_nowait(), ("error", "merge corruption", "gui.log"))

    def test_generation_cancelled_exception_is_reported_as_cancellation(self):
        app = _bare_app()
        event = threading.Event()
        event.set()

        with mock.patch.object(
                gui, "generate",
                side_effect=gui.GenerationCancelled("safe boundary")):
            app._run(3, None, 42, "name1", "regular", "all", "", "",
                     False, False, event)

        self.assertEqual(app.q.get_nowait(), ("cancelled", "safe boundary"))

    def test_delete_is_blocked_while_any_worker_is_alive(self):
        app = _bare_app()
        app.worker = _Worker(alive=True)
        app._selected_dataset = mock.Mock()

        with mock.patch.object(gui.messagebox, "showwarning") as warning:
            app._delete_selected_dataset()

        warning.assert_called_once()
        app._selected_dataset.assert_not_called()

    def test_close_requests_cancel_and_waits_for_worker(self):
        app = _bare_app()
        app.worker = _Worker(alive=True)
        app._closing = False
        app._signal_cancel = mock.Mock()
        app.destroy = mock.Mock()

        with mock.patch.object(gui.messagebox, "askyesno", return_value=True):
            app._on_close()

        self.assertTrue(app._closing)
        app._signal_cancel.assert_called_once_with()
        app.destroy.assert_not_called()

    def test_close_destroys_only_after_terminal_worker_exit(self):
        app = _bare_app()
        worker = _Worker(alive=True)
        app.worker = worker
        app.cancel_event = threading.Event()
        app.cancel_event.set()
        app._pending_terminal = None
        app._closing = True
        app.after = mock.Mock()
        app.destroy = mock.Mock()
        app.q.put(("cancelled", "done"))

        app._poll()
        app.destroy.assert_not_called()
        app.after.assert_called_once_with(50, app._poll)

        app.after.reset_mock()
        worker.alive = False
        app._poll()
        self.assertTrue(worker.joined)
        app.destroy.assert_called_once_with()
        app.after.assert_not_called()


if __name__ == "__main__":
    unittest.main()
