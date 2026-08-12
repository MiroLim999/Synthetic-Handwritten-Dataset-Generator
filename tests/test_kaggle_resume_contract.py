import ast
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "kaggle-code"
    / "trocr-finetuning-code.ipynb"
)


class _FakeScaler:
    def __init__(self, value=1.0):
        self.value = value

    def state_dict(self):
        return {"scale": self.value}

    def load_state_dict(self, state):
        self.value = state["scale"]


def _load_checkpoint_functions():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    tree = ast.parse(source)
    names = {
        "canonical_json",
        "capture_rng_states",
        "restore_rng_states",
        "checkpoint_payload",
        "validate_checkpoint_contract",
    }
    selected = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    if {node.name for node in selected} != names:
        raise AssertionError("Notebook checkpoint helpers are missing")
    namespace = {
        "json": json,
        "random": random,
        "np": np,
        "torch": torch,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(NOTEBOOK), "exec"),
         namespace)
    return namespace


class KaggleResumeContractTests(unittest.TestCase):
    def setUp(self):
        self.namespace = _load_checkpoint_functions()
        self.model = torch.nn.Linear(3, 2)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1)
        self.scaler = _FakeScaler(1024.0)
        self.generator = torch.Generator().manual_seed(911)
        self.namespace.update({
            "raw_model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "scaler": self.scaler,
            "DATA_LOADER_GENERATOR": self.generator,
            "RUN_KEY": "contract-test-123456789abc",
            "RUN_FINGERPRINT": "a" * 64,
            "RUN_CONFIGURATION": {"profile": "test", "seed": 911},
            "DATASET_PROVENANCE": {
                "manifest_sha256": "b" * 64,
                "images_sha256": "c" * 64,
            },
            "PROCESSOR_CHECKPOINT_DIR": Path("processor"),
            "PROCESSOR_ARTIFACT_SHA256": "d" * 64,
        })

    def test_checkpoint_round_trip_restores_training_state(self):
        loss = self.model(torch.ones(2, 3)).sum()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        payload = self.namespace["checkpoint_payload"](
            completed_epoch=2,
            best_cer=0.125,
            patience=1,
            best_artifact="epoch-002-valcer-0.12500000",
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training-state.pt"
            torch.save(payload, path)
            restored = torch.load(path, map_location="cpu", weights_only=False)

        self.namespace["validate_checkpoint_contract"](restored)
        self.assertEqual(restored["completed_epoch"], 2)
        self.assertEqual(restored["best_artifact"], "epoch-002-valcer-0.12500000")
        self.assertEqual(restored["scaler_state"], {"scale": 1024.0})

        replacement = torch.nn.Linear(3, 2)
        replacement.load_state_dict(restored["model_state"])
        for expected, actual in zip(self.model.parameters(), replacement.parameters()):
            self.assertTrue(torch.equal(expected, actual))

        replacement_optimizer = torch.optim.AdamW(replacement.parameters(), lr=0.5)
        replacement_optimizer.load_state_dict(restored["optimizer_state"])
        self.assertEqual(
            replacement_optimizer.param_groups[0]["lr"],
            self.optimizer.param_groups[0]["lr"],
        )
        replacement_scheduler = torch.optim.lr_scheduler.StepLR(
            replacement_optimizer, 1
        )
        replacement_scheduler.load_state_dict(restored["scheduler_state"])
        self.assertEqual(replacement_scheduler.last_epoch, self.scheduler.last_epoch)

    def test_rng_round_trip_replays_all_cpu_streams(self):
        random.seed(911)
        np.random.seed(911)
        torch.manual_seed(911)
        self.generator.manual_seed(911)
        states = self.namespace["capture_rng_states"]()
        expected = (
            random.random(),
            float(np.random.random()),
            float(torch.rand(1).item()),
            float(torch.rand(1, generator=self.generator).item()),
        )
        self.namespace["restore_rng_states"](states)
        replayed = (
            random.random(),
            float(np.random.random()),
            float(torch.rand(1).item()),
            float(torch.rand(1, generator=self.generator).item()),
        )
        self.assertEqual(expected, replayed)

    def test_resume_rejects_changed_dataset_or_configuration(self):
        payload = self.namespace["checkpoint_payload"](
            completed_epoch=0,
            best_cer=float("inf"),
            patience=0,
            best_artifact=None,
        )
        payload["dataset_provenance"] = dict(payload["dataset_provenance"])
        payload["dataset_provenance"]["images_sha256"] = "changed"
        with self.assertRaisesRegex(RuntimeError, "dataset provenance"):
            self.namespace["validate_checkpoint_contract"](payload)

        payload = self.namespace["checkpoint_payload"](
            completed_epoch=0,
            best_cer=float("inf"),
            patience=0,
            best_artifact=None,
        )
        payload["configuration"] = {"profile": "different", "seed": 911}
        with self.assertRaisesRegex(RuntimeError, "configuration"):
            self.namespace["validate_checkpoint_contract"](payload)


if __name__ == "__main__":
    unittest.main()
