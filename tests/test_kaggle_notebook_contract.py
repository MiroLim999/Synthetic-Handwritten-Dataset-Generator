import ast
import json
import unittest
from pathlib import Path


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "kaggle-code"
    / "trocr-finetuning-code.ipynb"
)


class KaggleNotebookContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code_cells = [
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        cls.code = "\n\n".join(cls.code_cells)

    def test_notebook_json_and_code_cells_are_valid(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        for index, source in enumerate(self.code_cells):
            ast.parse(source, filename=f"{NOTEBOOK}:code-cell-{index}")

    def test_requires_explicit_validated_unified_dataset(self):
        self.assertIn("DATA_ROOT = ''", self.code)
        self.assertIn("Set DATA_ROOT explicitly", self.code)
        self.assertIn("dataset-validation.json", self.code)
        self.assertIn("MANIFEST_SCHEMA_VERSION = '1'", self.code)
        for column in (
            "filename", "label", "split", "source", "field_type", "font",
            "sample_mode", "writer_id", "schema_version",
        ):
            self.assertIn(repr(column), self.code)
        self.assertIn("manifest_sha256", self.code)
        self.assertIn("images_sha256", self.code)
        self.assertIn("RUN_METADATA_JSON = DATA_ROOT / 'run-metadata.json'", self.code)
        self.assertIn(
            "EVALUATION_ANNOTATIONS_CSV = DATA_ROOT / 'evaluation-annotations.csv'",
            self.code,
        )

    def test_synthetic_evaluation_sidecar_is_validated_and_joined(self):
        for column in (
            "evaluation_condition", "format_profile", "format_id",
            "label_seen_in_train", "font_seen_in_train",
            "format_seen_in_train",
        ):
            self.assertIn(repr(column), self.code)
        self.assertIn("no matching synthetic row", self.code)
        self.assertIn("missing synthetic evaluation annotation", self.code)
        self.assertIn("condition violates policy", self.code)
        self.assertIn("format profile violates policy", self.code)
        self.assertIn("format id violates policy", self.code)
        self.assertIn(
            "annotations_df, on=['filename', 'split'], how='left'",
            self.code,
        )
        self.assertIn("real_writer_held_out", self.code)
        self.assertIn("evaluation_policy_sha256", self.code)
        self.assertIn("generator_preprocessing", self.code)

    def test_shared_aspect_preserving_preprocessing_is_used_everywhere(self):
        self.assertIn("PREPROCESSING_ID = 'aspect-pad-384-v1'", self.code)
        self.assertIn("ImageOps.contain", self.code)
        self.assertIn("do_resize=False", self.code)
        self.assertIn("image = load_preprocessed_image(row)", self.code)
        self.assertIn(
            "images = [load_preprocessed_image(row)", self.code)

    def test_final_test_is_one_shot_and_comparisons_use_validation(self):
        self.assertIn("base_val_metrics", self.code)
        self.assertNotIn("base_metrics", self.code)
        self.assertIn("_LOCKED_TEST_EVALUATED", self.code)
        self.assertIn("MODEL_CHOICES_FROZEN = False", self.code)
        final_test_calls = sum(
            "test_df, eval_model=best_model" in source
            for source in self.code_cells
        )
        self.assertEqual(final_test_calls, 1)

    def test_required_subgroup_metrics_and_clear_domains_are_present(self):
        for name in (
            "by_source", "by_field_type", "by_seen_unseen",
            "by_held_out_condition", "by_evaluation_condition",
            "by_format_profile", "by_format_id",
        ):
            self.assertIn(repr(name), self.code)
        self.assertIn("synthetic / in-distribution", self.code)
        self.assertIn("synthetic / held-out", self.code)
        self.assertIn("real / writer-held-out", self.code)

    def test_reproducibility_and_data_loader_contract(self):
        for seed_call in (
            "random.seed(seed)",
            "np.random.seed(seed)",
            "torch.manual_seed(seed)",
            "torch.cuda.manual_seed(seed)",
            "torch.cuda.manual_seed_all(seed)",
            "DATA_LOADER_GENERATOR.manual_seed(RUN_SEED)",
            "worker_init_fn=seed_data_loader_worker",
            "generator=DATA_LOADER_GENERATOR",
        ):
            self.assertIn(seed_call, self.code)
        self.assertIn("torch.use_deterministic_algorithms(True)", self.code)
        self.assertIn("CUBLAS_WORKSPACE_CONFIG", self.code)
        self.assertIn("drop_last=False", self.code)
        self.assertIn("NUM_WORKERS = 0", self.code)
        self.assertNotIn("torch.nn.DataParallel", self.code)

    def test_checkpoint_is_complete_and_provenance_guarded(self):
        for checkpoint_key in (
            "model_state", "optimizer_state", "scheduler_state",
            "scaler_state", "completed_epoch", "best_cer",
            "configuration", "dataset_provenance", "rng_states",
            "processor_artifact_sha256", "run_fingerprint",
        ):
            self.assertIn(repr(checkpoint_key), self.code)
        self.assertIn("validate_checkpoint_contract(resume_state)", self.code)
        self.assertIn("restore_rng_states(resume_state['rng_states'])", self.code)
        self.assertIn("os.replace(temporary, CHECKPOINT_PATH)", self.code)
        self.assertIn("RUN_KEY = f'{RUN_TAG}-{RUN_FINGERPRINT[:12]}'", self.code)

    def test_batching_accumulation_and_token_length_contract(self):
        self.assertIn(
            "math.ceil(len(train_loader) / cfg['GRAD_ACCUM_STEPS'])",
            self.code,
        )
        self.assertIn("step == len(train_loader)", self.code)
        self.assertIn("accumulation_group_samples = min(", self.code)
        self.assertIn(
            "raw_loss * labels.shape[0] / accumulation_group_samples",
            self.code,
        )
        self.assertIn("truncation=False", self.code)
        self.assertNotIn("truncation=True", self.code)
        self.assertIn("audit_token_lengths(", self.code)
        self.assertIn("training refuses silent truncation", self.code)

    def test_cuda_dynamic_evaluation_and_generation_contract(self):
        self.assertIn("if not torch.cuda.is_available()", self.code)
        self.assertIn("choose_eval_batch_size", self.code)
        self.assertIn("except torch.cuda.OutOfMemoryError", self.code)
        self.assertIn("max_length=cfg['MAX_LABEL_LENGTH']", self.code)
        self.assertNotIn("max_new_tokens", self.code)
        self.assertIn("Prediction count does not match evaluated rows", self.code)

    def test_versions_processor_and_evaluated_artifact_are_reported(self):
        self.assertRegex(self.code, r"MODEL_REVISION = '[0-9a-f]{40}'")
        for report_key in (
            "profile", "hyperparameters", "seeds", "dependencies",
            "processor", "split_definition", "dataset_provenance",
            "evaluated_artifact_sha256", "weights_sha256",
        ):
            self.assertIn(repr(report_key), self.code)
        self.assertIn(
            "TrOCRProcessor.from_pretrained(BEST_ARTIFACT_DIR)", self.code)
        self.assertIn(
            "VisionEncoderDecoderModel.from_pretrained(BEST_ARTIFACT_DIR)",
            self.code,
        )
        self.assertIn("current_artifact_sha256 != EVALUATED_ARTIFACT_SHA256", self.code)
        self.assertIn("archive.testzip()", self.code)
        self.assertNotIn("filterwarnings('ignore'", self.code)
        self.assertNotIn('simplefilter("ignore"', self.code)


if __name__ == "__main__":
    unittest.main()
