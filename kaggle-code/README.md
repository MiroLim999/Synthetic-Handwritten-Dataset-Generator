# Fine-tuning TrOCR on Kaggle

`trocr-finetuning-code.ipynb` is the supported training notebook. It is written
for a Kaggle CUDA GPU session and consumes one validated dataset produced by
this repository. It is not a Google Colab/Drive notebook.

## 1. Publish and attach the dataset

Generate the dataset locally, inspect `review-montage.jpg`, and confirm that
`dataset-validation.json` reports `"valid": true`. Upload the complete dataset
folder as a private Kaggle Dataset unless its approved sharing policy permits
broader access. Preserve all of these entries:

```text
dataset_NNN/
|-- train/
|-- val/
|-- test/
|-- manifest.csv
|-- labels.csv
|-- dataset-validation.json
|-- run-metadata.json
|-- evaluation-annotations.csv
|-- review-montage.jpg
`-- image-statistics.json
```

`labels.csv` is only a compatibility view; the notebook reads the unified
`manifest.csv`. If real handwriting is included, apply the consent, access,
retention, sharing, and deletion rules in the main README before uploading it.
If transferring the generator-created ZIP, keep its `.zip.sha256` sidecar and
verify the checksum before publishing the extracted folder as a Kaggle Dataset.

In the Kaggle notebook editor, use **Add Input** to attach the dataset. Enable a
CUDA GPU accelerator. Internet access is needed if the pinned base-model
revision is not already available in the session cache.

## 2. Set the explicit configuration

Run the environment/import cells, then edit the configuration cell:

```python
DATA_ROOT = "/kaggle/input/<dataset-slug>/dataset_NNN"
RUN_TAG = "experiment-001"
PROFILE = "50k"
```

`DATA_ROOT` must be the exact directory containing `manifest.csv`,
`dataset-validation.json`, and `train/`, `val/`, `test/`. Leave no wildcard and
do not point it at a parent containing several runs. When `DATA_ROOT` is blank,
the notebook prints candidate manifest directories and stops instead of
silently choosing one.

Change `RUN_TAG` for a genuinely new experiment. The run fingerprint also binds
the dataset hashes, model revision, preprocessing contract, seed,
hyperparameters, and dependency versions; an incompatible directory is
rejected rather than reused.

## 3. Run validation and training

Run cells in order. Before downloading a model, the notebook independently
checks:

- the exact version-1 unified-manifest schema;
- safe, unique PNG filenames and readable images;
- non-empty train/validation/test splits;
- missing and orphan images;
- synthetic/real row requirements;
- disjoint real writers across splits; and
- the manifest and image hashes recorded in `dataset-validation.json`.

One aspect-preserving 384 x 384 resize-and-pad transform is shared by training,
evaluation, and inference. Validation—not test—drives early stopping and model
selection. The locked test cell intentionally requires
`MODEL_CHOICES_FROZEN = True` and should be executed once after all choices are
final.

The final report separates:

- `synthetic / in-distribution`;
- `synthetic / held-out`; and
- `real / writer-held-out`.

Treat the real writer-held-out result as the most relevant new-writer estimate.
Synthetic scores are controlled diagnostic measurements, not evidence of
real-world registry performance.

## 4. Preserve or resume a run

Kaggle's `/kaggle/input` is read-only and `/kaggle/working` is temporary. The
notebook writes its run contract, checkpoints, best model, evaluation report,
and final archive beneath `/kaggle/working/trocr-runs/`. Before ending a session,
save a Kaggle notebook version with outputs or download the completed archive.

Resume only from a complete prior run directory with the same run key and
`run-contract.json`. Restore that directory under
`/kaggle/working/trocr-runs/` before training, keep `RESUME_IF_AVAILABLE = True`,
and retain the same dataset, `RUN_TAG`, profile, seed, model revision, and
dependency/runtime versions. A prior output can be attached as a Kaggle Dataset,
but because attached inputs are read-only, copy the one exact run directory into
the writable working location first. Do not merge checkpoint files from
different run keys.

Exact repeat/resume is targeted only for the same GPU architecture, CUDA,
PyTorch, Transformers, and dependency versions recorded in the contract.
Floating-point results may differ on another stack even with the same seed.

## 5. Export the evaluated artifact

After the one-shot locked test, run the report and packaging cells. Keep the
model weights together with `evaluation-report.json`; the report records the
weights hash and dataset hashes. An unreported checkpoint is not the evaluated
artifact.

If a path fails, print `DATA_ROOT`, `RUN_DIR`, and the manifest candidates, then
verify spelling and nesting in Kaggle's Input panel. If a checkpoint is refused,
compare the prior and current `run-contract.json`; use a new `RUN_TAG` when the
configuration or dataset changed.
