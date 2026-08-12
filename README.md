# Civil Registry Handwriting Dataset Toolkit

This project creates cropped synthetic handwriting images and reconciles
consented mock-handwriting samples for TrOCR experiments on Philippine civil
registry fields. It is a dataset-building tool; the Kaggle notebook under
`kaggle-code/` performs model training.

Synthetic output is an approximation, not evidence of authentic historical
handwriting frequencies. Inspect the generated montage and measure performance
on a separately collected, writer-held-out real test set before making claims
about real documents.

## Requirements

- Python 3.10 or newer
- Tkinter for the optional desktop GUI (normally included with Windows Python;
  some Linux distributions package it separately)
- At least one legally obtained `.ttf` or `.otf` font

The repository does **not** distribute handwriting font binaries. Windows may
use locally installed fallback fonts listed in `config.py`; Linux and macOS
users must normally add their own fonts. See
[`resources/fonts/README.md`](resources/fonts/README.md) before installing one.

## Clean setup

Create an isolated environment from the repository root.

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy a licensed font into `resources/fonts/`, record its provenance as described
in the font guide, and verify discovery:

```bash
python -c "from src.render import available_fonts; print(*available_fonts(), sep='\n')"
```

The dependency ranges are in `requirements.txt`. `constraints.txt` pins the
versions tested by CI; it is a constraints snapshot, not a hash-locked supply
chain lock. OpenCV is intentionally absent because this repository currently
has no scan-alignment or sheet-processing implementation. Faker is an optional
development dependency used only by `tools/expand_names.py`.

## Generate a dataset

```bash
# Next available dataset_NNN folder
python -m src.generate_synthetic --count 200

# Explicit safe name or number
python -m src.generate_synthetic --count 5000 --dataset 2 --seed 42
python -m src.generate_synthetic --count 200 --dataset pilot_run

# Damage profile and font-pool examples
python -m src.generate_synthetic --count 2000 --mode semi_broken_mixed
python -m src.generate_synthetic --count 5000 --font-style cursive

# Versioned writer/scan realism (these are the current defaults)
python -m src.generate_synthetic --count 5000 \
  --writer-profile writer_style_v1 \
  --augmentation-profile historical_scan_v1 \
  --samples-per-writer 32

# Publish both a folder and ZIP, or retain only the verified ZIP
python -m src.generate_synthetic --count 5000 --zip
python -m src.generate_synthetic --count 5000 --zip-only

# Jobs at or above the configured large-run threshold require confirmation
python -m src.generate_synthetic --count 100000 --yes
```

Dataset names are portable single directory names; absolute paths, separators,
`..`, drive-qualified names, and reserved Windows names are rejected. A run is
written to a private staging directory and published only after validation.
Existing datasets and archives are never silently overwritten.

Generation preflights the required resources, estimated peak disk/archive
space, and available free space before writing. Field/split planning and CSV
output are streamed, so memory does not grow with the manifest row count. The
versioned `writer_style_v1` profile gives repeated samples stable synthetic
writer traits, while `historical_scan_v1` defines the recorded augmentation
parameters. These profiles are engineered approximations rather than
empirically calibrated population distributions.

The optional GUI uses a light theme and exposes the sample count, dataset name,
name pool, damage mode, font style/group, real-data merge, ZIP packaging,
progress, cancellation, and safe dataset deletion. Launch it with:

```bash
python gui.py
```

On Windows, `Generate Images.bat` searches `.venv`, an active Conda environment,
the Python launcher, and `PATH`; it contains no machine-specific username or
environment path.

## Output contract

Each successful run appears under `dataset/datasets/`:

```text
dataset_NNN/
|-- train/*.png
|-- val/*.png
|-- test/*.png
|-- manifest.csv
|-- labels.csv
|-- evaluation-annotations.csv
|-- dataset-validation.json
|-- review-montage.jpg
|-- image-statistics.json
`-- run-metadata.json
```

`manifest.csv` is authoritative. It has exactly these version-1 columns:

| Column | Contract |
|---|---|
| `filename` | Globally unique direct-child `.png` filename; no path components |
| `label` | Non-empty ground-truth transcription |
| `split` | `train`, `val`, or `test` |
| `source` | `synthetic` or `real` |
| `field_type` | Required for synthetic rows; may be empty for real rows |
| `font` | Font filename for synthetic rows; empty for real rows |
| `sample_mode` | Synthetic damage mode; empty for real rows |
| `writer_id` | Required for real rows; empty for synthetic rows |
| `schema_version` | `1` |

`labels.csv` is a compatibility view containing only `filename,label,split`.
Do not treat it as a second source of truth. `evaluation-annotations.csv`
records per-sample seen/held-out conditions. `run-metadata.json` binds the seed,
effective configuration, source/resource/dependency versions, split policy,
manifest hash, and image-set hash to the run.

`image-statistics.json` summarizes a bounded sample of generated image geometry,
luminance, ink, and edge characteristics. When `dataset/real/images/` is
available, it also records a descriptive generated-versus-real comparison. This
helps spot domain gaps but is not proof that the two distributions are
equivalent. Writer profile IDs and effective parameters, augmentation profile
IDs and effective parameters, samples-per-writer policy, and resource estimates
are recorded in `run-metadata.json`.

Before publication, the validator checks the schema, path safety, duplicate,
missing and orphan files, split counts, image readability, contrast, ink
coverage, clipping, and dimensions. The result is written to
`dataset-validation.json`; inspect `review-montage.jpg` as a human quality gate.
A valid report does not prove that labels are semantically correct or that the
data resembles the intended deployment population.

ZIP creation validates archive members and CRCs before publication and writes
a sibling `<dataset>.zip.sha256` checksum. `--zip-only` deletes the source
folder only after the archive and checksum verify. Retain the checksum with any
copied archive and verify it before extraction or training.

## Merge consented real handwriting

This project does not include the previously described `make_sheets.py` or
`process_scans.py` workflow. Prepare already cropped PNGs yourself:

```text
dataset/real/
|-- images/
|   |-- writer001_001.png
|   `-- writer002_001.png
`-- labels.csv
```

The input CSV must contain exactly usable values for:

```text
filename,label,writer_id
```

Use an opaque, stable `writer_id`; never put a person's name, contact details,
or consent record in a filename or OCR label. Reconcile the input into the
latest or a selected dataset:

```bash
python -m src.build_splits
python -m src.build_splits --dataset dataset_014
```

The operation is transactional and idempotent. It persists writer assignments
in `real_writer_splits.csv`, keeps each writer in one split, and updates the
unified manifest, validation report, and run metadata. Removing a real row from
the input reconciles the corresponding dataset row; retain your controlled
source and consent records according to your approved policy.

## Interpreting evaluation results

- **Synthetic / in-distribution** measures generated examples that share the
  training domain. It is useful for debugging, not a deployment claim.
- **Synthetic / held-out** uses declared unseen fonts, formatting patterns, or
  degradation conditions. It tests controlled domain shift but remains synthetic.
- **Real / writer-held-out** contains mock handwriting from writers absent from
  training and validation. This is the primary estimate for new-writer
  generalization when collection and labeling are representative.

Use validation for model and hyperparameter selection. Touch the locked test
split only after choices are frozen, and report CER, WER, exact match, sample
count, source, field type, and held-out condition separately. Never combine
these domains into one unlabeled accuracy claim. See `kaggle-code/README.md` for
the training workflow.

## Real-data privacy and governance

Before collecting or processing handwriting, obtain documented informed
consent that covers the intended training, evaluation, retention, and sharing.
Confirm the lawful basis and institutional requirements that apply in your
jurisdiction; this section is operational guidance, not legal advice.

- Collect only mock responses needed for the task. Do not ingest official civil
  records or unrelated personal data without explicit authority.
- Keep identity/consent records separate from images. Use random writer IDs and
  restrict the re-identification key to authorized custodians.
- Encrypt source scans and backups, limit access by role, log exports, and set a
  documented retention deadline.
- Share only the minimum approved, de-identified artifact under terms that
  prohibit re-identification and unauthorized redistribution.
- Provide a withdrawal/deletion channel. Delete source scans, crops, exports,
  and feasible backups; rebuild affected manifests/datasets and document whether
  trained models can or must also be retired.
- Record collection version, consent scope, custodian, access approvals,
  retention date, deletion actions, and known limitations outside the public
  OCR manifest.

## Troubleshooting

**No fonts found.** Put at least one authorized `.ttf`/`.otf` directly in
`resources/fonts/`, restart a running GUI so its font cache is cleared, and run
the discovery command from the setup section. On non-Windows systems there is
no bundled fallback.

**A run stopped or the machine lost power.** Final dataset directories are
transactional; do not rename a hidden `.dataset_*.tmp-*` directory into place.
First confirm no generator process is active, then inspect hidden staging and
`.reserve` entries under `dataset/datasets/`. Remove only the exact stale entry
you have verified. A new run will not overwrite it.

**Kaggle cannot find the dataset.** Attach the generated dataset as a Kaggle
Dataset and set `DATA_ROOT` to the exact directory containing `manifest.csv`,
`dataset-validation.json`, and the three split directories. The notebook prints
candidate manifest locations but never auto-selects one.

**Kaggle checkpoint recovery.** `/kaggle/working` is temporary. Save the run as
a Kaggle output/version or download it before the session ends. To resume, put
the complete prior run directory back at the same run-key path and retain the
same dataset hashes, `RUN_TAG`, configuration, dependency versions, and model
revision. Never combine checkpoints from different run contracts.

## Development and automated checks

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python -m ruff check .
python -m mypy
python -m pip_audit -r requirements.txt
```

CI runs the unit, integration, security, reproducibility, and dataset-integrity
tests on Windows and Ubuntu. Ruff lint is repository-wide except for the
stateful Kaggle notebook. Type and formatter gates currently use an explicit
incremental baseline recorded in `pyproject.toml` and the workflow; expand that
baseline as existing files are normalized.

Third-party notices and the font-license recording policy are in `NOTICE.md`
and `LICENSES/README.md`. No project-wide source-code license is asserted by
those notices.
