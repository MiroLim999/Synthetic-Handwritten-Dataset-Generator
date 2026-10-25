# Synthetic Handwriting Dataset Generator

This project creates cropped synthetic handwriting images and reconciles consented mock-handwriting samples for TrOCR experiments on document fields. It is a dataset-building tool; the Kaggle notebook under `kaggle-code/` performs model training.

Synthetic output is an approximation, not evidence of authentic historical handwriting frequencies. Inspect the generated montage and measure performance on a separately collected, writer-held-out real test set before making claims about real documents.

---

## Key Features & Highlights

- **⚡ Parallel Engine**: Small runs (20 samples or fewer, or `--workers 1`) render on one thread; larger runs use a process pool sized to your CPU cores (`--workers N` to choose).
- **🔒 Deterministic & Reproducible**: Every sample has its own seed and the manifest is written in plan order, so a seed gives byte-identical datasets regardless of the worker count. Each dataset stores its effective `studio-recipe.json`; re-running it with the same seed and count reproduces the dataset.
- **🧪 Studio Recipes**: Every effect is a chance plus a min/max range. Recipes (presets) have optional sections for augmentation, stroke damage, character cutoff, and run settings, and work the same in the GUI and on the command line.
- **✂️ Real Character Cutoff**: Cuts are measured from the ink, so letters are actually clipped. Ink loss is capped, and every cut sample is recorded in `cutoff-annotations.csv`.
- **🛡️ Short-Label Protection**: Length-aware safeguards keep 1-3 character labels (`age`, `sex`, `character`) readable under stroke damage and cutoff, with progressive backoff on retries.
- **🎨 Desktop GUI**: Tkinter app with a live preview, a preflight estimator (disk space, ZIP size, duration), custom field weights and split ratios, and preset loading and saving.

---

## Requirements

- Python 3.10 or newer
- Tkinter for the optional desktop GUI (included with Windows Python)
- At least one legally obtained `.ttf` or `.otf` handwriting font

The repository does **not** distribute font binaries. Windows may use locally installed fallback fonts listed in `config.py`; Linux and macOS users can place their fonts in `resources/fonts/`. See [`resources/fonts/README.md`](resources/fonts/README.md).

---

## Quick Setup

Create an isolated virtual environment:

**Windows PowerShell:**
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify font discovery:
```bash
python -c "from src.render import available_fonts; print(*available_fonts(), sep='\n')"
```

---

## 🤖 Agentic AI Setup Instructions (Gemini Antigravity, Kiro AI, Cursor AI)

If you are using an Agentic AI coding assistant (such as **Google Gemini Antigravity**, **Kiro AI IDE**, **Cursor AI**, **Windsurf**, or **VS Code Copilot Workspace**), you can copy-paste the instructions below directly into your AI prompt window or agent prompt to let the AI set up, test, or generate datasets autonomously.

### 📋 Copy-Paste Prompt for AI Agents:

```text
Please set up and run the Synthetic Handwriting Dataset Generator repository:
1. Environment Setup: Create a Python virtual environment (`.venv`), activate it, upgrade pip, and install all dependencies from `requirements.txt`.
2. Health Check: Run `python -m unittest discover -s tests -v` to ensure all unit tests pass cleanly.
3. Dataset Generation: Run the synthetic generator to produce a dataset (e.g. `python -m src.generate_synthetic --count 1000 --seed 42`).
4. Verification: Verify that `manifest.csv`, `dataset-validation.json`, and images are present in the output folder under `dataset/datasets/`.
```

### 💻 One-Liner Execution for AI Terminal Tools:

**Windows PowerShell:**
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; python -m pip install --upgrade pip; python -m pip install -r requirements.txt; python -m unittest discover -s tests -v; python gui.py
```

**Linux / macOS:**
```bash
python3 -m venv .venv && source .venv/bin/activate && python -m pip install --upgrade pip && python -m pip install -r requirements.txt && python -m unittest discover -s tests -v && python gui.py
```

---

## Launch Desktop GUI

Launch the responsive desktop GUI using:

```bash
python gui.py
```

Or on Windows, simply double-click `Generate Images.bat`.

### GUI Capabilities:
- **Interactive Live Preview**: Preview handwriting fonts, text inputs, and degradation effects in real time with the `🎲 Re-roll` button.
- **Presets**: Load any built-in or user preset and save the current studio settings as a user preset. Loading applies only the sections a preset contains. See [Presets](#presets).
- **Custom Field Weights & Split Ratios**: The field-weight grid follows the sample mode (and says so when it replaces your edits); you can then edit individual field types (`full_name`, `date_written`, `age`, etc.) and set exact Train / Val / Test percentages.
- **Stroke damage** sliders only apply, and are only enabled, in the semi-broken sample modes.
- **Preflight Estimates Card**: Calculates estimated disk space, ZIP archive size, and generation time before starting.

---

## Command Line Usage

Generate synthetic samples directly from the terminal:

```bash
# Generate 200 samples in the next available dataset_NNN folder
python -m src.generate_synthetic --count 200

# Specify explicit dataset name and seed
python -m src.generate_synthetic --count 5000 --dataset pilot_run --seed 42

# Select damage profile and font style
python -m src.generate_synthetic --count 2000 --mode semi_broken_mixed --font-style cursive

# Versioned writer style and scan realism profiles (defaults: writer_style_v1, historical_scan_v1)
python -m src.generate_synthetic --count 5000 \
  --writer-profile writer_style_v1 \
  --augmentation-profile historical_scan_v1 \
  --samples-per-writer 32

# Custom split fractions (e.g. 70% Train, 15% Val, 15% Test)
python -m src.generate_synthetic --count 5000 --custom-split-fractions 0.70 0.15 0.15

# Custom field weights JSON string
python -m src.generate_synthetic --count 5000 --custom-field-weights '{"full_name": 10, "date_written": 5}'

# Real, ink-relative character cutoff: none, light, moderate or heavy
python -m src.generate_synthetic --count 2000 --cutoff moderate

# Start from a studio recipe (preset); explicit flags override its values
python -m src.generate_synthetic --count 2000 --recipe resources/presets/historical_faded.json
python -m src.generate_synthetic --count 2000 --recipe resources/presets/heavy_damage.json --cutoff none

# Print the effective recipe and exit without writing anything
python -m src.generate_synthetic --recipe resources/presets/historical_faded.json --print-recipe

# Reproduce an existing dataset from its own recipe (same seed and count)
python -m src.generate_synthetic --count 200 --seed 42 --recipe dataset/datasets/dataset_001/studio-recipe.json

# Choose the number of worker processes (output is identical for any count)
python -m src.generate_synthetic --count 5000 --workers 4

# Package as ZIP archive automatically with .zip.sha256 checksum sidecar
python -m src.generate_synthetic --count 10000 --zip

# Confirmation prompt override for large runs (100k+ samples)
python -m src.generate_synthetic --count 100000 --yes
```

Precedence is: built-in defaults < `--recipe` < explicit flags. Every flag and
the recipe are validated before an output folder is reserved, so a typo never
leaves a half-created dataset behind.

---

## Output Structure & Contract

Each successful run creates a self-contained dataset under `dataset/datasets/`:

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
|-- studio-recipe.json          (effective recipe; replay it to reproduce the run)
|-- cutoff-annotations.csv      (only when a cutoff other than none is used)
`-- run-metadata.json
```

`manifest.csv` is authoritative with the following schema:

| Column | Description |
|---|---|
| `filename` | Globally unique `.png` filename (`syn_000000.png`) |
| `label` | Non-empty ground-truth transcription |
| `split` | Assigned split (`train`, `val`, or `test`) |
| `source` | Source tag (`synthetic` or `real`) |
| `field_type` | Synthetic field category |
| `font` | Font filename used for rendering |
| `sample_mode` | Degradation profile applied |
| `writer_id` | Synthetic or real writer profile identifier |
| `schema_version` | Manifest contract version (`1`) |

`run-metadata.json` records generation provenance, `dataset-validation.json` verifies dataset integrity, and `image-statistics.json` documents brightness/contrast metrics.

`studio-recipe.json` is the fully resolved recipe the run used (all sections). Its SHA-256, name, source preset, and whether the run changed the source preset are recorded under `studio_recipe` in `run-metadata.json`.

`cutoff-annotations.csv` lists every sample that was cut, in manifest order:
`filename, split, cut_top, cut_bottom, cut_left, cut_right, ink_loss`. Each
`cut_*` value is the fraction of the ink box removed from that side (0 = not
cut) and `ink_loss` is the share of ink pixels removed. The file's hash is
recorded in `run-metadata.json`; dataset validation uses it to allow ink only
at the sides that were cut on purpose.

---

## Presets

Presets are Studio Recipe v2 JSON files with optional sections:
`augmentation`, `stroke_damage`, `cutoff`, and `run`. Loading a preset applies
only the sections it contains. The format, bounds, and precedence rules are
documented in [`resources/presets/README.md`](resources/presets/README.md).

- **Built-in presets** (`resources/presets/*.json`) are read-only.
- **User presets** are saved to `resources/presets/user/` (created on demand and
  not tracked by git). A name that clashes with a built-in is refused, and
  replacing a user preset asks first.
- Old flat presets (single-value sliders, no `schema_version`) still load; they
  are converted in memory, and saving always writes version 2.
- Any preset file can be used on the command line with `--recipe`.
- Duplicate, rename, delete, import, and export are available from Python
  (`src.studio_recipe`: `duplicate_preset`, `rename_user_preset`,
  `delete_user_preset`, `import_preset`, `export_preset`).

---

## Reconcile Real Handwriting Data & Evaluation

Prepare cropped PNG images and ground-truth labels under `dataset/real/`:

```text
dataset/real/
|-- images/
|   |-- writer001_001.png
|   `-- writer002_001.png
`-- labels.csv
```

Reconcile into datasets using:

```bash
python -m src.build_splits
```

Writer assignments are persisted in `real_writer_splits.csv` ensuring zero writer overlap across splits.

### Evaluation Domains
- **Synthetic / in-distribution**: Diagnostic check on synthetic training distribution.
- **Synthetic / held-out**: Evaluates synthetic generalization on unseen fonts and formats.
- **Real / writer-held-out**: Primary benchmark for evaluating model generalization on authentic new writers.

---

## Testing & Quality Assurance

Run the complete test suite:

```bash
python -m unittest discover -s tests -v
```

Additional dev checks:
```bash
python -m ruff check .
python -m mypy
# Output regression check: record once, then compare after changes.
python -m tools.golden_hashes --update
python -m tools.golden_hashes --check
# What each cutoff profile does (attempts, share cut, ink loss)
python -m tools.cutoff_stats
```

The golden hashes depend on your installed fonts, so the baseline is stored
locally in `dataset/.golden-hashes.json` and is not tracked.

---

## Governance & Privacy

Always obtain documented consent before processing real handwriting scans. Keep identity records separate from image datasets and enforce role-based access control.
