# Synthetic Handwriting Dataset Generator

This project creates cropped synthetic handwriting images and reconciles consented mock-handwriting samples for TrOCR experiments on document fields. It is a dataset-building tool; the Kaggle notebook under `kaggle-code/` performs model training.

Synthetic output is an approximation, not evidence of authentic historical handwriting frequencies. Inspect the generated montage and measure performance on a separately collected, writer-held-out real test set before making claims about real documents.

---

## Key Features & Highlights

- **⚡ Multi-Threaded Parallel Engine**: Uses a thread-pool worker architecture (`ThreadPoolExecutor`) auto-scaling across multi-core CPUs (up to 16 worker threads), delivering generation speeds of **130 to 350+ images/second**.
- **🔒 100% Deterministic & Reproducible**: Parallel rendering uses per-sample thread-local random seeds and sequential manifest sorting, guaranteeing byte-identical datasets for any given seed.
- **🛡️ Fail-Safe Short-Label Protection**: Length-aware stroke damage safeguards protect 1–3 character fields (`age`, `sex`, `character`, `numeric`) from being erased by heavy erosion. Progressive stochastic backoff guarantees zero pipeline crashes even on **500,000+ image runs**.
- **🎨 Interactive Responsive Desktop GUI**: Built with Python Tkinter featuring dual scrollable column viewports, an **Interactive Live Preview** with `🎲 Re-roll`, a **Preflight Resource Estimator** (calculating folder space, ZIP size, and duration beforehand), **Custom Field Weights**, **Custom Split Ratios**, and **Preset Management**.

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
2. Health Check: Run `python -m unittest discover -s tests -v` to ensure all 175 unit tests pass cleanly.
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
- **Preset Management**: Load, save, export, and delete custom degradation presets (`.json` files stored under `resources/presets/`).
- **Custom Field Weights & Split Ratios**: Adjust frequency sliders for individual field types (`full_name`, `date_written`, `age`, etc.) and set exact Train / Val / Test percentages.
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

# Package as ZIP archive automatically with .zip.sha256 checksum sidecar
python -m src.generate_synthetic --count 10000 --zip

# Confirmation prompt override for large runs (100k+ samples)
python -m src.generate_synthetic --count 100000 --yes
```

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

Run the complete test suite (175+ automated unit tests):

```bash
python -m unittest discover -s tests -v
```

Additional dev checks:
```bash
python -m ruff check .
```

---

## Governance & Privacy

Always obtain documented consent before processing real handwriting scans. Keep identity records separate from image datasets and enforce role-based access control.
