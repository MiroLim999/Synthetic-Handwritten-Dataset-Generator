# Training TrOCR on Google Colab (free T4 GPU)

This folder has `train_trocr_colab.ipynb` — a ready-to-run notebook that
fine-tunes `microsoft/trocr-base-handwritten` on your generated dataset.

## Step 1 — Zip your dataset

On your PC, zip the **splits** folder (the one with `train/`, `val/`, `test/`,
and `labels.csv`):

- Right-click `dataset\splits` → **Send to → Compressed (zipped) folder**
- You'll get `splits.zip`

> Tip: a 60k dataset is ~3–6 GB. If upload is slow, you can zip just
> `dataset/synthetic` instead and adjust the paths in the notebook.

## Step 2 — Upload to Google Drive

Put `splits.zip` somewhere in your Google Drive, e.g. `MyDrive/splits.zip`.

## Step 3 — Open the notebook in Colab

1. Go to https://colab.research.google.com
2. **File → Upload notebook** → choose `train_trocr_colab.ipynb`
3. **Runtime → Change runtime type → T4 GPU**

## Step 4 — Run the cells top to bottom

- Cell 4 mounts your Drive.
- Cell 4 (unzip): make sure `DATA_ZIP` matches where you put the zip.
- Cell 5 (CONFIG): make sure `DATA_DIR` matches where it extracted
  (usually `/content/data/splits`).
- Then just run each cell. Stage 1 trains on the synthetic data and saves the
  model to your Drive.

## Step 5 — Later: real handwriting (Stage 2)

When you've collected real samples, zip their `train/val/test` + `labels.csv`,
upload, set `REAL_DIR` and `RUN_STAGE2 = True` in the Stage 2 cell, and run it.
It continues from the Stage-1 model with a lower learning rate.

## If the Colab session disconnects (free tier)

Checkpoints are saved to your Drive **every epoch** (in `..._stage1_ckpt`).
To resume after a disconnect:

1. Reconnect / open the notebook again, set runtime to **T4 GPU**.
2. Re-run cells 1-9 (install, mount, unzip, config, model, datasets, trainer).
3. Run the training cell (11). It detects the last checkpoint on Drive and
   **continues from where it stopped** — you don't lose finished epochs.

The "before" accuracy is also saved to Drive, so the before/after comparison
still works after a resume.

## Before / after accuracy

- Cell 10 measures the **base model** accuracy (before fine-tuning).
- Cell 13 measures it **after** Stage 1 and prints a before -> after table
  (Character Error Rate and exact-match accuracy).
- Stage 2 does the same on your **real** validation data.

## Notes

- **fp16 + batch size 8** fits comfortably on a T4 (16 GB). Lower `BATCH_SIZE`
  if you ever see out-of-memory.
- Validation uses generation (slow), so the notebook caps val at `VAL_SUBSET`
  (1500) for speed. Increase if you want.
- Your **real** test set is the honest accuracy measure. Synthetic accuracy
  will look high — don't rely on it alone.
- The trained model is saved to your Drive (`trocr_maasin_stage1`), so it
  survives the Colab session ending.
