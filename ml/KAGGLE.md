# Kaggle GPU run — DermNet training

Same code, no edits. Only `DERMNET_DIR` changes.

## 1. Notebook setup
- **Add data** → search `dermnet` → add the DermNet dataset. Note the mount path
  it appears under (usually `/kaggle/input/dermnet`).
- **Settings** → Accelerator → **GPU T4 x2** (or P100). Internet **ON** for the
  first run only — `timm` downloads pretrained ImageNet weights.

## 2. Get the code in
Either upload this repo's `ml/` folder as a Kaggle Dataset, or clone if it's on git:

```python
!git clone https://github.com/<you>/derma-triage.git /kaggle/working/derma-triage
%cd /kaggle/working/derma-triage
!pip install -q timm
```

## 3. Build the manifest
```python
import os
os.environ["DERMNET_DIR"] = "/kaggle/input/dermnet"   # read-only mount
# out_dir auto-resolves to /kaggle/working/ml on Kaggle — no need to set it
!python ml/config.py            # confirm what resolved BEFORE spending GPU time
!python ml/dermnet_adapter.py   # ~19.5k images; hashing takes a few minutes
```

If the mount is nested (`/kaggle/input/<ds>/dermnet/train`), the adapter finds it —
`find_split_dirs` tolerates one level of nesting. If `config.py` prints
`UNRESOLVED`, point `DERMNET_DIR` at the directory that directly contains
`train/` and `test/`.

## 4. Train
```python
!python ml/train.py --model resnet50 --epochs 15 --batch-size 64 --img-size 224
```

Roughly ~2–3 min/epoch on a T4 at 224px/batch 64 with 13k train images, so 15
epochs ≈ 40 min — inside Kaggle's session limit. Bigger backbones to try once the
baseline is in: `convnext_tiny`, `efficientnet_b3`, `swin_tiny_patch4_window7_224`.

Outputs land in `/kaggle/working/ml/runs/`:
- `best.pt` — best-val-macro-F1 checkpoint (carries `classes` + `img_size`, so
  inference never has to re-derive the label order)
- `report.json` — per-epoch history + final test metrics + the caveat block

Download `best.pt` from the notebook's Output tab before the session expires.

## 5. Read the numbers honestly
- **macro-F1 / balanced accuracy** are the headline; the classes are skewed ~6:1
  so plain top-1 flatters the big folders.
- **top-3** is the triage-relevant number — the product shows a shortlist.
- Sanity floor: a 2-epoch resnet18 @160px on this exact manifest scored
  test top1 0.281 / top3 0.483 / macro-F1 0.229 (see `ml/README.md`). A 15-epoch
  resnet50 @224px should land clearly above that. Published DermNet numbers vary
  widely because the folders overlap clinically — treat any figure as a sanity
  range, not a target.
- If your run lands *below* that floor, suspect config (wrong `label_column`,
  `--no-pretrained`, or a bad `DERMNET_DIR`) before tuning hyperparameters.
- **No per-skin-tone metric exists for this corpus** — DermNet has no skin-tone
  labels. Nothing in this run supports a fairness claim, and the app's skin-tone
  reliability caveat stays exactly as it is.

## What this model may and may not be used for
It is a coarse **category** model trained on public-atlas folder labels, several
of which bundle multiple diseases. It is **not** a diagnosis model and must not be
wired into the app as one, must not produce treatment/product suggestions, and does
not remove the urgency + red-flag escalation path. See `ml/README.md`.
