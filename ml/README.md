# `ml/` — DermNet adapter + training (research track)

Turns the DermNet public atlas into this repo's manifest schema and trains a coarse
category classifier. **Separate from the app and from the clinical console corpus.**
Nothing here is wired into the triage UI.

```
ml/
  config.py           paths + hyperparams (env > config.json > defaults)
  vocab_map.py        DermNet's 23 folders -> triage vocabulary, with confidence
  dermnet_adapter.py  class folders -> manifest.csv / manifest.json / classes.json
  dataset.py          torch Dataset over any pipeline manifest
  train.py            training + eval (--smoke for a CPU sanity run)
  predict.py          offline top-k inspection of a checkpoint (NOT wired to the app)
  evaluate.py         honest test-split report: per-class, confusions, calibration
  kaggle/launch.py    push + run + fetch the GPU job headlessly from this machine
  kaggle/run_dermnet.py  the kernel entrypoint that runs on Kaggle
  KAGGLE.md           GPU run instructions (manual notebook path)
```

## Headless GPU run

Instead of clicking through a notebook, `ml/kaggle/launch.py` uploads the code as a
private dataset, pushes a private GPU kernel, polls it, and downloads `best.pt` +
`report.json` into `data/ml/kaggle_out/`.

```bash
# one-time: kaggle.com/settings/api -> Create New Token (gives a KGAT_… value).
# Paste it into a hidden prompt so it never lands in shell history:
printf 'Paste Kaggle token: '; read -rs T; echo; mkdir -p ~/.kaggle; \
  printf '%s' "$T" > ~/.kaggle/access_token; chmod 600 ~/.kaggle/access_token; unset T

npm run ml:kaggle -- --check        # verify creds + resolve a DermNet dataset
npm run ml:kaggle -- --dry-run      # build the payloads, push nothing
npm run ml:kaggle                   # full run (~40 min + queue time)
npm run ml:kaggle -- --fetch-only   # re-download a finished run
```

The token is read by the kaggle CLI from `~/.kaggle/access_token` (or legacy
`~/.kaggle/kaggle.json`) — never read, printed, or copied by this code, and never
committed. Datasets and kernels are both created **private** (`datasets create` is
private by default; `-u/--public` is never passed). Revoke the token from the same
Settings page at any time; it does not invalidate completed runs.

**A token that has been screenshotted or pasted anywhere should be revoked and
regenerated** — `kaggle config view` printing your username is local config, not
proof that auth works, so verify with `--check` instead.

## Verified local baseline (reference point, not a target)

`resnet18 · 160px · batch 64 · 2 epochs` on Apple MPS, full manifest
(train 12201 / val 2151 / test 3950):

| epoch | train loss | val top1 | val top3 | val macro-F1 | secs |
|---|---|---|---|---|---|
| 1 | 2.893 | 0.236 | 0.438 | 0.181 | 80 |
| 2 | 2.509 | 0.274 | 0.493 | 0.226 | 125 |

`TEST top1 0.281 · top3 0.483 · macro-F1 0.229 · balanced-acc 0.277 (n=3950)`

Deliberately undertrained — the point was to prove the non-smoke path end to end
(pretrained weight loading, per-epoch val, checkpointing, test eval, confusion
matrix). Metrics move in the right direction, so scale it up on a GPU rather than
tuning here. ~2 min/epoch at 160px on MPS ⇒ budget roughly 2–4 min/epoch for
resnet50 @224px on a Kaggle T4.

## Quickstart

```bash
python3 -m venv ml/.venv && ml/.venv/bin/pip install -r ml/requirements.txt

ml/.venv/bin/python ml/config.py             # show what paths/settings resolved
ml/.venv/bin/python ml/dermnet_adapter.py    # build the manifest (~19.5k images)
ml/.venv/bin/python ml/train.py --smoke      # CPU sanity run, ~5 s
```

Full training belongs on a GPU — see [KAGGLE.md](KAGGLE.md).

## Paths are configuration, never hardcoded

| setting | env var | local default | Kaggle |
|---|---|---|---|
| dataset root | `DERMNET_DIR` | `./data` | `/kaggle/input/dermnet` |
| outputs | `DERMA_ML_OUT` | `./data/ml` | `/kaggle/working/ml` |

Resolution order is **explicit flag > env var > `ml/config.json` > default**. Any
hyperparameter in `config.DEFAULTS` is also env-overridable by its upper-cased name
(`EPOCHS=20 MODEL=convnext_tiny BATCH_SIZE=64 ...`). `image_path` in the manifest is
stored **relative to the dataset root**, so the same manifest works on both machines.
Run `python3 ml/config.py` first — it prints every resolved value, which is cheaper
than discovering a bad path after the GPU clock starts.

## What DermNet is, in this pipeline

DermNet is a **public atlas**: folder-level categories, not per-case dermatologist
confirmations, and **no skin-tone labels at all**. The clinical pipeline requires a
dermatologist-confirmed label with `confirmation_method` and skin tone on every
exported case — no DermNet row can satisfy that. So every row is stamped:

| column | value | why |
|---|---|---|
| `label_authority` | `public_atlas` | never `dermatologist_confirmed` |
| `confirmation_method` | `none` | no KOH/biopsy/dermoscopy behind the label |
| `fitzpatrick` / `monk` | `unknown` | DermNet has no skin-tone labels |
| `skin_tone_known` | `0` | blocks any per-skin-tone metric |
| `gradable` | `ungraded` | no expert graded these images |
| `resolution` | `atlas_public` | how the label was arrived at |

This makes DermNet **pretraining / bootstrap data**. It never becomes ground truth
and never enters the clinical export — `config.out_dir()` hard-refuses to write into
`data/exports/`, and `console/shared/vocab.js` `validateLabel()` rejects
`fitzpatrick: unknown`, so an atlas row cannot pass as a confirmed label.

**Consequence to keep in mind:** no result from this corpus can support a fairness
claim across skin tones. The app's skin-tone reliability caveat and
placeholder-engine notice both stay exactly as they are.

## Labels: 23 coarse folders, not 23 diagnoses

Several folders bundle clinically distinct diseases (`Psoriasis pictures Lichen
Planus and related diseases`), so the trained model is a **category** model.
`vocab_map.py` records a `map_kind` per folder — `direct` (3), `coarse` (7),
`none` (13) — and refuses to guess where a folder mixes benign with malignant
(`Melanoma Skin Cancer Nevi and Moles` → `other`, not `suspected_malignancy`) or
spans several vocab keys at once (`Light Diseases and Disorders of Pigmentation`).

Default training target is `original_class` (23-way — what the data actually
supports). The triage-vocab mapping is opt-in: `LABEL_COLUMN=confirmed_diagnosis`.

**DermNet cannot teach:** `melasma`, `pih`, `pityriasis_versicolor`,
`tinea_steroid_modified`, `vitiligo` — i.e. the pigmentation and steroid-modified
presentations that matter most in Indian outpatient practice. Those need the
dermatologist-labelled console corpus.

## Splits, and the leakage the adapter caught

DermNet's official `test/` is preserved untouched; `val` is carved from `train/`
per class at 15% with `seed=42` (the same seed as `console/lib/export.js`).

Exact-duplicate images are dropped by content hash. On this copy of the dataset:

```
train 12201   val 2151   test 3950   excluded_duplicate 1257
```

**1257 duplicates (6.4%), of which 577 spanned train/ and test/.** Left in, those
577 would have leaked test images into training and inflated the reported test
score. When a duplicate spans splits the **train** copy is dropped so the test set
stays clean. Nothing is silently discarded — excluded rows remain in the manifest
with `split=excluded_duplicate` and an `excluded_reason`, and are counted in the
run summary. `--fast` skips hashing (faster, but then duplicates are **not**
detected — the summary says so).

## Boundaries this code does not cross

- No treatment, medicine, cream, or product output — the model emits category
  scores only.
- Not a diagnosis model, not validated; does not change the app's triage +
  education framing, its urgency assessment, or its red-flag escalation path.
- Public-atlas data only. Patient images collected by the console are a separate,
  consent-gated corpus and are not touched by anything in this directory.
