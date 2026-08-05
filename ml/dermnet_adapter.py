"""ml/dermnet_adapter.py
=========================================================================
Maps DermNet's class-folder layout into the SAME manifest schema the clinical
pipeline exports (console/lib/export.js), so one training loader reads either
corpus — while keeping the two corpora provably separate.

  <root>/train/<Class Name>/*.jpg        ->  manifest.csv / manifest.json
  <root>/test/<Class Name>/*.jpg             classes.json, dataset_card.md

WHAT THIS DATA IS (and is not)
------------------------------
DermNet is a PUBLIC ATLAS. Its labels are folder-level atlas categories, not
per-case dermatologist confirmations, and it carries NO skin-tone labels at all.
The clinical pipeline's rules — dermatologist-confirmed label required,
confirmation_method and skin tone mandatory — cannot be satisfied by any DermNet
row. So every row here is stamped:

    label_authority = public_atlas      (never dermatologist_confirmed)
    confirmation_method = none
    fitzpatrick = unknown, monk = unknown, skin_tone_known = 0
    gradable = ungraded                 (no expert graded these images)

That makes DermNet PRETRAINING / BOOTSTRAP data. It never becomes ground truth,
never enters the clinical export, and cannot support any per-skin-tone fairness
claim. config.out_dir() refuses to write into data/exports for the same reason.

SPLITS
------
DermNet's official test/ split is preserved as `test` (never re-shuffled), and
`val` is carved out of train/ per class with a fixed seed (SEED matches
console/lib/export.js). Exact-duplicate images are dropped — DermNet mirrors
contain repeats, and a train/test repeat would silently inflate accuracy. When a
duplicate spans splits the TRAIN copy is dropped so test integrity wins.

Nothing is silently discarded: dropped rows stay in the manifest with
split = excluded_duplicate / excluded_unreadable and are counted in the summary.

Usage
-----
    python3 ml/dermnet_adapter.py                  # uses ml/config.py resolution
    DERMNET_DIR=/kaggle/input/dermnet python3 ml/dermnet_adapter.py
    python3 ml/dermnet_adapter.py --limit-per-class 20 --out /tmp/smoke   # tiny slice
    python3 ml/dermnet_adapter.py --fast           # skip hashing/decode verification
=========================================================================
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import vocab_map  # noqa: E402

DATASET_NAME = "dermnet"
LABEL_AUTHORITY = "public_atlas"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Columns 1-11 are byte-identical to COLS in console/lib/export.js so a loader can
# read either manifest. The rest are provenance columns this corpus needs.
CLINICAL_COLS = ["case_id", "image_path", "confirmed_diagnosis", "confirmation_method",
                 "fitzpatrick", "monk", "body_site", "prior_steroid_use", "gradable",
                 "split", "resolution"]
PROVENANCE_COLS = ["source", "label_authority", "original_class", "original_class_id",
                   "map_kind", "map_note", "skin_tone_known", "origin_split",
                   "image_sha256", "excluded_reason"]
COLS = CLINICAL_COLS + PROVENANCE_COLS


def _rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _case_id(rel: str) -> str:
    """Stable id derived from the dataset-relative path: same id locally and on Kaggle."""
    return f"dermnet_{hashlib.sha256(rel.encode()).hexdigest()[:12]}"


def scan(root: Path, limit_per_class: int | None = None) -> tuple[list[dict], list[str]]:
    """Walk the class folders and build one manifest row per image (pre-split)."""
    split_dirs = config.find_split_dirs(root)
    if not split_dirs:
        raise SystemExit(f"{root} has no train/ + test/ class folders")
    train_dir, test_dir = split_dirs

    classes = sorted({p.name for d in (train_dir, test_dir) for p in d.iterdir() if p.is_dir()})
    class_id = {c: i for i, c in enumerate(classes)}

    rows: list[dict] = []
    for origin_split, sdir in (("train", train_dir), ("test", test_dir)):
        for cls in classes:
            cdir = sdir / cls
            if not cdir.is_dir():
                continue
            files = sorted(p for p in cdir.iterdir()
                           if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
            if limit_per_class:
                files = files[:limit_per_class]
            key, kind, note = vocab_map.map_class(cls)
            for f in files:
                rel = _rel_posix(f, root)
                rows.append({
                    # --- clinical-schema columns ---
                    "case_id": _case_id(rel),
                    "image_path": rel,
                    "confirmed_diagnosis": key,
                    "confirmation_method": "none",
                    "fitzpatrick": "unknown",       # DermNet has no skin-tone labels
                    "monk": "unknown",
                    "body_site": vocab_map.body_site(cls),
                    "prior_steroid_use": "unknown",
                    "gradable": "ungraded",         # nobody graded these images
                    "split": "",                    # assigned below
                    "resolution": "atlas_public",   # how the label was resolved
                    # --- provenance ---
                    "source": DATASET_NAME,
                    "label_authority": LABEL_AUTHORITY,
                    "original_class": cls,
                    "original_class_id": class_id[cls],
                    "map_kind": kind,
                    "map_note": note,
                    "skin_tone_known": 0,
                    "origin_split": origin_split,
                    "image_sha256": "",
                    "excluded_reason": "",
                })
    return rows, classes


def hash_and_verify(rows: list[dict], root: Path, fast: bool = False) -> None:
    """Fill image_sha256 and flag images that will not decode.

    One pass over the bytes: hash them, then let PIL verify the same buffer. `fast`
    skips this entirely (manifest still valid; duplicates are then NOT detected).
    """
    if fast:
        return
    try:
        import io

        from PIL import Image
    except ImportError:
        print("  (pillow missing — skipping hash/verify; duplicates will NOT be detected)")
        return

    total = len(rows)
    for i, r in enumerate(rows, 1):
        p = root / r["image_path"]
        try:
            data = p.read_bytes()
            r["image_sha256"] = hashlib.sha256(data).hexdigest()
            Image.open(io.BytesIO(data)).verify()
        except Exception as e:  # noqa: BLE001 — any unreadable/corrupt file
            r["excluded_reason"] = f"unreadable: {type(e).__name__}"
        if i % 2000 == 0 or i == total:
            print(f"  hashed/verified {i}/{total}")


def assign_splits(rows: list[dict], val_fraction: float, seed: int) -> None:
    """test = DermNet's official test/ (untouched). val = per-class slice of train/.

    Deterministic: per-class shuffle seeded with (seed, class name).
    """
    unreadable = [r for r in rows if r["excluded_reason"]]
    for r in unreadable:
        r["split"] = "excluded_unreadable"

    usable = [r for r in rows if not r["excluded_reason"]]

    # --- exact-duplicate removal; test copies always win over train copies ---
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for r in usable:
        if r["image_sha256"]:
            by_hash[r["image_sha256"]].append(r)
    dup_cross, dup_within = 0, 0
    for _h, group in by_hash.items():
        if len(group) < 2:
            continue
        # keep a test copy if one exists, else the first train copy (sorted order)
        group.sort(key=lambda r: (r["origin_split"] != "test", r["image_path"]))
        keeper = group[0]
        for r in group[1:]:
            r["split"] = "excluded_duplicate"
            r["excluded_reason"] = f"duplicate of {keeper['case_id']}"
            if r["origin_split"] != keeper["origin_split"]:
                dup_cross += 1
            else:
                dup_within += 1
    if dup_cross or dup_within:
        print(f"  duplicates dropped: {dup_within} within-split, "
              f"{dup_cross} CROSS-SPLIT (train/test leakage prevented)")

    live = [r for r in usable if not r["split"]]

    for r in (r for r in live if r["origin_split"] == "test"):
        r["split"] = "test"

    per_class: dict[str, list[dict]] = defaultdict(list)
    for r in (r for r in live if r["origin_split"] == "train"):
        per_class[r["original_class"]].append(r)

    for cls in sorted(per_class):
        members = sorted(per_class[cls], key=lambda r: r["image_path"])
        rnd = random.Random(f"{seed}:{cls}")
        rnd.shuffle(members)
        # at least 1 val image per class whenever the class has >1 train image
        n_val = max(1, round(len(members) * val_fraction)) if len(members) > 1 else 0
        for i, r in enumerate(members):
            r["split"] = "val" if i < n_val else "train"


def write_outputs(rows: list[dict], classes: list[str], root: Path, out: Path,
                  seed: int, val_fraction: float) -> dict:
    out.mkdir(parents=True, exist_ok=True)

    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    counts = Counter(r["split"] for r in rows)
    per_class_split = defaultdict(Counter)
    for r in rows:
        per_class_split[r["original_class"]][r["split"]] += 1

    summary = {
        "dataset": DATASET_NAME,
        "label_authority": LABEL_AUTHORITY,
        "dataset_root": str(root),
        "image_path_is_relative_to": "dataset_root",
        "seed": seed,
        "val_fraction": val_fraction,
        "total_rows": len(rows),
        "splits": dict(sorted(counts.items())),
        "skin_tone_labels": False,
        "usable_for_skin_tone_fairness_metrics": False,
        "clinical_ground_truth": False,
        "num_original_classes": len(classes),
        "classes": classes,
        "map_kind_counts": dict(sorted(Counter(r["map_kind"] for r in rows).items())),
        "vocab_keys_not_covered_by_dermnet": vocab_map.UNCOVERED_KEYS,
        "per_class": {c: dict(sorted(v.items())) for c, v in sorted(per_class_split.items())},
    }
    (out / "manifest.json").write_text(json.dumps(summary, indent=2))

    # Frozen class index: the loader MUST use this, not whatever classes happen to
    # appear in a split, or train/val label spaces silently diverge.
    (out / "classes.json").write_text(json.dumps({
        "original_class": classes,
        "confirmed_diagnosis": sorted({r["confirmed_diagnosis"] for r in rows}),
    }, indent=2))

    (out / "dataset_card.md").write_text(_dataset_card(summary))
    return summary


def _dataset_card(s: dict) -> str:
    trainable = {k: v for k, v in s["splits"].items() if not k.startswith("excluded")}
    return f"""# DermNet adapter output — dataset card

**Source:** DermNet public atlas · **label_authority:** `{s['label_authority']}`
**Rows:** {s['total_rows']} · **classes:** {s['num_original_classes']} · **seed:** {s['seed']}

## What these labels are
Folder-level ATLAS categories, not per-case dermatologist confirmations. Several
folders bundle multiple diseases (e.g. "Psoriasis pictures Lichen Planus and
related diseases"), so a model trained here is a coarse CATEGORY model — not a
diagnosis model. Consistent with the app staying triage + education.

## Limits that are structural, not fixable by tuning
- **No skin-tone labels.** Every row is `fitzpatrick=unknown`, `monk=unknown`,
  `skin_tone_known=0`. No per-skin-tone fairness metric can be computed from this
  corpus, and the app's skin-tone reliability caveat therefore stays as-is.
- **Not clinical ground truth.** `confirmation_method=none`, `gradable=ungraded`.
  No row is eligible for the dermatologist-confirmed clinical export.
- **Atlas skew.** Public atlases over-represent classic/photogenic presentations
  and lighter skin; real outpatient distributions differ.

## Vocab keys DermNet cannot teach
{', '.join('`' + k + '`' for k in s['vocab_keys_not_covered_by_dermnet']) or '(none)'}
These need the dermatologist-labelled console corpus.

## Splits
{json.dumps(trainable, indent=2)}

Excluded (kept in manifest, never loaded):
{json.dumps({k: v for k, v in s['splits'].items() if k.startswith('excluded')}, indent=2)}

`test` is DermNet's official test/ split, never re-shuffled. `val` is carved from
train/ per class at {s['val_fraction']:.0%} with seed {s['seed']}. Exact duplicates are dropped;
cross-split duplicates drop the TRAIN copy so the test set stays clean.

## Mapping confidence (`map_kind`)
{json.dumps(s['map_kind_counts'], indent=2)}
`direct` = folder ≈ vocab entity · `coarse` = folder contains it plus others ·
`none` = no defensible equivalent, mapped to `other`. Default training target is
`original_class`; the vocab mapping is opt-in via `LABEL_COLUMN=confirmed_diagnosis`.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a pipeline manifest from DermNet class folders.")
    ap.add_argument("--root", help="dataset root (default: ml/config.py resolution)")
    ap.add_argument("--out", help="output dir (default: ml/config.py out_dir()/dermnet)")
    ap.add_argument("--limit-per-class", type=int, help="tiny slice for smoke tests")
    ap.add_argument("--fast", action="store_true", help="skip hashing/decode verification")
    ap.add_argument("--val-fraction", type=float)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    root = config.dataset_root(args.root)
    out = Path(args.out).expanduser().resolve() if args.out else config.out_dir() / DATASET_NAME
    seed = config.setting("seed", args.seed)
    val_fraction = config.setting("val_fraction", args.val_fraction)

    print(f"DermNet root : {root}")
    print(f"Output       : {out}")
    if args.limit_per_class:
        print(f"Slice        : first {args.limit_per_class} images per class per split")

    rows, classes = scan(root, args.limit_per_class)
    print(f"Scanned      : {len(rows)} images across {len(classes)} classes")
    if not rows:
        raise SystemExit("no images found — check the dataset root")

    unknown = sorted({r["original_class"] for r in rows if r["map_kind"] == "unknown_folder"})
    if unknown:
        print("  WARNING: folders not in the mapping table (mapped to 'other'): "
              + ", ".join(unknown))

    hash_and_verify(rows, root, args.fast)
    assign_splits(rows, val_fraction, seed)
    summary = write_outputs(rows, classes, root, out, seed, val_fraction)

    print("\nSplits:")
    for k, v in summary["splits"].items():
        print(f"  {k:<22} {v}")
    print(f"\nWrote manifest.csv, manifest.json, classes.json, dataset_card.md -> {out}")
    print("NOTE: public-atlas pretraining data — no skin-tone labels, not clinical ground truth.")


if __name__ == "__main__":
    main()
