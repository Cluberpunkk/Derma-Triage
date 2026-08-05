"""ml/resolution_buckets.py
=========================================================================
Is the mimicry ceiling imposed by the TASK, or by the CORPUS?

The FixRes-broken ablation could not answer this because it changed the model's
operating point. This asks a cleaner question with no such confound: the model runs
at a single fixed 224px throughout, and we bucket the test images by the resolution
of the SOURCE FILE.

  higher source resolution -> more real detail survives the resize to 224
  if discrimination improves monotonically across buckets, the ceiling is IMAGE
  QUALITY, and our blockers are measured on a corpus that structurally cannot
  support the distinction
  if it is flat, the mimicry reading is strengthened: the information is not there
  at any resolution this corpus offers

CONTROLLING FOR CLASS COMPOSITION
---------------------------------
Buckets differ in which diseases they contain, and some classes are simply
photographed better. Reporting raw per-bucket accuracy would partly measure that.
So every headline number is CLASS-REWEIGHTED: per-class metrics are computed inside
the bucket, then averaged with the class weights of the whole test split. Raw
numbers are printed alongside so the size of the composition effect is visible.

The two lesion-level metrics need no reweighting by construction: SK<->AK/BCC
confusion is computed only over SK and AK/BCC images, and the pigmented flag rate
only over benign pigmented images.

    ml/.venv/bin/python ml/resolution_buckets.py data/ml/runs_final/best.pt
=========================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import runmeta  # noqa: E402
from dataset import ManifestImageDataset  # noqa: E402
from pigmentation_audit import BENIGN_PIGMENTED  # noqa: E402
from resolution_ablation import AKBCC, SK  # noqa: E402
from train import load_checkpoint  # noqa: E402
from urgency import (GROUPS_YAML, Platt, Temperature, load_groups, logits_labels,  # noqa: E402
                     tier_of)

BUCKETS = [(0, 400, "<400"), (400, 600, "400-600"), (600, 900, "600-900"),
           (900, 10 ** 9, ">900")]
MIN_PER_CLASS = 5           # a class must have this many in a bucket to be reweighted


def short_side(p: Path) -> int:
    try:
        with Image.open(p) as im:
            return min(im.width, im.height)
    except Exception:  # noqa: BLE001
        return -1


def reweighted_top1(pred, y, mask, class_weights, n_classes) -> tuple[float, float, int]:
    """(class-reweighted top-1, raw top-1, classes used) inside `mask`."""
    raw = float((pred[mask] == y[mask]).mean()) if mask.sum() else float("nan")
    num = den = 0.0
    used = 0
    for c in range(n_classes):
        m = mask & (y == c)
        if m.sum() < MIN_PER_CLASS:
            continue
        acc = float((pred[m] == c).mean())
        w = class_weights[c]
        num += w * acc
        den += w
        used += 1
    return (num / den if den else float("nan")), raw, used


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--manifest"), ap.add_argument("--root"), ap.add_argument("--device")
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()

    ckpt = Path(a.checkpoint).resolve()
    device = config.resolve_device(a.device)
    model, ck = load_checkpoint(ckpt, device)
    classes = ck["classes"]
    ci = {c: i for i, c in enumerate(classes)}
    lc = ck.get("label_column", "original_class")
    manifest_dir = Path(a.manifest) if a.manifest else config.out_dir() / "dermnet"
    root = config.dataset_root(a.root)
    groups = load_groups(GROUPS_YAML)

    val = ManifestImageDataset(manifest_dir, root, "val", lc, img_size=ck["img_size"])
    te = ManifestImageDataset(manifest_dir, root, "test", lc, img_size=ck["img_size"])
    frozen = runmeta.freeze(manifest_dir, "test", te.rows)
    meta = runmeta.manifest(tool="resolution_buckets.py", checkpoint=ckpt,
                            files={"groups": str(GROUPS_YAML)},
                            extra={"frozen_test": frozen, "eval_size_fixed": ck["img_size"],
                                   "buckets": [b[2] for b in BUCKETS]})
    runmeta.print_header(meta, frozen)
    print(f"\nModel runs at a FIXED {ck['img_size']}px throughout — no FixRes confound.")

    lv, yv = logits_labels(model, DataLoader(val, batch_size=a.batch_size), device)
    lt, yt = logits_labels(model, DataLoader(te, batch_size=a.batch_size), device)
    temp = Temperature().fit(lv, yv)
    soon = torch.tensor([i for i, c in enumerate(classes) if tier_of(groups, c)[0] == "soon"])
    platt = Platt().fit(temp(lv).softmax(1)[:, soon].sum(1), torch.isin(yv, soon))
    p = temp(lt).softmax(1)
    score = platt(p[:, soon].sum(1)).numpy()
    pred = p.argmax(1).numpy()
    y = yt.numpy()

    print(f"reading source resolutions for {len(te.rows)} images…")
    ss = np.array([short_side(root / r["image_path"]) for r in te.rows])
    n = len(y)
    ranked = np.empty(n, dtype=int)
    ranked[np.argsort(-score)] = np.arange(n)
    top10 = ranked < int(round(0.10 * n))
    is_urgent = np.isin(y, soon.numpy())
    pig_idx = [ci[c] for c in BENIGN_PIGMENTED if c in ci]
    pig_mask = np.isin(y, pig_idx)
    sk_ak_mask = np.isin(y, [ci[SK], ci[AKBCC]])
    class_weights = Counter(y.tolist())

    print(f"\nsource short side: median {np.median(ss[ss > 0]):.0f}px  "
          f"range {ss[ss > 0].min()}-{ss.max()}")

    rows = []
    print(f"\n{'bucket':<10} {'n':>5} {'top1(rw)':>9} {'top1(raw)':>10} {'cls':>4} "
          f"{'pigFlag':>8} {'SK<->AK':>8} {'urgCap10':>9}")
    for lo, hi, name in BUCKETS:
        m = (ss >= lo) & (ss < hi) & (ss > 0)
        if m.sum() < 30:
            print(f"{name:<10} {int(m.sum()):>5}   (too few, skipped)")
            continue
        rw, raw, used = reweighted_top1(pred, y, m, class_weights, len(classes))
        mp = m & pig_mask
        pig_flag = float((score[mp] >= 0.12).mean()) if mp.sum() else float("nan")
        msa = m & sk_ak_mask
        conf = float(np.mean(
            ((y[msa] == ci[SK]) & (pred[msa] == ci[AKBCC])) |
            ((y[msa] == ci[AKBCC]) & (pred[msa] == ci[SK]))
        )) if msa.sum() else float("nan")
        mu = m & is_urgent
        cap = float(top10[mu].mean()) if mu.sum() else float("nan")
        rows.append({"bucket": name, "n": int(m.sum()), "top1_reweighted": rw,
                     "top1_raw": raw, "classes_used": used,
                     "n_pigmented": int(mp.sum()), "pigmented_flag_rate": pig_flag,
                     "n_sk_akbcc": int(msa.sum()), "sk_akbcc_confusion_rate": conf,
                     "n_urgent": int(mu.sum()), "urgent_top10_capture": cap,
                     "median_short_side": float(np.median(ss[m]))})
        print(f"{name:<10} {int(m.sum()):>5} {rw:>9.4f} {raw:>10.4f} {used:>4} "
              f"{pig_flag:>8.4f} {conf:>8.4f} {cap:>9.4f}")

    print("\n  top1(rw) = class-reweighted to the whole-split class mix; top1(raw) is not.")
    print("  A gap between them is the composition effect we are controlling for.")

    # ---- monotonicity verdict ----
    valid = [r for r in rows if not np.isnan(r["top1_reweighted"])]
    verdict = "insufficient buckets"
    if len(valid) >= 3:
        t = [r["top1_reweighted"] for r in valid]
        c = [r["sk_akbcc_confusion_rate"] for r in valid]
        f = [r["pigmented_flag_rate"] for r in valid]
        mono_t = all(t[i] <= t[i + 1] for i in range(len(t) - 1))
        span_t = t[-1] - t[0]
        span_c = c[-1] - c[0]
        span_f = f[-1] - f[0]
        print(f"\n  lowest -> highest bucket: top1(rw) {span_t:+.4f}, "
              f"SK<->AK/BCC confusion {span_c:+.4f}, pigmented flag {span_f:+.4f}")
        if mono_t and span_t > 0.05:
            verdict = (
                f"CORPUS-LIMITED. Class-reweighted top-1 rises monotonically with source "
                f"resolution ({span_t:+.4f} across buckets), so discrimination improves when "
                f"more real detail survives. The blockers are measured on a corpus that "
                f"structurally under-supports the distinction, and full-resolution phone "
                f"photos have headroom this dataset cannot show.")
        elif abs(span_t) <= 0.05 and abs(span_c) <= 0.05:
            verdict = (
                f"FLAT — TASK-LIMITED. Discrimination does not improve with source resolution "
                f"(top1(rw) {span_t:+.4f}, SK<->AK/BCC confusion {span_c:+.4f}). This "
                f"strengthens the mimicry reading: the separating information is absent at "
                f"every resolution this corpus offers, so better capture alone will not fix "
                f"Blocker 1.")
        else:
            verdict = (
                f"MIXED. top1(rw) {span_t:+.4f} but not monotonic, SK<->AK/BCC "
                f"{span_c:+.4f}. Resolution is not the dominant factor; treat both the "
                f"corpus-limited and task-limited readings as live.")
    print("\n=== VERDICT ===\n  " + verdict.replace(". ", ".\n  "))

    out = {**meta, "buckets": rows, "verdict": verdict,
           "caveat": "Source resolution correlates with the era and source of a photo, so "
                     "it is not a clean instrument; class reweighting removes disease-mix "
                     "effects but not photographer or equipment effects."}
    pth = ckpt.parent / "resolution_buckets.json"
    pth.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
