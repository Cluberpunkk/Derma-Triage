"""ml/traffic_weighted.py
=========================================================================
Converts confusion counts into expected errors per 1,000 uploads.

A confusion matrix is measured on DermNet's class mix, which is nothing like an app's
traffic. DermNet gives AK/BCC 7.2% of the test split; in a consumer app in India it
will be a fraction of a percent, while eczema/psoriasis/tinea will be most of the
volume. The two error directions therefore scale in OPPOSITE directions from what the
raw counts suggest:

  inflammatory -> AK/BCC    31 cases in test. Scales with a huge denominator, so it
                            becomes a flood of false alarms. Costly, not dangerous.
  AK/BCC -> inflammatory    16 cases in test. Scales with a tiny denominator, so it
                            stays small in absolute count — but each one is a missed
                            skin cancer. Rare and dangerous.

Reporting both as "counts per 1,000 uploads" makes them comparable in the only unit
that matters operationally.

Two error definitions are reported because they answer different questions:
  argmax     the model's top-1 prediction was the other class
  flag@0.12  the urgency gate fired (or failed to fire) at the high-recall point

    ml/.venv/bin/python ml/traffic_weighted.py data/ml/runs_final/best.pt
=========================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import runmeta  # noqa: E402
from dataset import ManifestImageDataset  # noqa: E402
from resolution_ablation import AKBCC, INFLAMMATORY  # noqa: E402
from shortcut_test import darkness_stats  # noqa: E402
from train import load_checkpoint  # noqa: E402
from urgency import (GROUPS_YAML, Platt, Temperature, load_groups, logits_labels,  # noqa: E402
                     tier_of)

INFLAMMATORY_SHARE = [0.50, 0.65, 0.80]
AKBCC_SHARE = [0.005, 0.010, 0.020]
UPLOADS = 1000
FLAG_T = 0.12


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
    meta = runmeta.manifest(tool="traffic_weighted.py", checkpoint=ckpt,
                            files={"groups": str(GROUPS_YAML)},
                            extra={"frozen_test": frozen, "uploads_basis": UPLOADS,
                                   "flag_threshold": FLAG_T})
    runmeta.print_header(meta, frozen)

    lv, yv = logits_labels(model, DataLoader(val, batch_size=a.batch_size), device)
    lt, yt = logits_labels(model, DataLoader(te, batch_size=a.batch_size), device)
    temp = Temperature().fit(lv, yv)
    soon = torch.tensor([i for i, c in enumerate(classes) if tier_of(groups, c)[0] == "soon"])
    platt = Platt().fit(temp(lv).softmax(1)[:, soon].sum(1), torch.isin(yv, soon))
    p = temp(lt).softmax(1)
    score = platt(p[:, soon].sum(1)).numpy()
    pred = p.argmax(1).numpy()
    y = yt.numpy()

    infl_idx = [ci[c] for c in INFLAMMATORY if c in ci]
    ak_i = ci[AKBCC]
    infl_mask = np.isin(y, infl_idx)
    ak_mask = y == ak_i

    dL = np.array([darkness_stats(root / r["image_path"])["delta_L"] for r in te.rows])
    cut = float(np.median(dL))
    ak_np = ak_mask & (dL < cut)

    # ---- per-case RATES measured on DermNet (these transfer; counts do not) ----
    r_infl_to_ak_argmax = float((pred[infl_mask] == ak_i).mean())
    r_infl_flagged = float((score[infl_mask] >= FLAG_T).mean())
    r_ak_to_infl_argmax = float(np.isin(pred[ak_mask], infl_idx).mean())
    r_ak_np_to_infl_argmax = float(np.isin(pred[ak_np], infl_idx).mean())
    r_ak_missed_flag = float((score[ak_mask] < FLAG_T).mean())
    r_ak_np_missed_flag = float((score[ak_np] < FLAG_T).mean())

    print(f"\n=== PER-CASE ERROR RATES (measured on DermNet; these are what transfer) ===")
    print(f"  inflammatory (n={int(infl_mask.sum())})")
    print(f"    predicted AK/BCC (argmax)        {r_infl_to_ak_argmax:.4f}")
    print(f"    flagged urgent at {FLAG_T}           {r_infl_flagged:.4f}")
    print(f"  AK/BCC (n={int(ak_mask.sum())}, of which {int(ak_np.sum())} non-pigmented)")
    print(f"    predicted inflammatory (argmax)  {r_ak_to_infl_argmax:.4f}   "
          f"(non-pigmented only: {r_ak_np_to_infl_argmax:.4f})")
    print(f"    NOT flagged at {FLAG_T}              {r_ak_missed_flag:.4f}   "
          f"(non-pigmented only: {r_ak_np_missed_flag:.4f})")

    print(f"\n=== EXPECTED ERRORS PER {UPLOADS} UPLOADS ===")
    print(f"  {'infl%':>6} {'AK%':>6} | {'infl->AK/BCC':>13} {'infl flagged':>13} | "
          f"{'AK/BCC cases':>12} {'AK->infl':>9} {'AK missed':>10} | {'ratio':>8}")
    rows = []
    for infl_share in INFLAMMATORY_SHARE:
        for ak_share in AKBCC_SHARE:
            n_infl = UPLOADS * infl_share
            n_ak = UPLOADS * ak_share
            fa_argmax = n_infl * r_infl_to_ak_argmax
            fa_flag = n_infl * r_infl_flagged
            miss_argmax = n_ak * r_ak_to_infl_argmax
            miss_flag = n_ak * r_ak_missed_flag
            ratio = fa_flag / miss_flag if miss_flag > 0 else float("inf")
            rows.append({"inflammatory_share": infl_share, "akbcc_share": ak_share,
                         "n_inflammatory": n_infl, "n_akbcc": n_ak,
                         "infl_to_akbcc_argmax": fa_argmax,
                         "infl_flagged_urgent": fa_flag,
                         "akbcc_to_infl_argmax": miss_argmax,
                         "akbcc_missed_by_flag": miss_flag,
                         "false_alarms_per_miss": ratio})
            print(f"  {infl_share:>6.0%} {ak_share:>6.1%} | {fa_argmax:>13.1f} "
                  f"{fa_flag:>13.1f} | {n_ak:>12.1f} {miss_argmax:>9.2f} "
                  f"{miss_flag:>10.2f} | {ratio:>8.0f}:1")

    mid = next(r for r in rows if r["inflammatory_share"] == 0.65 and r["akbcc_share"] == 0.01)
    print(f"\n=== READING (central case: 65% inflammatory, 1% AK/BCC) ===")
    print(f"  per 1,000 uploads:")
    print(f"    {mid['infl_flagged_urgent']:.0f} inflammatory cases flagged urgent "
          f"— nuisance, erodes trust, costs visits")
    print(f"    {mid['akbcc_missed_by_flag']:.2f} AK/BCC cases missed by the gate "
          f"— rare, dangerous")
    print(f"    ratio {mid['false_alarms_per_miss']:.0f} false alarms per missed cancer")
    print(f"  At 10,000 uploads/day that is ~{mid['infl_flagged_urgent'] * 10:.0f} "
          f"false alarms and ~{mid['akbcc_missed_by_flag'] * 10:.1f} missed AK/BCC daily.")
    print(f"\n  The asymmetry is the point: the dangerous error is ~{mid['false_alarms_per_miss']:.0f}x")
    print(f"  rarer than the costly one, but you cannot trade them at par. Whether "
          f"{mid['false_alarms_per_miss']:.0f}:1")
    print(f"  is acceptable is a clinical and business call, not a modelling one — "
          f"ml/decision_curve.py\n  is where that trade is made explicit.")

    out = {**meta, "rates": {
        "infl_to_akbcc_argmax": r_infl_to_ak_argmax,
        "infl_flagged_urgent": r_infl_flagged,
        "akbcc_to_infl_argmax": r_ak_to_infl_argmax,
        "akbcc_nonpigmented_to_infl_argmax": r_ak_np_to_infl_argmax,
        "akbcc_missed_by_flag": r_ak_missed_flag,
        "akbcc_nonpigmented_missed_by_flag": r_ak_np_missed_flag},
        "scenarios": rows, "central_case": mid,
        "caveat": "Rates are measured on DermNet photography and may shift on real user "
                  "photos; the traffic shares are assumptions, not measurements."}
    pth = ckpt.parent / "traffic_weighted.json"
    pth.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
