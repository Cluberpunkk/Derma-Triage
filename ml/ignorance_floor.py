"""ml/ignorance_floor.py
=========================================================================
THE IGNORANCE FLOOR, and what it invalidates.

A uniform prediction over 23 classes puts 8/23 = 0.348 of its mass into the urgent
group; calibrated, that is ~0.309. So a model that knows NOTHING scores 0.309.

    No threshold below the floor can distinguish urgency from ignorance.

Every operational number we produced at t=0.12 was therefore measured in a regime
where a blank prediction fires the gate. This re-runs the operational analysis
restricted to thresholds ABOVE the floor and prints each result beside its old
below-floor value, so the size of the artefact is visible rather than assumed.

SECOND PART — ignorance-corrected scores, with a caveat that has to be stated first:

    Any MONOTONE transform of the sum — including (s - floor)/(1 - floor) and the
    log-odds-relative-to-uniform version — leaves AUROC, AUPRC and top-k capture
    EXACTLY unchanged. Those metrics depend only on the ranking, and a monotone map
    does not reorder anything.

So correcting for the floor cannot improve discrimination. What it changes is the
MEANING of a threshold: after correction, 0 means "no better than ignorance" and the
number is interpretable. That is worth having, but it is not a performance gain, and
reporting identical AUROCs as though the correction achieved something would be
misleading.

The only comparisons that can actually move discrimination are against scores that
are NOT monotone transforms of the sum: max-urgent-class probability, and
combinations. Those are compared at MATCHED RECALL, which is the only fair way to
compare flag rates across differently-scaled scores.

    ml/.venv/bin/python ml/ignorance_floor.py data/ml/runs_final/best.pt
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
from pigmentation_audit import BENIGN_PIGMENTED  # noqa: E402
from resolution_ablation import INFLAMMATORY  # noqa: E402
from train import load_checkpoint  # noqa: E402
from urgency import (GROUPS_YAML, ASSUMED_PREVALENCE, Platt, Temperature, auprc,  # noqa: E402
                     auroc, load_groups, logits_labels, precision_at_prevalence,
                     tier_of)

OLD_THRESHOLDS = [0.12, 0.32]
MATCHED_RECALLS = [0.95, 0.90, 0.80]


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def threshold_for_recall(score: np.ndarray, is_pos: np.ndarray, target: float) -> float:
    """Lowest threshold achieving at least `target` recall."""
    s = np.sort(score[is_pos])[::-1]
    k = int(np.ceil(target * len(s))) - 1
    return float(s[min(max(k, 0), len(s) - 1)])


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

    lv, yv = logits_labels(model, DataLoader(val, batch_size=a.batch_size), device)
    lt, yt = logits_labels(model, DataLoader(te, batch_size=a.batch_size), device)
    temp = Temperature().fit(lv, yv)
    soon = torch.tensor([i for i, c in enumerate(classes) if tier_of(groups, c)[0] == "soon"])
    platt = Platt().fit(temp(lv).softmax(1)[:, soon].sum(1), torch.isin(yv, soon))
    p = temp(lt).softmax(1)

    FLOOR = float(platt(torch.tensor([len(soon) / len(classes)])))
    meta = runmeta.manifest(tool="ignorance_floor.py", checkpoint=ckpt,
                            files={"groups": str(GROUPS_YAML)},
                            extra={"frozen_test": frozen, "IGNORANCE_FLOOR": FLOOR,
                                   "n_urgent_classes": len(soon), "n_classes": len(classes)})
    runmeta.print_header(meta, frozen)
    print(f"\n*** IGNORANCE FLOOR = {FLOOR:.4f} "
          f"({len(soon)}/{len(classes)} classes urgent, calibrated) ***")
    print("*** No threshold below this separates urgency from ignorance. ***")

    s_sum = platt(p[:, soon].sum(1)).numpy()
    s_max = p[:, soon].max(1).values.numpy()
    y = yt.numpy()
    is_pos = np.isin(y, soon.numpy())
    infl = np.isin(y, [ci[c] for c in INFLAMMATORY if c in ci])
    pig = np.isin(y, [ci[c] for c in BENIGN_PIGMENTED if c in ci])
    n = len(y)
    n_pos, n_neg = int(is_pos.sum()), int((~is_pos).sum())

    # ---------------- item 1: re-baseline above the floor ----------------
    above = [round(FLOOR + 0.01, 3), 0.40, 0.50, 0.60, 0.72, 0.85]
    print(f"\n=== OPERATIONAL TABLE: below-floor (old) vs above-floor ===")
    print(f"  {'thresh':>7} {'':>6} {'recall':>7} {'prec':>6} {'flagged':>8} "
          f"{'infl flag':>10} {'pig flag':>9} {'missed':>7}")
    rows = []
    for t, tag in [(x, "OLD*") for x in OLD_THRESHOLDS] + [(x, "") for x in above]:
        flag = s_sum >= t
        tp = int((flag & is_pos).sum())
        fp = int((flag & ~is_pos).sum())
        r = {"threshold": t, "below_floor": t < FLOOR,
             "recall": tp / max(n_pos, 1), "precision": tp / max(tp + fp, 1),
             "flagged_rate": float(flag.mean()),
             "inflammatory_flag_rate": float(flag[infl].mean()),
             "pigmented_flag_rate": float(flag[pig].mean()),
             "missed": n_pos - tp,
             "fpr": fp / max(n_neg, 1)}
        r["precision_at_prev"] = {f"{q:.0%}": precision_at_prevalence(r["recall"], r["fpr"], q)
                                  for q in ASSUMED_PREVALENCE}
        rows.append(r)
        mark = tag if tag else ("" if t >= FLOOR else "OLD*")
        print(f"  {t:>7.3f} {mark:>6} {r['recall']:>7.3f} {r['precision']:>6.3f} "
              f"{r['flagged_rate']:>8.3f} {r['inflammatory_flag_rate']:>10.3f} "
              f"{r['pigmented_flag_rate']:>9.3f} {r['missed']:>7}")
    print(f"  OLD* = below the {FLOOR:.3f} floor; a uniform prediction also clears these.")

    old12 = rows[0]
    first_above = next(r for r in rows if not r["below_floor"])
    print(f"\n  BLOCKER 1 re-baselined: benign-pigmented flag rate "
          f"{old12['pigmented_flag_rate']:.3f} (t=0.12, below floor) -> "
          f"{first_above['pigmented_flag_rate']:.3f} (t={first_above['threshold']:.3f})")
    print(f"  inflammatory flag rate  {old12['inflammatory_flag_rate']:.3f} -> "
          f"{first_above['inflammatory_flag_rate']:.3f}")
    print(f"  urgent recall           {old12['recall']:.3f} -> {first_above['recall']:.3f}")

    # 588:1 recomputed
    print(f"\n=== FALSE-ALARM : MISSED-CANCER RATIO (65% infl, 1% AK/BCC per 1000) ===")
    ak = y == ci["Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions"]
    print(f"  {'thresh':>7} {'':>6} {'infl flagged':>13} {'AK/BCC missed':>14} {'ratio':>9}")
    ratios = []
    for r in rows:
        t = r["threshold"]
        fa = 650 * r["inflammatory_flag_rate"]
        miss = 10 * float((s_sum[ak] < t).mean())
        ratio = fa / miss if miss > 0 else float("inf")
        ratios.append({"threshold": t, "below_floor": r["below_floor"],
                       "infl_flagged_per_1000": fa, "akbcc_missed_per_1000": miss,
                       "false_alarms_per_miss": ratio})
        print(f"  {t:>7.3f} {'OLD*' if r['below_floor'] else '':>6} {fa:>13.1f} "
              f"{miss:>14.2f} {ratio:>9.0f}:1")

    # ---------------- item 2: corrected scores ----------------
    print(f"\n=== IGNORANCE-CORRECTED SCORES ===")
    s_lin = np.clip((s_sum - FLOOR) / (1 - FLOOR), 0, 1)
    s_lo = 1 / (1 + np.exp(-(logit(s_sum) - logit(FLOOR))))
    s_comb = np.sqrt(np.clip(s_lin, 0, 1) * s_max)     # geometric mean, not monotone in either

    cand = {"sum (raw)": s_sum, "linear-corrected": s_lin,
            "log-odds vs uniform": s_lo, "max-urgent-prob": s_max,
            "sqrt(lin x maxurg)": s_comb}
    print(f"  {'score':<22} {'AUROC':>8} {'AUPRC':>8}   (no-skill AUPRC = "
          f"{is_pos.mean():.4f})")
    ranking = {}
    for name, s in cand.items():
        ts, ti = torch.tensor(s), torch.tensor(is_pos)
        ranking[name] = {"auroc": auroc(ts, ti), "auprc": auprc(ts, ti)}
        print(f"  {name:<22} {ranking[name]['auroc']:>8.4f} {ranking[name]['auprc']:>8.4f}")
    print(f"\n  NOTE: 'linear-corrected' and 'log-odds' are MONOTONE transforms of the sum,")
    print(f"  so their AUROC/AUPRC are identical to it BY CONSTRUCTION. Correcting for the")
    print(f"  floor makes thresholds interpretable; it cannot improve discrimination.")

    print(f"\n=== MATCHED-RECALL COMPARISON (the only fair way to compare flag rates) ===")
    matched = {}
    for target in MATCHED_RECALLS:
        print(f"\n  at urgent recall = {target:.0%}")
        print(f"    {'score':<22} {'thresh':>8} {'infl flag':>10} {'pig flag':>9} "
              f"{'flagged':>8} {'top10 cap':>10}")
        for name, s in cand.items():
            t = threshold_for_recall(s, is_pos, target)
            flag = s >= t
            ranked = np.empty(n, dtype=int)
            ranked[np.argsort(-s)] = np.arange(n)
            cap = float((ranked[is_pos] < int(round(0.1 * n))).mean())
            rec = float(flag[is_pos].mean())
            matched.setdefault(f"{target}", {})[name] = {
                "threshold": t, "actual_recall": rec,
                "inflammatory_flag_rate": float(flag[infl].mean()),
                "pigmented_flag_rate": float(flag[pig].mean()),
                "flagged_rate": float(flag.mean()), "top10_capture": cap}
            m = matched[f"{target}"][name]
            print(f"    {name:<22} {t:>8.4f} {m['inflammatory_flag_rate']:>10.4f} "
                  f"{m['pigmented_flag_rate']:>9.4f} {m['flagged_rate']:>8.4f} "
                  f"{cap:>10.4f}")

    best90 = min(matched["0.9"].items(), key=lambda kv: kv[1]["pigmented_flag_rate"])
    print(f"\n  lowest benign-pigmented flag rate at 90% recall: '{best90[0]}' "
          f"({best90[1]['pigmented_flag_rate']:.4f} vs sum "
          f"{matched['0.9']['sum (raw)']['pigmented_flag_rate']:.4f})")

    out = {**meta, "IGNORANCE_FLOOR": FLOOR,
           "operational_table": rows, "false_alarm_ratio": ratios,
           "score_ranking_metrics": ranking, "matched_recall": matched,
           "monotone_note": "linear-corrected and log-odds are monotone transforms of the "
                            "sum; identical AUROC/AUPRC/capture by construction.",
           "caveat": f"Every number previously reported at t=0.12 was measured below the "
                     f"{FLOOR:.4f} ignorance floor."}
    pth = ckpt.parent / "ignorance_floor.json"
    pth.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
