"""ml/decision_curve.py
=========================================================================
Decision-curve / net-benefit analysis — the formal version of "this cannot be used
to filter users", in a form a clinical or business stakeholder can check.

Accuracy, AUROC and AUPRC all ignore the two things that decide whether a triage
model is worth deploying:

  base rate   how common urgent disease actually is among the people using the app
  cost ratio  how much worse a missed urgent case is than a false alarm

Net benefit folds both in. Per Vickers' decision-curve analysis, with cost ratio
C = cost(miss) / cost(false alarm):

    NB(threshold) = TPR * prevalence  -  FPR * (1 - prevalence) / C

expressed in "true positives per patient", so it is directly comparable against the
two strategies any product could adopt without a model at all:

    flag everyone : NB = prevalence - (1 - prevalence) / C
    flag no one   : NB = 0

A model is only worth deploying where its curve sits ABOVE BOTH defaults. If the
best a model can do is match "flag everyone", it is adding cost and no decision.

Note that TPR and FPR are properties of the model and do NOT change with assumed
prevalence — only the weighting does. That is why recall stays fixed across the
prevalence sweep while precision collapses.

    ml/.venv/bin/python ml/decision_curve.py data/ml/runs_final/best.pt
=========================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import runmeta  # noqa: E402
from dataset import ManifestImageDataset  # noqa: E402
from train import load_checkpoint  # noqa: E402
from urgency import (GROUPS_YAML, Platt, Temperature, load_groups, logits_labels,  # noqa: E402
                     tier_of)

PREVALENCES = [0.05, 0.10, 0.15, 0.20, 0.30]
COST_RATIOS = [10, 50, 200]           # miss : false alarm
THRESHOLDS = [0.05, 0.12, 0.20, 0.32, 0.50, 0.72, 0.90]


def roc_points(score, is_pos, ths):
    n_pos = int(is_pos.sum())
    n_neg = len(score) - n_pos
    out = []
    for t in ths:
        flag = score >= t
        tpr = int((flag & is_pos).sum()) / max(n_pos, 1)
        fpr = int((flag & ~is_pos).sum()) / max(n_neg, 1)
        out.append({"threshold": t, "tpr": tpr, "fpr": fpr})
    return out


def net_benefit(tpr, fpr, prev, C):
    return tpr * prev - fpr * (1 - prev) / C


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
    lc = ck.get("label_column", "original_class")
    manifest_dir = Path(a.manifest) if a.manifest else config.out_dir() / "dermnet"
    root = config.dataset_root(a.root)
    groups = load_groups(GROUPS_YAML)

    val = ManifestImageDataset(manifest_dir, root, "val", lc, img_size=ck["img_size"])
    te = ManifestImageDataset(manifest_dir, root, "test", lc, img_size=ck["img_size"])
    frozen = runmeta.freeze(manifest_dir, "test", te.rows)
    meta = runmeta.manifest(tool="decision_curve.py", checkpoint=ckpt,
                            files={"groups": str(GROUPS_YAML)},
                            extra={"frozen_test": frozen, "cost_ratios": COST_RATIOS,
                                   "prevalences": PREVALENCES})
    runmeta.print_header(meta, frozen)

    lv, yv = logits_labels(model, DataLoader(val, batch_size=a.batch_size), device)
    lt, yt = logits_labels(model, DataLoader(te, batch_size=a.batch_size), device)
    temp = Temperature().fit(lv, yv)
    soon = torch.tensor([i for i, c in enumerate(classes) if tier_of(groups, c)[0] == "soon"])
    platt = Platt().fit(temp(lv).softmax(1)[:, soon].sum(1), torch.isin(yv, soon))
    score = platt(temp(lt).softmax(1)[:, soon].sum(1))
    is_pos = torch.isin(yt, soon)

    pts = roc_points(score, is_pos, THRESHOLDS)
    print(f"\nmodel operating points (TPR/FPR are prevalence-independent)")
    for p in pts:
        print(f"  t={p['threshold']:.2f}  TPR {p['tpr']:.3f}  FPR {p['fpr']:.3f}")

    results = []
    for C in COST_RATIOS:
        print(f"\n=== COST RATIO {C}:1  (one missed urgent case = {C} false alarms) ===")
        print("  prev   flag-all   flag-none      best model NB @ t     verdict")
        for prev in PREVALENCES:
            nb_all = net_benefit(1.0, 1.0, prev, C)
            nb_none = 0.0
            best = max(pts, key=lambda p: net_benefit(p["tpr"], p["fpr"], prev, C))
            nb_best = net_benefit(best["tpr"], best["fpr"], prev, C)
            beats_all = nb_best > nb_all
            beats_none = nb_best > nb_none
            verdict = ("USEFUL" if beats_all and beats_none else
                       "no better than flag-everyone" if not beats_all else
                       "no better than doing nothing")
            print(f"  {prev:>4.0%}  {nb_all:>9.4f}  {nb_none:>9.4f}   "
                  f"{nb_best:>9.4f} @ {best['threshold']:.2f}     {verdict}")
            results.append({"cost_ratio": C, "prevalence": prev,
                            "nb_flag_all": nb_all, "nb_flag_none": nb_none,
                            "nb_best_model": nb_best, "best_threshold": best["threshold"],
                            "beats_flag_all": beats_all, "beats_flag_none": beats_none,
                            "verdict": verdict,
                            "curve": [{"threshold": p["threshold"],
                                       "net_benefit": net_benefit(p["tpr"], p["fpr"], prev, C)}
                                      for p in pts]})

    useful = [r for r in results if r["beats_flag_all"] and r["beats_flag_none"]]
    print(f"\n=== SUMMARY ===")
    print(f"  scenarios where the model beats BOTH defaults: {len(useful)}/{len(results)}")
    if useful:
        lo = min(r["prevalence"] for r in useful)
        print(f"  useful from prevalence >= {lo:.0%} in at least one cost regime")
        for r in useful:
            print(f"    C={r['cost_ratio']}:1  prev={r['prevalence']:.0%}  "
                  f"t={r['best_threshold']:.2f}  NB={r['nb_best_model']:.4f} "
                  f"(vs flag-all {r['nb_flag_all']:.4f})")
    else:
        print("  NONE. At every prevalence and cost ratio tested, a no-model default is at "
              "least as good.")
    print("\n  Reading this: at HIGH cost ratios (a miss is catastrophic) 'flag everyone'")
    print("  becomes hard to beat, because the model's false-alarm savings stop mattering.")
    print("  At LOW prevalence the model's true positives are too few to offset its false")
    print("  alarms. The model earns its place only in the middle — and that band is where")
    print("  the pigmentation blocker also lives.")

    out = {**meta, "operating_points": pts, "scenarios": results,
           "n_useful": len(useful), "n_scenarios": len(results),
           "note": "Net benefit in true-positives-per-patient. A model is worth deploying "
                   "only where its curve exceeds BOTH flag-everyone and flag-no-one."}
    p = ckpt.parent / "decision_curve.json"
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
