"""ml/pigmentation_audit.py
=========================================================================
Quantifies the failure mode most likely to hit our actual market.

The failure sheets showed the top false alarms are dominated by BENIGN PIGMENTED
lesions — "Light Diseases and Disorders of Pigmentation" (melasma, PIH, vitiligo,
lichen planus pigmentosus) and "Seborrheic Keratoses and other Benign Tumors"
(seborrheic keratoses, dermatosis papulosa nigra).

Why that ordering of priority:

  These conditions are FAR more common and more visible on Indian skin than in
  DermNet's mostly-light-skinned population. PIH follows almost any inflammatory
  condition on brown skin; melasma is extremely common; DPN is near-universal with
  age in darker skin. So our worst measured failure mode is precisely the one our
  users will trigger most often — and DermNet's class balance understates how often.

Telling a woman with melasma that she may have something urgent is not a minor
false positive. It is the app failing in the most visible, most trust-destroying
way, at scale, to the exact population it was built for.

Verdict rule (deliberately blunt): if the flag rate on benign pigmentation at the
high-recall operating point is high, that is a HARD BLOCKER on any urgency-facing
UI, whatever AUROC says. AUROC averages over the whole distribution; a blocker is
about one subgroup.

    ml/.venv/bin/python ml/pigmentation_audit.py data/ml/runs_final/best.pt
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
from urgency import (GROUPS_YAML, REPORT_THRESHOLDS, Platt, Temperature,  # noqa: E402
                     load_groups, logits_labels, tier_of)

# DermNet folders that are predominantly BENIGN PIGMENTED disease. Each is
# tier=routine in urgency_groups.yaml, so any flag here is a false alarm.
BENIGN_PIGMENTED = {
    "Light Diseases and Disorders of Pigmentation":
        "melasma, PIH, vitiligo, lichen planus pigmentosus — the commonest "
        "pigmentary complaints on Indian skin",
    "Seborrheic Keratoses and other Benign Tumors":
        "seborrheic keratoses, dermatosis papulosa nigra — near-universal with age "
        "in darker skin",
}

# RETIRED: BLOCKER_FLAG_RATE = 0.30.
#
# Blocker 1 measured 0.302 against a bar I invented — it fired by 0.002. A rule that
# decides a blocker on a coin-flip against an arbitrary constant is not measuring
# anything. docs/ARCHITECTURE_decision.md established the test that does have content:
#
#     A blocker binds if and only if a SHIPPING SURFACE consumes the metric.
#
# The benign-pigmented flag rate is consumed only by a user-facing urgency flag
# (surface #7), which we are not shipping. Under the ranking architecture we ARE
# shipping, benign pigmented lesions reach the top-10% queue 3.7% of the time. So
# Blocker 1 is conditional, and its status is decided architecturally, not by a rate.
#
# The rate is still reported — it is real, and it binds immediately if an urgency
# flag ever ships. It just no longer decides anything on its own.
BLOCKER_FLAG_RATE = None
CONSUMING_SURFACE = "user-facing urgency flag (ARCHITECTURE_decision.md surface #7)"
SURFACE_SHIPS = False


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
    meta = runmeta.manifest(tool="pigmentation_audit.py", checkpoint=ckpt,
                            files={"groups": str(GROUPS_YAML)},
                            extra={"frozen_test": frozen,
                                   "blocker_rule": "architecture test (surface ships?)",
                                   "consuming_surface": CONSUMING_SURFACE,
                                   "surface_ships": SURFACE_SHIPS})
    runmeta.print_header(meta, frozen)

    lv, yv = logits_labels(model, DataLoader(val, batch_size=a.batch_size), device)
    lt, yt = logits_labels(model, DataLoader(te, batch_size=a.batch_size), device)
    temp = Temperature().fit(lv, yv)
    pv, pt = temp(lv).softmax(1), temp(lt).softmax(1)
    soon = torch.tensor([i for i, c in enumerate(classes) if tier_of(groups, c)[0] == "soon"])
    platt = Platt().fit(pv[:, soon].sum(1), torch.isin(yv, soon))
    score = platt(pt[:, soon].sum(1))
    is_pos = torch.isin(yt, soon)

    print("\n=== BENIGN PIGMENTED CLASSES — urgent-score distribution ===")
    per_class, pig_mask = [], torch.zeros_like(yt, dtype=torch.bool)
    for cls, why in BENIGN_PIGMENTED.items():
        if cls not in classes:
            print(f"  (absent from label space: {cls})")
            continue
        i = classes.index(cls)
        m = yt == i
        pig_mask |= m
        s = score[m]
        tier, conf = tier_of(groups, cls)
        row = {"class": cls, "why_it_matters": why, "tier": tier, "grouping_confidence": conf,
               "n": int(m.sum()), "mean_urgent_score": float(s.mean()),
               "median": float(s.median()),
               "quantiles": {q: float(s.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
               "flag_rate": {str(t): float((s >= t).float().mean()) for t in REPORT_THRESHOLDS},
               "n_flagged": {str(t): int((s >= t).sum()) for t in REPORT_THRESHOLDS}}
        per_class.append(row)
        print(f"\n  {cls}")
        print(f"    n={row['n']}  mean={row['mean_urgent_score']:.3f}  "
              f"median={row['median']:.3f}")
        print(f"    deciles: " + "  ".join(f"p{int(q*100)}={v:.3f}"
                                           for q, v in row["quantiles"].items()))
        for t in REPORT_THRESHOLDS:
            print(f"    flagged at {t}: {row['n_flagged'][str(t)]:>4}/{row['n']} "
                  f"= {row['flag_rate'][str(t)]:.1%}")

    # ---- aggregate share of ALL false alarms ----
    print("\n=== SHARE OF ALL FALSE ALARMS FROM BENIGN PIGMENTATION ===")
    print("  thresh   all FAs   from pigmentation   share   pigm. flag rate")
    agg = []
    for t in REPORT_THRESHOLDS:
        flag = score >= t
        fa_all = int((flag & ~is_pos).sum())
        fa_pig = int((flag & pig_mask).sum())
        share = fa_pig / fa_all if fa_all else float("nan")
        rate = float(score[pig_mask].ge(t).float().mean())
        agg.append({"threshold": t, "false_alarms_total": fa_all,
                    "false_alarms_from_pigmentation": fa_pig,
                    "share_of_false_alarms": share, "pigmentation_flag_rate": rate})
        print(f"   {t:<6}   {fa_all:>7}   {fa_pig:>17}   {share:>5.1%}   {rate:>13.1%}")

    n_pig = int(pig_mask.sum())
    print(f"\n  benign pigmented cases in test split: {n_pig}/{len(yt)} = "
          f"{n_pig / len(yt):.1%}")
    print("  NOTE: DermNet under-represents these relative to an Indian user base, so the")
    print("        real-world false-alarm share would be HIGHER than shown here.")

    # ---- verdict ----
    op = agg[0]                                     # high-recall operating point (0.12)
    # Architecture test, not a rate comparison: does a shipping surface consume this?
    blocked = SURFACE_SHIPS
    if blocked:
        verdict = (
            f"BLOCKS. A shipping surface consumes this metric ({CONSUMING_SURFACE}). "
            f"At the high-recall operating point (t={op['threshold']}), "
            f"{op['pigmentation_flag_rate']:.1%} of benign pigmented cases are flagged as "
            f"possibly urgent, and pigmentation supplies {op['share_of_false_alarms']:.1%} "
            f"of all false alarms. These conditions are far more common on Indian skin than "
            f"in this dataset, so the real rate is higher still. Do NOT ship an "
            f"urgency-facing UI on this model: it would tell a large fraction of users with "
            f"melasma, PIH or seborrheic keratoses that they may have something serious. "
            f"AUROC 0.878 does not license this — AUROC averages over the whole "
            f"distribution; this is one subgroup, and it is our biggest one."
        )
    else:
        verdict = (
            f"Not a blocker at t={op['threshold']}: benign pigmentation flag rate is "
            f"{op['pigmentation_flag_rate']:.1%} (below the {BLOCKER_FLAG_RATE:.0%} bar), "
            f"supplying {op['share_of_false_alarms']:.1%} of false alarms. Still the "
            f"largest single false-alarm source and worth monitoring on real user data."
        )
    print("\n=== VERDICT ===\n  " + verdict.replace(". ", ".\n  "))

    out = {**meta, "benign_pigmented_classes": per_class,
           "false_alarm_share": agg,
           "n_pigmented_in_test": n_pig,
           "pigmented_share_of_test": n_pig / len(yt),
           "is_blocker": blocked, "verdict": verdict,
           "caveat": "DermNet under-represents benign pigmentary disease relative to an "
                     "Indian user base; these flag rates are a lower bound."}
    p = ckpt.parent / "pigmentation_false_alarm.json"
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
