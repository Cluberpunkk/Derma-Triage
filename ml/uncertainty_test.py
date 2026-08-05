"""ml/uncertainty_test.py
=========================================================================
Is the urgency score measuring URGENCY, or measuring UNCERTAINTY?

The suspicion: our urgency score is a SUM OF EIGHT MARGINALS. A diffuse prediction
spreads probability across all 23 classes, and eight of them are in the urgent
group, so a confused model accumulates urgent mass without any single urgent class
being likely. If so, the score conflates "I don't know" with "this is serious" — and
half of all inflammatory cases tripping the 0.12 gate would be explained not by them
looking dangerous, but by the model being unsure.

Direct tests, all on existing outputs, no retraining:

  argmax check   of inflammatory cases flagged at 0.12, how many have a ROUTINE
                 top-1 prediction? Those are flagged purely by accumulation.
  entropy/maxp   compare flagged vs non-flagged inflammatory cases.
  the null       a uniform 23-way distribution puts 8/23 = 0.348 into the urgent
                 sum before calibration. If that lands near or above the operating
                 threshold, the gate fires on pure ignorance.

Two cheap remedies, also without retraining:

  (a) require the single largest URGENT-class probability to exceed X, in addition
      to the sum. A genuinely urgent-looking lesion should concentrate mass on one
      urgent class, not smear it across eight.
  (b) an entropy gate: route diffuse predictions to "uncertain — retake the photo or
      answer a few questions" instead of to urgent.

If confirmed, the dedicated binary head is the principled version of the same fix:
it predicts urgency directly and never inherits the sum-over-eight structure.

    ml/.venv/bin/python ml/uncertainty_test.py data/ml/runs_final/best.pt
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
from urgency import (GROUPS_YAML, Platt, Temperature, auroc, load_groups,  # noqa: E402
                     logits_labels, tier_of)

T_FLAG = 0.12
MAXP_SWEEP = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
ENTROPY_SWEEP = [1.01, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60]


def norm_entropy(p: torch.Tensor) -> torch.Tensor:
    h = -(p.clamp(min=1e-12).log() * p).sum(1)
    return h / np.log(p.shape[1])


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
    meta = runmeta.manifest(tool="uncertainty_test.py", checkpoint=ckpt,
                            files={"groups": str(GROUPS_YAML)},
                            extra={"frozen_test": frozen, "flag_threshold": T_FLAG})
    runmeta.print_header(meta, frozen)

    lv, yv = logits_labels(model, DataLoader(val, batch_size=a.batch_size), device)
    lt, yt = logits_labels(model, DataLoader(te, batch_size=a.batch_size), device)
    temp = Temperature().fit(lv, yv)
    soon = torch.tensor([i for i, c in enumerate(classes) if tier_of(groups, c)[0] == "soon"])
    soon_set = set(soon.tolist())
    platt = Platt().fit(temp(lv).softmax(1)[:, soon].sum(1), torch.isin(yv, soon))

    p = temp(lt).softmax(1)
    raw_sum = p[:, soon].sum(1)
    score = platt(raw_sum)
    ent = norm_entropy(p)
    maxp = p.max(1).values
    max_urgent = p[:, soon].max(1).values
    argmax = p.argmax(1)
    y = yt
    is_urgent = torch.isin(y, soon)

    infl = torch.isin(y, torch.tensor([ci[c] for c in INFLAMMATORY if c in ci]))
    pig = torch.isin(y, torch.tensor([ci[c] for c in BENIGN_PIGMENTED if c in ci]))

    # ---------------- the null ----------------
    n_cls = len(classes)
    uni_raw = len(soon) / n_cls
    uni_cal = float(platt(torch.tensor([uni_raw])))
    print(f"\n=== THE NULL ===")
    print(f"  uniform over {n_cls} classes -> urgent sum {uni_raw:.4f} raw, "
          f"{uni_cal:.4f} calibrated")
    print(f"  operating threshold is {T_FLAG} -> a UNIFORM prediction "
          f"{'FIRES' if uni_cal >= T_FLAG else 'does not fire'} the gate")
    hi_ent = ent >= ent.quantile(0.9)
    print(f"  highest-entropy decile (n={int(hi_ent.sum())}): mean urgent score "
          f"{float(score[hi_ent].mean()):.4f}, flag rate "
          f"{float((score[hi_ent] >= T_FLAG).float().mean()):.4f}")
    print(f"  lowest-entropy decile: mean urgent score "
          f"{float(score[ent <= ent.quantile(0.1)].mean()):.4f}")

    # ---------------- argmax check ----------------
    flagged_infl = infl & (score >= T_FLAG)
    routine_argmax = torch.tensor([int(v) not in soon_set for v in argmax])
    acc_only = flagged_infl & routine_argmax
    print(f"\n=== FLAGGED INFLAMMATORY: URGENT-LOOKING, OR JUST UNCERTAIN? ===")
    print(f"  inflammatory cases: {int(infl.sum())}, flagged at {T_FLAG}: "
          f"{int(flagged_infl.sum())} ({float(flagged_infl.float().mean()) * 100:.1f}%)")
    print(f"  of those flagged, top-1 prediction is a ROUTINE class: "
          f"{int(acc_only.sum())}/{int(flagged_infl.sum())} = "
          f"{float(acc_only.sum()) / max(int(flagged_infl.sum()), 1):.1%}")
    print(f"    -> flagged purely by ACCUMULATION across the eight urgent classes")
    nf = infl & (score < T_FLAG)
    print(f"\n  {'':<22}{'flagged':>10}{'not flagged':>13}")
    for name, v in (("norm. entropy", ent), ("max class prob", maxp),
                    ("max URGENT prob", max_urgent), ("urgent sum (cal)", score)):
        print(f"  {name:<22}{float(v[flagged_infl].mean()):>10.4f}"
              f"{float(v[nf].mean()):>13.4f}")

    conf = float(np.corrcoef(ent.numpy(), score.numpy())[0, 1])
    print(f"\n  corr(entropy, urgent score) over the whole test split: {conf:+.4f}")
    verdict_conflated = (float(acc_only.sum()) / max(int(flagged_infl.sum()), 1) > 0.5
                         and conf > 0.2)

    # ---------------- remedy (a): require a concentrated urgent class ----------------
    print(f"\n=== REMEDY (a): also require max single URGENT-class prob >= X ===")
    print(f"  {'X':>6} {'infl flag':>10} {'pig flag':>9} {'urgent recall':>14} "
          f"{'urgentCap10':>12}")
    n = len(score)
    rows_a = []
    for X in MAXP_SWEEP:
        flag = (score >= T_FLAG) & (max_urgent >= X)
        # rank by max_urgent among survivors, else by score (keeps a total order)
        rank_score = torch.where(max_urgent >= X, max_urgent, torch.zeros_like(max_urgent))
        ranked = torch.empty(n, dtype=torch.long)
        ranked[rank_score.argsort(descending=True)] = torch.arange(n)
        cap = float((ranked[is_urgent] < int(round(0.1 * n))).float().mean())
        r = {"X": X, "inflammatory_flag_rate": float(flag[infl].float().mean()),
             "pigmented_flag_rate": float(flag[pig].float().mean()),
             "urgent_recall": float(flag[is_urgent].float().mean()),
             "urgent_top10_capture": cap}
        rows_a.append(r)
        print(f"  {X:>6.2f} {r['inflammatory_flag_rate']:>10.4f} "
              f"{r['pigmented_flag_rate']:>9.4f} {r['urgent_recall']:>14.4f} "
              f"{cap:>12.4f}")
    print(f"  X=0.00 is the current rule. AUROC of max-urgent-prob as a standalone "
          f"score: {auroc(max_urgent, is_urgent):.4f}")
    print(f"  (sum-of-marginals AUROC for comparison: {auroc(score, is_urgent):.4f})")

    # ---------------- remedy (b): entropy gate ----------------
    print(f"\n=== REMEDY (b): route diffuse predictions to 'uncertain' (entropy >= E) ===")
    print(f"  {'E':>6} {'->uncertain':>12} {'infl flag':>10} {'pig flag':>9} "
          f"{'urgent recall':>14} {'urgent->unc':>12}")
    rows_b = []
    for E in ENTROPY_SWEEP:
        diffuse = ent >= E
        flag = (score >= T_FLAG) & ~diffuse
        r = {"E": E, "routed_uncertain": float(diffuse.float().mean()),
             "inflammatory_flag_rate": float(flag[infl].float().mean()),
             "pigmented_flag_rate": float(flag[pig].float().mean()),
             "urgent_recall": float(flag[is_urgent].float().mean()),
             "urgent_routed_uncertain": float(diffuse[is_urgent].float().mean())}
        rows_b.append(r)
        print(f"  {E:>6.2f} {r['routed_uncertain']:>12.4f} "
              f"{r['inflammatory_flag_rate']:>10.4f} {r['pigmented_flag_rate']:>9.4f} "
              f"{r['urgent_recall']:>14.4f} {r['urgent_routed_uncertain']:>12.4f}")
    print("  'urgent->unc' is the cost: urgent cases diverted to 'uncertain'. In a triage")
    print("  product that is a SAFE failure (still routed to a human), unlike 'routine'.")

    print("\n=== VERDICT ===")
    if verdict_conflated:
        v = ("CONFLATED. Most flagged inflammatory cases have a routine top-1 and are "
             "flagged by accumulation alone, and entropy correlates with the urgent score. "
             "The sum-of-marginals score is partly an uncertainty detector. The dedicated "
             "binary head is the principled fix; inflammatory flag rate should be a "
             "headline success metric alongside benign-pigmented rate.")
    else:
        v = ("NOT CONFLATED (or only weakly). Flagged inflammatory cases are not "
             "predominantly routine-argmax accumulation, and/or entropy does not track the "
             "urgent score strongly. The false alarms are substantive rather than an "
             "artefact of the sum-over-eight structure.")
    print("  " + v.replace(". ", ".\n  "))

    out = {**meta, "null": {"uniform_raw_sum": uni_raw, "uniform_calibrated": uni_cal,
                            "fires_gate": bool(uni_cal >= T_FLAG)},
           "flagged_inflammatory": {
               "n_inflammatory": int(infl.sum()), "n_flagged": int(flagged_infl.sum()),
               "frac_routine_argmax": float(acc_only.sum()) / max(int(flagged_infl.sum()), 1),
               "mean_entropy_flagged": float(ent[flagged_infl].mean()),
               "mean_entropy_not_flagged": float(ent[nf].mean()),
               "mean_maxp_flagged": float(maxp[flagged_infl].mean()),
               "mean_maxp_not_flagged": float(maxp[nf].mean())},
           "corr_entropy_urgentscore": conf,
           "remedy_a_max_urgent_prob": rows_a,
           "remedy_b_entropy_gate": rows_b,
           "auroc_max_urgent_prob": auroc(max_urgent, is_urgent),
           "auroc_sum_marginals": auroc(score, is_urgent),
           "conflated": bool(verdict_conflated), "verdict": v}
    pth = ckpt.parent / "uncertainty_test.json"
    pth.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
