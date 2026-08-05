"""ml/abstain.py
=========================================================================
SELECTIVE PREDICTION — teaches the model to say "I don't know".

Competitors answer every query with one confident-looking label. Our evaluation
showed this model is confidently wrong (p>=0.70 and incorrect) on ~10% of test
images. For triage that is the worst failure mode: a firmly stated wrong answer is
more dangerous than an honest "unsure".

This script does not retrain anything. It finds the CONFIDENCE THRESHOLD at which
the model should abstain and route to a clinician, and reports the trade-off
honestly:

  coverage        fraction of cases the model is willing to answer
  selective risk  error rate ON THE ANSWERED CASES only
  abstain rate    fraction routed to "see a clinician" instead

The trade is real and must be shown, not hidden: higher confidence thresholds mean
fewer wrong answers but more people told "we can't say". A product decision, so the
numbers are printed rather than a single threshold being chosen silently.

Two extra views that matter more than overall accuracy:

  can't-miss recall vs threshold   abstention must not silently reduce recall on
                                   the malignancy folders — a missed cancer routed
                                   to "unsure" is fine; missed and called benign
                                   is not
  entropy vs max-prob              two ways to score uncertainty; entropy uses the
                                   whole distribution and is usually better
                                   calibrated for multi-class confusion

    ml/.venv/bin/python ml/abstain.py data/ml/runs_final/best.pt
    ml/.venv/bin/python ml/abstain.py best.pt --target-risk 0.25
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
from dataset import ManifestImageDataset  # noqa: E402
from evaluate import CANT_MISS_SUBSTRINGS  # noqa: E402
from train import load_checkpoint  # noqa: E402

THRESHOLDS = [0.0, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


@torch.no_grad()
def gather(model, loader, device):
    """Return (probs, targets) for the whole split — one forward pass, reused below."""
    P, Y = [], []
    for x, y in loader:
        P.append(model(x.to(device)).softmax(dim=1).cpu())
        Y.append(y)
    return torch.cat(P), torch.cat(Y)


def entropy_conf(probs: torch.Tensor) -> torch.Tensor:
    """1 - normalised entropy: 1.0 = certain, 0.0 = uniform over all classes."""
    n = probs.shape[1]
    ent = -(probs.clamp(min=1e-12).log() * probs).sum(1)
    return 1.0 - ent / torch.log(torch.tensor(float(n)))


def curve(conf: torch.Tensor, correct: torch.Tensor, thresholds) -> list[dict]:
    rows = []
    total = len(conf)
    for t in thresholds:
        keep = conf >= t
        n_kept = int(keep.sum())
        rows.append({
            "threshold": t,
            "coverage": n_kept / total,
            "abstain_rate": 1 - n_kept / total,
            "selective_accuracy": float(correct[keep].float().mean()) if n_kept else float("nan"),
            "selective_risk": 1 - float(correct[keep].float().mean()) if n_kept else float("nan"),
            "n_answered": n_kept,
            "n_wrong_answered": int((~correct[keep]).sum()) if n_kept else 0,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--manifest"), ap.add_argument("--root")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device")
    ap.add_argument("--target-risk", type=float, default=0.30,
                    help="max acceptable error rate among ANSWERED cases")
    a = ap.parse_args()

    ckpt_path = Path(a.checkpoint).resolve()
    device = config.resolve_device(a.device)
    model, ck = load_checkpoint(ckpt_path, device)
    classes = ck["classes"]

    manifest_dir = Path(a.manifest) if a.manifest else config.out_dir() / "dermnet"
    root = config.dataset_root(a.root)
    ds = ManifestImageDataset(manifest_dir, root, a.split,
                              ck.get("label_column", "original_class"),
                              img_size=ck["img_size"])
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=0)

    print(f"checkpoint {ckpt_path}\n  {ck['model_name']} · split={a.split} · n={len(ds)} · {device}\n")
    probs, y = gather(model, loader, device)
    pred = probs.argmax(1)
    correct = pred == y
    maxp = probs.max(1).values
    ent = entropy_conf(probs)

    print(f"baseline (answer everything): accuracy {float(correct.float().mean()):.3f}\n")

    print("max-probability thresholding")
    print("  thresh  coverage  abstain  sel.acc  sel.risk  answered  wrong")
    rows_max = curve(maxp, correct, THRESHOLDS)
    for r in rows_max:
        print(f"   {r['threshold']:.2f}    {r['coverage']:.3f}    {r['abstain_rate']:.3f}"
              f"    {r['selective_accuracy']:.3f}    {r['selective_risk']:.3f}"
              f"    {r['n_answered']:<6}  {r['n_wrong_answered']}")

    print("\nentropy-based thresholding (uses the whole distribution)")
    print("  thresh  coverage  abstain  sel.acc  sel.risk  answered  wrong")
    rows_ent = curve(ent, correct, THRESHOLDS)
    for r in rows_ent:
        print(f"   {r['threshold']:.2f}    {r['coverage']:.3f}    {r['abstain_rate']:.3f}"
              f"    {r['selective_accuracy']:.3f}    {r['selective_risk']:.3f}"
              f"    {r['n_answered']:<6}  {r['n_wrong_answered']}")

    # --- recommended operating point for the target risk ---
    ok = [r for r in rows_max if r["n_answered"] and r["selective_risk"] <= a.target_risk]
    rec = max(ok, key=lambda r: r["coverage"]) if ok else None
    print(f"\ntarget selective risk <= {a.target_risk:.2f}")
    if rec:
        print(f"  -> threshold {rec['threshold']:.2f}: answers {rec['coverage']:.1%} of cases "
              f"at {rec['selective_risk']:.3f} risk; {rec['abstain_rate']:.1%} routed to a clinician")
    else:
        print("  -> UNREACHABLE at any tested threshold. The model cannot hit that error rate "
              "on any confident subset; it should not present a single answer at all.")

    # --- can't-miss safety check: does abstaining hide cancers? ---
    cm_idx = [i for i, c in enumerate(classes)
              if any(s in c.lower() for s in CANT_MISS_SUBSTRINGS)]
    cm_report = []
    if cm_idx:
        is_cm = torch.zeros_like(y, dtype=torch.bool)
        for i in cm_idx:
            is_cm |= (y == i)
        n_cm = int(is_cm.sum())
        print(f"\ncan't-miss folders (n={n_cm}) — abstention must not turn misses into "
              f"confident 'benign' calls")
        print("  thresh  answered  recall_on_answered  routed_to_clinician  missed_confidently")
        for t in THRESHOLDS:
            keep = (maxp >= t) & is_cm
            n_ans = int(keep.sum())
            rec_ans = float(correct[keep].float().mean()) if n_ans else float("nan")
            # answered, wrong, AND predicted something not in the can't-miss set
            wrong_conf = int(((~correct) & keep &
                              ~torch.isin(pred, torch.tensor(cm_idx))).sum())
            row = {"threshold": t, "n_answered": n_ans,
                   "recall_on_answered": rec_ans,
                   "routed": n_cm - n_ans, "missed_confidently": wrong_conf}
            cm_report.append(row)
            print(f"   {t:.2f}    {n_ans:<8}  {rec_ans:.3f}               "
                  f"{n_cm - n_ans:<19}  {wrong_conf}")
        print("  'missed_confidently' = a malignancy answered as something benign. This is the "
              "number to drive to zero;\n  'routed_to_clinician' is the safe failure mode.")

    out = ckpt_path.parent / "abstain_report.json"
    out.write_text(json.dumps({
        "checkpoint": str(ckpt_path), "split": a.split, "n": len(ds),
        "baseline_accuracy": float(correct.float().mean()),
        "max_prob": rows_max, "entropy": rows_ent,
        "target_risk": a.target_risk, "recommended": rec,
        "cant_miss": cm_report,
        "note": "Abstention converts confident errors into referrals. It does not make the "
                "model diagnostic; it makes its uncertainty visible.",
    }, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
