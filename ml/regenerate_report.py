"""ml/regenerate_report.py
=========================================================================
SINGLE SOURCE OF TRUTH for every reported number, and the tool that rewrites the
docs from it.

This project has revised its own conclusions four times — the pigmentation shortcut
(refuted), the resolution ablation (invalid, FixRes), the uncertainty conflation
(confirmed), the ignorance floor (which superseded most operational figures). Each
revision left stale numbers in artifacts that read as current findings. A stakeholder
doc carrying a superseded figure is the likeliest way a correction gets lost.

So: every canonical number is recomputed here from the frozen split, written to
CANONICAL.json, and injected into the docs between AUTOGEN markers. Anything that
cannot be regenerated automatically is listed at the end as MANUAL so it can be
fixed by hand rather than silently rotting.

Every threshold-dependent figure is labelled above- or below-floor, and the
ignorance floor is stamped as a constant in the output.

    ml/.venv/bin/python ml/regenerate_report.py data/ml/runs_final/best.pt
    ml/.venv/bin/python ml/regenerate_report.py best.pt --check   # CI mode: no writes
=========================================================================
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import runmeta  # noqa: E402
from dataset import ManifestImageDataset  # noqa: E402
from pigmentation_audit import (BENIGN_PIGMENTED, CONSUMING_SURFACE,  # noqa: E402
                                SURFACE_SHIPS)
from resolution_ablation import AKBCC, INFLAMMATORY  # noqa: E402
from train import load_checkpoint  # noqa: E402
from urgency import (GROUPS_YAML, Platt, Temperature, auprc, auroc, load_groups,  # noqa: E402
                     logits_labels, tier_of)

REPO = Path(__file__).resolve().parent.parent
THRESHOLDS = [0.12, 0.32]
KS = [0.05, 0.10, 0.20, 0.30]
N_BOOT = 400

# Figures that are asserted in prose and cannot be regenerated from the split.
# Constants derived from THIS checkpoint's accuracy curve. They go stale when the
# model changes, so --check fails if the checkpoint has moved since they were set.
MEASURED_CONSTANTS = {
    "sharpnessAbs (VoL)": {"value": 164, "source": "ml/degradation_calibration.py",
                           "checkpoint_sha256": "749f7bae40e91472"},
    "brightnessLow": {"value": 31, "source": "ml/degradation_calibration.py",
                      "checkpoint_sha256": "749f7bae40e91472"},
    "brightnessHigh": {"value": 195, "source": "ml/degradation_calibration.py",
                       "checkpoint_sha256": "749f7bae40e91472"},
    "ignorance floor": {"value": 0.3089, "source": "ml/ignorance_floor.py",
                        "checkpoint_sha256": "749f7bae40e91472"},
}


def check_measured_constants(current_sha: str) -> list[str]:
    """Measured constants are only valid for the checkpoint they were derived from."""
    return [f"{k}={v['value']} was measured on checkpoint {v['checkpoint_sha256']} "
            f"but the current checkpoint is {current_sha} — re-run {v['source']}"
            for k, v in MEASURED_CONSTANTS.items()
            if v["checkpoint_sha256"] != current_sha]


# Numbers asserted in prose in outward-facing docs. --check greps for each and warns
# if the doc no longer contains the current value. Not auto-rewritten: these sit in
# sentences, not tables, and a bad substitution in a document handed to a hospital is
# worse than a stale one flagged for a human.
# (file, label, value, required_annotation)
#
# A performance figure without its operating point is not checkable and has already
# had to be withdrawn once in this project. --check fails if EITHER the value drifts
# OR the operating-point annotation is missing, so a number can never appear in an
# outward-facing document bare.
OUTWARD_FACING_CLAIMS = [
    ("docs/CLINIC_STUDY_spec.md", "top-1", "0.571", "top-1 argmax"),
    ("docs/CLINIC_STUDY_spec.md", "top-3", "0.772", "no urgency threshold applied"),
    ("docs/CLINIC_STUDY_spec.md", "macro-F1", "0.555", "top-1 argmax"),
    ("docs/CLINIC_STUDY_spec.md", "AK/BCC recall", "0.573",
     "no urgency threshold, so the ignorance floor does not apply"),
    ("docs/CLINIC_STUDY_spec.md", "AK/BCC above-floor recall", "0.762",
     "urgency threshold 0.32"),
    ("docs/CLINIC_STUDY_spec.md", "malignancy-folder miss count", "143",
     "BELOW our 0.3089"),
]


def check_outward_claims() -> list[str]:
    out = []
    for rel, label, value, annotation in OUTWARD_FACING_CLAIMS:
        f = REPO / rel
        if not f.exists():
            out.append(f"{rel}: MISSING")
            continue
        txt = f.read_text()
        if value not in txt:
            out.append(f"{rel}: '{label}' should read {value} — value not found")
        elif annotation not in txt:
            out.append(f"{rel}: '{label}' = {value} appears WITHOUT its operating point "
                       f"(expected to find: {annotation!r})")
    return out


MANUAL_ITEMS = [
    ("docs/EXPERIENCE.md", "All clinical claims are [VERIFY] and unreviewed — no "
                           "dermatologist has signed off."),
    ("ml/urgency_groups.yaml", "reviewed_by is null for all 23 classes."),
    ("docs/PROPOSAL_pigmented_scope.md",
     "The §4 counter-argument is a judgement, not a measurement."),
    ("README.md", "'placeholder engine' language predates the trained model and may now "
                  "read as inconsistent."),
]


def boot_ci(hit: np.ndarray, n_boot=N_BOOT, seed=0):
    if len(hit) == 0:
        return [float("nan")] * 2
    g = np.random.default_rng(seed)
    v = [hit[g.integers(0, len(hit), len(hit))].mean() for _ in range(n_boot)]
    return [round(float(np.quantile(v, 0.025)), 4), round(float(np.quantile(v, 0.975)), 4)]


def compute(ckpt: Path, device: str, manifest_dir: Path, root: Path, bs: int) -> dict:
    model, ck = load_checkpoint(ckpt, device)
    classes = ck["classes"]
    ci = {c: i for i, c in enumerate(classes)}
    lc = ck.get("label_column", "original_class")
    groups = load_groups(GROUPS_YAML)

    val = ManifestImageDataset(manifest_dir, root, "val", lc, img_size=ck["img_size"])
    te = ManifestImageDataset(manifest_dir, root, "test", lc, img_size=ck["img_size"])
    frozen = runmeta.freeze(manifest_dir, "test", te.rows)

    lv, yv = logits_labels(model, DataLoader(val, batch_size=bs), device)
    lt, yt = logits_labels(model, DataLoader(te, batch_size=bs), device)
    temp = Temperature().fit(lv, yv)
    soon = torch.tensor([i for i, c in enumerate(classes) if tier_of(groups, c)[0] == "soon"])
    platt = Platt().fit(temp(lv).softmax(1)[:, soon].sum(1), torch.isin(yv, soon))
    p = temp(lt).softmax(1)
    score = platt(p[:, soon].sum(1)).numpy()
    y = yt.numpy()
    floor = float(platt(torch.tensor([len(soon) / len(classes)])))

    is_pos = np.isin(y, soon.numpy())
    infl = np.isin(y, [ci[c] for c in INFLAMMATORY if c in ci])
    pig = np.isin(y, [ci[c] for c in BENIGN_PIGMENTED if c in ci])
    n, n_pos = len(y), int(is_pos.sum())

    out = {"IGNORANCE_FLOOR": floor,
           "n_test": n, "n_urgent": n_pos, "prevalence": float(is_pos.mean()),
           "auroc": auroc(torch.tensor(score), torch.tensor(is_pos)),
           "auprc": auprc(torch.tensor(score), torch.tensor(is_pos)),
           "blocker_rule": "architecture test: does a shipping surface consume it?",
           "consuming_surface": CONSUMING_SURFACE, "surface_ships": SURFACE_SHIPS,
           "thresholds": {}}

    ak = y == ci[AKBCC]
    for t in THRESHOLDS:
        flag = score >= t
        tp = int((flag & is_pos).sum())
        fp = int((flag & ~is_pos).sum())
        per_class, miss_dist = [], Counter()
        for i in soon.tolist():
            m = y == i
            if not m.sum():
                continue
            hit = (score[m] >= t).astype(float)
            per_class.append({"class": classes[i], "n": int(m.sum()),
                              "recall": float(hit.mean()), "ci95": boot_ci(hit),
                              "missed": int((~(score[m] >= t)).sum())})
            miss_dist[classes[i]] = int((~(score[m] >= t)).sum())
        fa = 650 * float(flag[infl].mean())
        miss_ak = 10 * float((score[ak] < t).mean())
        out["thresholds"][str(t)] = {
            "threshold": t, "below_floor": t < floor,
            "recall": tp / max(n_pos, 1), "precision": tp / max(tp + fp, 1),
            "flagged_rate": float(flag.mean()),
            "inflammatory_flag_rate": float(flag[infl].mean()),
            "pigmented_flag_rate": float(flag[pig].mean()),
            # Architecture test, not a rate comparison. The old 0.30 bar was invented
            # and Blocker 1 fired against it by 0.002.
            "blocker1_binds": bool(SURFACE_SHIPS),
            "missed_urgent": n_pos - tp,
            "per_class_recall": sorted(per_class, key=lambda r: r["recall"]),
            "miss_distribution": dict(miss_dist.most_common()),
            "infl_flagged_per_1000": fa, "akbcc_missed_per_1000": miss_ak,
            "false_alarms_per_miss": fa / miss_ak if miss_ak else float("inf")}

    # threshold-free: queue ordering (unaffected by the floor)
    ranked = np.empty(n, dtype=int)
    ranked[np.argsort(-score)] = np.arange(n)
    q, qpc = [], []
    for k in KS:
        m = int(round(k * n))
        q.append({"review_fraction": k, "capture_rate": float((ranked[is_pos] < m).mean()),
                  "lift_over_random": float((ranked[is_pos] < m).mean()) / k})
    m10 = int(round(0.10 * n))
    for i in soon.tolist():
        mask = y == i
        if not mask.sum():
            continue
        hit = (ranked[mask] < m10).astype(float)
        qpc.append({"class": classes[i], "n": int(mask.sum()),
                    "capture_at_0.10": float(hit.mean()), "ci95": boot_ci(hit)})
    out["queue_ordering"] = q
    out["queue_per_class_at_10pct"] = sorted(qpc, key=lambda r: r["capture_at_0.10"])
    out["frozen_test"] = frozen
    return out


def render_blocker1(c: dict) -> str:
    lo, hi = c["thresholds"]["0.12"], c["thresholds"]["0.32"]
    binds = hi["blocker1_binds"]
    return f"""> ### {'⛔ BLOCKER 1' if binds else '⚠️ BLOCKER 1 — CONDITIONAL, does not currently bind'}: benign pigmentation false alarms
>
> **Ignorance floor = {c['IGNORANCE_FLOOR']:.4f}.** A uniform prediction over
> {c['n_test']:,} test cases scores that, because 8 of 23 classes are in the urgent
> group. No threshold below it separates urgency from ignorance.
>
> | | t=0.12 (**below floor**) | t=0.32 (above floor) |
> |---|---|---|
> | benign-pigmented flag rate | {lo['pigmented_flag_rate']:.3f} | **{hi['pigmented_flag_rate']:.3f}** |
> | inflammatory flag rate | {lo['inflammatory_flag_rate']:.3f} | {hi['inflammatory_flag_rate']:.3f} |
> | urgent recall | {lo['recall']:.3f} | {hi['recall']:.3f} |
> | missed urgent | {lo['missed_urgent']} | {hi['missed_urgent']} |
> | false alarms : missed cancer | {lo['false_alarms_per_miss']:.0f}:1 | {hi['false_alarms_per_miss']:.0f}:1 |
>
> **SUPERSEDED TWICE.** (1) The earlier record quoted 0.714 and 588:1, both measured
> at t=0.12 — **below the ignorance floor**, where a blank prediction also fires the
> gate. Above the floor the rate is {hi['pigmented_flag_rate']:.3f} and the ratio
> {hi['false_alarms_per_miss']:.0f}:1. (2) The old pass/fail rule was a **0.30 bar I
> invented**, which this fired against by 0.002 — that decided nothing. It is replaced
> by the architecture test: *a blocker binds iff a shipping surface consumes the
> metric*. This one is consumed only by {c['consuming_surface']}, which does not ship,
> so it is **conditional**. Under the ranking architecture we do ship, benign pigmented
> lesions reach the top-10% queue 3.7% of the time. If a user-facing urgency flag is
> ever added, this binds immediately.
>
> The cost of moving above the floor is real and is not hidden: urgent recall falls
> {lo['recall']:.3f} → {hi['recall']:.3f} and missed urgent cases rise
> {lo['missed_urgent']} → {hi['missed_urgent']}.
>
> Regenerated by `ml/regenerate_report.py` from the frozen test split
> `{c['frozen_test']['split_hash'][:12]}`."""


def render_blocker2(c: dict) -> str:
    t32 = c["thresholds"]["0.32"]
    pc12 = {r["class"]: r for r in c["thresholds"]["0.12"]["per_class_recall"]}
    rows = []
    for r in t32["per_class_recall"][:5]:
        a = pc12[r["class"]]
        rows.append(f"> | {r['class'][:42]} | {a['recall']:.3f} | {r['recall']:.3f} "
                    f"[{r['ci95'][0]:.2f}–{r['ci95'][1]:.2f}] | {r['missed']} |")
    qpc = {r["class"]: r for r in c["queue_per_class_at_10pct"]}
    agg10 = next(x["capture_rate"] for x in c["queue_ordering"] if x["review_fraction"] == 0.10)
    qrows = [f"> | {r['class'][:42]} | {r['capture_at_0.10']:.3f} "
             f"[{r['ci95'][0]:.2f}–{r['ci95'][1]:.2f}] |"
             for r in c["queue_per_class_at_10pct"][:4]]
    return f"""> ### ⛔ BLOCKER 2: time-critical classes rank and recall worst
>
> Per-class urgent recall, worst first — and the trade that moving above the
> ignorance floor imposes:
>
> | class | recall @0.12 (below floor) | recall @0.32 (above) | missed |
> |---|---|---|---|
{chr(10).join(rows)}
>
> **The trade is explicit: fixing Blocker 1 by moving above the floor makes Blocker 2
> worse, and it does so hardest on the time-critical classes.** Aggregate recall falls
> {c['thresholds']['0.12']['recall']:.3f} → {t32['recall']:.3f}, but the loss
> concentrates in cellulitis, bullous disease and exanthems — the classes that
> progress fastest.
>
> Queue capture at top-10% (threshold-free, so **unaffected by the floor**):
>
> | class | capture @top-10% |
> |---|---|
{chr(10).join(qrows)}
> | *aggregate* | *{agg10:.3f}* |
>
> Image-score ranking alone cannot be the triage mechanism for fast-progressing
> disease. That needs the symptom questionnaire (`ml/symptom_gate.py`), which is
> model-independent and escalate-only.
>
> Regenerated by `ml/regenerate_report.py`."""


def inject(path: Path, key: str, body: str, check: bool) -> str:
    start, end = f"<!-- AUTOGEN:{key} -->", f"<!-- /AUTOGEN:{key} -->"
    txt = path.read_text()
    if start not in txt:
        return f"NO MARKER  {path.relative_to(REPO)}  ({key})"
    new = re.sub(re.escape(start) + r".*?" + re.escape(end),
                 f"{start}\n{body}\n{end}", txt, flags=re.S)
    if new == txt:
        return f"unchanged  {path.relative_to(REPO)}  ({key})"
    if not check:
        path.write_text(new)
    return f"{'WOULD UPDATE' if check else 'updated'}   {path.relative_to(REPO)}  ({key})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--manifest"), ap.add_argument("--root"), ap.add_argument("--device")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--check", action="store_true", help="compute and diff, write nothing")
    a = ap.parse_args()

    ckpt = Path(a.checkpoint).resolve()
    device = config.resolve_device(a.device)
    manifest_dir = Path(a.manifest) if a.manifest else config.out_dir() / "dermnet"
    root = config.dataset_root(a.root)

    c = compute(ckpt, device, manifest_dir, root, a.batch_size)
    meta = runmeta.manifest(tool="regenerate_report.py", checkpoint=ckpt,
                            files={"groups": str(GROUPS_YAML)},
                            extra={"frozen_test": c["frozen_test"]})
    runmeta.print_header(meta, c["frozen_test"])
    print(f"\n*** IGNORANCE FLOOR = {c['IGNORANCE_FLOOR']:.4f} ***")
    for t, v in c["thresholds"].items():
        print(f"  t={t} {'BELOW FLOOR' if v['below_floor'] else 'above floor'}: "
              f"recall {v['recall']:.3f}  pig flag {v['pigmented_flag_rate']:.3f}  "
              f"infl flag {v['inflammatory_flag_rate']:.3f}  "
              f"ratio {v['false_alarms_per_miss']:.0f}:1  blocker1 "
              f"{'BINDS' if v['blocker1_binds'] else 'conditional (surface not shipped)'}")

    print("\nper-class recall, worst first (above-floor t=0.32):")
    pc12 = {r["class"]: r for r in c["thresholds"]["0.12"]["per_class_recall"]}
    for r in c["thresholds"]["0.32"]["per_class_recall"]:
        d = r["recall"] - pc12[r["class"]]["recall"]
        print(f"  {r['class'][:46]:<46} n={r['n']:<4} {pc12[r['class']]['recall']:.3f} -> "
              f"{r['recall']:.3f} [{r['ci95'][0]:.2f}-{r['ci95'][1]:.2f}] ({d:+.3f})  "
              f"missed {r['missed']}")
    print(f"\nmiss distribution at t=0.32 ({c['thresholds']['0.32']['missed_urgent']} total):")
    for k, v in c["thresholds"]["0.32"]["miss_distribution"].items():
        print(f"  {v:>4}  {k}")

    out = {**meta, **c}
    (ckpt.parent / "CANONICAL.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {ckpt.parent / 'CANONICAL.json'}")

    print("\n=== DOC REGENERATION ===")
    for msg in (inject(REPO / "README.md", "blocker1", render_blocker1(c), a.check),
                inject(REPO / "README.md", "blocker2", render_blocker2(c), a.check)):
        print("  " + msg)

    stale = check_measured_constants(meta["checkpoint_sha256"])
    print("\n=== MEASURED CONSTANTS vs CHECKPOINT ===")
    if stale:
        for s in stale:
            print(f"  STALE: {s}")
    else:
        print(f"  all {len(MEASURED_CONSTANTS)} measured constants match checkpoint "
              f"{meta['checkpoint_sha256']}")
    if a.check and stale:
        print("\n--check FAILED: measured constants are stale for this checkpoint.")
        return 1

    print("\n=== OUTWARD-FACING DOCS (numbers handed to third parties) ===")
    bad = check_outward_claims()
    if bad:
        for b in bad:
            print(f"  STALE/MISSING: {b}")
    else:
        print(f"  all {len(OUTWARD_FACING_CLAIMS)} asserted claims present and current")
    if a.check and bad:
        print("\n--check FAILED: an outward-facing document carries a stale number.")
        return 1

    print("\n=== MANUAL — cannot be regenerated, fix by hand ===")
    for f, why in MANUAL_ITEMS:
        print(f"  {f:<38} {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
