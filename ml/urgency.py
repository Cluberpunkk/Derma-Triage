"""ml/urgency.py
=========================================================================
Binary "does this need seeing soon?" evaluation, done properly.

An earlier version of this file reported a recall/precision table that was wrong in
three ways. All three are fixed here, and the fixes matter more than the headline:

  1. THE TARGET WAS CONTAMINATED. "Melanoma Skin Cancer Nevi and Moles" merges
     benign nevi with melanoma. Counting the whole folder as urgent inflates both
     prevalence and the miss count. Every metric is now reported twice: with that
     class, and with it excluded.
  2. THE BINARY SCORE WAS NEVER CALIBRATED. A temperature fitted on the 23-way
     softmax does not transfer to a SUM OF MARGINALS — summing calibrated marginals
     does not give a calibrated binary probability. A separate Platt scaler is now
     fitted on the summed urgent score.
  3. EVERY NUMBER WAS THRESHOLD-DEPENDENT AND POINT-ESTIMATED. AUROC/AUPRC (with the
     no-skill baseline printed beside them) and bootstrap CIs are now reported, so a
     difference can be told apart from noise.

Plus prevalence sensitivity: DermNet's 26.4% urgent prevalence is optimistic for a
consumer app whose users skew to acne/fungal/eczema. Precision is recomputed under
5/10/15% assumed prevalence, because precision — unlike recall — moves with the base
rate, and the DermNet figure is an upper bound.

The grouping lives in ml/urgency_groups.yaml so a dermatologist can revise it
without touching Python.

    ml/.venv/bin/python ml/urgency.py data/ml/runs_final/best.pt
=========================================================================
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import runmeta  # noqa: E402
from dataset import ManifestImageDataset  # noqa: E402
from train import load_checkpoint  # noqa: E402

GROUPS_YAML = Path(__file__).resolve().parent / "urgency_groups.yaml"
REPORT_THRESHOLDS = [0.12, 0.32, 0.72]
ASSUMED_PREVALENCE = [0.05, 0.10, 0.15]


# ------------------------------------------------------------------ grouping

def load_groups(path=GROUPS_YAML) -> dict:
    d = yaml.safe_load(Path(path).read_text())
    return {"default": d.get("default_tier", "soon"),
            "classes": d.get("classes", {}), "version": d.get("version")}


def tier_of(groups: dict, cls: str) -> tuple[str, str]:
    e = groups["classes"].get(cls)
    if not e:
        return groups["default"], "unlisted"
    return e.get("tier", groups["default"]), e.get("confidence", "unknown")


# ------------------------------------------------------------------ calibration

class Temperature:
    """Scalar temperature for the 23-way logits (multi-class calibration)."""

    def __init__(self):
        self.T = 1.0

    def fit(self, logits, y):
        logT = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=300)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(logits / logT.exp(), y)
            loss.backward()
            return loss

        opt.step(closure)
        self.T = float(logT.exp())
        return self

    def __call__(self, logits):
        return logits / self.T


class Platt:
    """sigmoid(a * logit(p) + b) fitted on the BINARY score.

    Needed because a sum of calibrated marginals is not itself calibrated: summing
    probabilities over a class subset compounds their individual errors and has no
    reason to land on the right binary probability.
    """

    def __init__(self):
        self.a, self.b = 1.0, 0.0

    @staticmethod
    def _logit(p, eps=1e-6):
        p = p.clamp(eps, 1 - eps)
        return torch.log(p / (1 - p))

    def fit(self, p, is_pos):
        z = self._logit(p)
        a = torch.ones(1, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([a, b], lr=0.1, max_iter=300)
        t = is_pos.float()

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(a * z + b, t)
            loss.backward()
            return loss

        opt.step(closure)
        self.a, self.b = float(a), float(b)
        return self

    def __call__(self, p):
        return torch.sigmoid(self.a * self._logit(p) + self.b)


def ece_binary(p, is_pos, bins=15):
    e, n = 0.0, len(p)
    edges = torch.linspace(0, 1, bins + 1)
    curve = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any():
            conf, freq = float(p[m].mean()), float(is_pos[m].float().mean())
            e += float(m.sum()) / n * abs(freq - conf)
            curve.append({"bin_lo": round(float(lo), 3), "bin_hi": round(float(hi), 3),
                          "n": int(m.sum()), "mean_pred": round(conf, 4),
                          "observed_rate": round(freq, 4)})
    return e, curve


# ------------------------------------------------------------------ metrics

def auroc(score, is_pos) -> float:
    r = score.argsort().argsort().float() + 1
    n1 = int(is_pos.sum())
    n0 = len(score) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[is_pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(score, is_pos) -> float:
    """Average precision — area under precision/recall, step-wise."""
    order = score.argsort(descending=True)
    pos = is_pos[order].float()
    tp = pos.cumsum(0)
    prec = tp / torch.arange(1, len(pos) + 1, dtype=torch.float)
    n_pos = float(pos.sum())
    return float((prec * pos).sum() / n_pos) if n_pos else float("nan")


def recall_precision(score, is_pos, t):
    flag = score >= t
    tp = int((flag & is_pos).sum())
    fp = int((flag & ~is_pos).sum())
    n_pos = int(is_pos.sum())
    return (tp / max(n_pos, 1), tp / max(tp + fp, 1),
            float(flag.float().mean()), n_pos - tp)


def bootstrap_ci(score, is_pos, t, n_boot=1000, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = len(score)
    recs, precs = [], []
    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g)
        r, p, _, _ = recall_precision(score[idx], is_pos[idx], t)
        recs.append(r)
        precs.append(p)
    q = lambda v: (round(float(torch.tensor(v).quantile(0.025)), 4),  # noqa: E731
                   round(float(torch.tensor(v).quantile(0.975)), 4))
    return q(recs), q(precs)


def precision_at_prevalence(tpr, fpr, prev):
    """Precision under an assumed base rate — recall is invariant, precision is not."""
    num = tpr * prev
    den = num + fpr * (1 - prev)
    return num / den if den > 0 else float("nan")


def threshold_table(score, is_pos, ths):
    rows = []
    n_pos = int(is_pos.sum())
    n_neg = len(score) - n_pos
    for t in ths:
        flag = score >= t
        tp = int((flag & is_pos).sum())
        fp = int((flag & ~is_pos).sum())
        tpr = tp / max(n_pos, 1)
        fpr = fp / max(n_neg, 1)
        rows.append({"threshold": round(float(t), 3), "recall": tpr,
                     "precision": tp / max(tp + fp, 1),
                     "fpr": fpr, "flagged_rate": float(flag.float().mean()),
                     "missed": n_pos - tp,
                     "precision_at_prev": {f"{p:.0%}": precision_at_prevalence(tpr, fpr, p)
                                           for p in ASSUMED_PREVALENCE}})
    return rows


def per_class_recall(score, yt, classes, soon_idx, ths):
    """Recall per URGENT class at each threshold.

    Aggregate recall hides which class is being missed. AK/BCC is the largest miss
    bucket and the most consequential to miss, so it must be visible in the report
    rather than only in a contact sheet.
    """
    rows = []
    for i in soon_idx:
        m = yt == i
        n = int(m.sum())
        if n == 0:
            continue
        r = {"class": classes[i], "n": n}
        for t_ in ths:
            r[f"recall_at_{t_}"] = float((score[m] >= t_).float().mean())
            r[f"missed_at_{t_}"] = int((score[m] < t_).sum())
        rows.append(r)
    return sorted(rows, key=lambda r: r[f"recall_at_{ths[0]}"])


def queue_ordering(score, is_pos, ks=(0.05, 0.10, 0.20, 0.30)):
    """The model as a PRIORITISER, not a filter.

    AUROC 0.878 says the model ranks well, and ranking survives low prevalence in a
    way that thresholding does not: if a human reviews the top-k% by urgency score,
    nobody is excluded, so the cost of a "miss" collapses from "never seen" to
    "seen later". This is the only urgency framing the data actually supports.
    """
    n = len(score)
    n_pos = int(is_pos.sum())
    pos_sorted = is_pos[score.argsort(descending=True)]
    rows = []
    for k in ks:
        m = max(1, int(round(k * n)))
        caught = int(pos_sorted[:m].sum())
        cap = caught / max(n_pos, 1)
        rows.append({"review_fraction": k, "n_reviewed": m, "urgent_found": caught,
                     "capture_rate": cap, "random_baseline": k,
                     "lift_over_random": cap / k if k else float("nan")})
    return rows


def per_class_queue_capture(score, yt, classes, soon_idx, ks=(0.05, 0.10, 0.20, 0.30),
                            n_boot=400, seed=0):
    """Per-class capture in the top-k% of the queue, with bootstrap CIs.

    Aggregate capture can look healthy while a specific class ranks badly, and the
    queue framing is only safe if the TIME-CRITICAL classes rank high. Cellulitis is
    the one to watch: it has the worst threshold recall and it progresses in hours,
    so poor ranking there means queue-ordering fails exactly where failure costs
    most.
    """
    n = len(score)
    ranked_y = yt[score.argsort(descending=True)]
    rows = {}
    for i in soon_idx:
        n_c = int((yt == i).sum())
        if n_c == 0:
            continue
        r = {"class": classes[i], "n": n_c}
        for k in ks:
            m = max(1, int(round(k * n)))
            r[f"capture_at_{k}"] = int((ranked_y[:m] == i).sum()) / n_c
        rows[int(i)] = r

    g = torch.Generator().manual_seed(seed)
    boots = {(int(i), k): [] for i in soon_idx for k in ks}
    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g)
        s_b, y_b = score[idx], yt[idx]
        ry = y_b[s_b.argsort(descending=True)]
        for i in soon_idx:
            n_c = int((y_b == i).sum())
            if n_c == 0:
                continue
            for k in ks:
                m = max(1, int(round(k * n)))
                boots[(int(i), k)].append(int((ry[:m] == i).sum()) / n_c)
    for i in soon_idx:
        if int(i) not in rows:
            continue
        for k in ks:
            v = boots[(int(i), k)]
            if v:
                tv = torch.tensor(v)
                rows[int(i)][f"ci95_at_{k}"] = [round(float(tv.quantile(0.025)), 4),
                                                round(float(tv.quantile(0.975)), 4)]
    return sorted(rows.values(), key=lambda r: r[f"capture_at_{ks[1]}"])


@torch.no_grad()
def logits_labels(model, loader, device):
    L, Y = [], []
    for x, y in loader:
        L.append(model(x.to(device)).cpu())
        Y.append(y)
    return torch.cat(L), torch.cat(Y)


# ------------------------------------------------------------------ main

def evaluate_target(name, p_soon_raw, platt, is_pos, classes_soon, out_lines):
    """One full metric block for a given definition of 'urgent'."""
    p = platt(p_soon_raw)
    prev = float(is_pos.float().mean())
    ece, curve = ece_binary(p, is_pos)
    ece_uncal, _ = ece_binary(p_soon_raw, is_pos)
    a_roc, a_prc = auroc(p, is_pos), auprc(p, is_pos)

    out_lines.append(f"\n=== TARGET: {name} ===")
    out_lines.append(f"  urgent classes: {len(classes_soon)}   prevalence {prev:.4f} "
                     f"({int(is_pos.sum())}/{len(is_pos)})")
    out_lines.append(f"  binary ECE   uncalibrated {ece_uncal:.4f} -> Platt {ece:.4f}"
                     f"   (a={platt.a:.3f}, b={platt.b:.3f})")
    out_lines.append(f"  AUROC {a_roc:.4f}   AUPRC {a_prc:.4f}   "
                     f"[no-skill AUPRC = prevalence = {prev:.4f}]")
    lift = a_prc / prev if prev else float("nan")
    out_lines.append(f"  AUPRC lift over no-skill: {lift:.2f}x")

    rows = threshold_table(p, is_pos, REPORT_THRESHOLDS)
    out_lines.append("  thresh  recall [95% CI]        precision [95% CI]     flagged  missed")
    for r in rows:
        rci, pci = bootstrap_ci(p, is_pos, r["threshold"])
        r["recall_ci95"], r["precision_ci95"] = rci, pci
        out_lines.append(
            f"   {r['threshold']:.2f}   {r['recall']:.3f} [{rci[0]:.3f}-{rci[1]:.3f}]"
            f"   {r['precision']:.3f} [{pci[0]:.3f}-{pci[1]:.3f}]"
            f"   {r['flagged_rate']:.3f}   {r['missed']}")

    out_lines.append("  precision if real-world urgent prevalence were lower:")
    for r in rows:
        pr = "  ".join(f"{k}:{v:.3f}" for k, v in r["precision_at_prev"].items())
        out_lines.append(f"   t={r['threshold']:.2f}  observed:{r['precision']:.3f}   {pr}")

    return {"name": name, "prevalence": prev, "n_urgent": int(is_pos.sum()),
            "ece_uncalibrated": ece_uncal, "ece_calibrated": ece,
            "platt": {"a": platt.a, "b": platt.b},
            "auroc": a_roc, "auprc": a_prc, "auprc_no_skill": prev,
            "auprc_lift": lift, "reliability_curve": curve, "thresholds": rows,
            "urgent_classes": classes_soon}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--manifest"), ap.add_argument("--root"), ap.add_argument("--device")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--groups", default=str(GROUPS_YAML))
    a = ap.parse_args()

    ckpt = Path(a.checkpoint).resolve()
    device = config.resolve_device(a.device)
    model, ck = load_checkpoint(ckpt, device)
    classes = ck["classes"]
    lc = ck.get("label_column", "original_class")
    manifest_dir = Path(a.manifest) if a.manifest else config.out_dir() / "dermnet"
    root = config.dataset_root(a.root)

    groups = load_groups(a.groups)
    val = ManifestImageDataset(manifest_dir, root, "val", lc, img_size=ck["img_size"])
    te = ManifestImageDataset(manifest_dir, root, "test", lc, img_size=ck["img_size"])

    frozen = runmeta.freeze(manifest_dir, "test", te.rows)
    meta = runmeta.manifest(tool="urgency.py", checkpoint=ckpt,
                            files={"groups": a.groups},
                            extra={"groups_version": groups["version"],
                                   "frozen_test": frozen})
    runmeta.print_header(meta, frozen)

    lv, yv = logits_labels(model, DataLoader(val, batch_size=a.batch_size), device)
    lt, yt = logits_labels(model, DataLoader(te, batch_size=a.batch_size), device)

    temp = Temperature().fit(lv, yv)
    pv, pt = temp(lv).softmax(1), temp(lt).softmax(1)
    print(f"\n23-way temperature T = {temp.T:.3f}")

    tiers = {c: tier_of(groups, c) for c in classes}
    soon = [i for i, c in enumerate(classes) if tiers[c][0] == "soon"]
    soon_t = torch.tensor(soon)

    # ---------------- item 1: per-class audit ----------------
    is_pos_full = torch.isin(yt, soon_t)
    p_soon_full_raw = pt[:, soon_t].sum(1)
    platt_full = Platt().fit(pv[:, soon_t].sum(1), torch.isin(yv, soon_t))
    p_full = platt_full(p_soon_full_raw)

    print("\n=== PER-CLASS AUDIT (test split) ===")
    print(f"{'class':<52} {'tier':<8} {'conf':<7} {'n':>5} {'%urg':>6} "
          f"{'mean_p':>7} {'miss@.12':>9} {'miss@.32':>9} {'miss@.72':>9}")
    audit = []
    n_urg_total = int(is_pos_full.sum())
    for i, c in enumerate(classes):
        m = yt == i
        n = int(m.sum())
        if n == 0:
            continue
        tier, conf = tiers[c]
        share = (n / n_urg_total) if tier == "soon" else 0.0
        row = {"class": c, "tier": tier, "confidence": conf, "n_test": n,
               "share_of_urgent_mass": share, "mean_urgent_score": float(p_full[m].mean())}
        for t in REPORT_THRESHOLDS:
            row[f"missed_at_{t}"] = int(((p_full[m] < t) & is_pos_full[m]).sum())
        audit.append(row)
        print(f"{c[:52]:<52} {tier:<8} {conf:<7} {n:>5} {share:>6.1%} "
              f"{row['mean_urgent_score']:>7.3f} "
              f"{row['missed_at_0.12']:>9} {row['missed_at_0.32']:>9} {row['missed_at_0.72']:>9}")

    mel = "Melanoma Skin Cancer Nevi and Moles"
    mel_i = classes.index(mel) if mel in classes else None
    if mel_i is not None:
        n_mel = int((yt == mel_i).sum())
        print(f"\n  >>> '{mel}' contributes {n_mel}/{n_urg_total} = "
              f"{n_mel / n_urg_total:.1%} of ALL urgent ground-truth mass.")
        print(f"      It is confidence:low — the folder merges benign nevi with melanoma, "
              f"so that\n      share is largely benign and inflates prevalence and misses alike.")

    # ---------------- items 3,4,5: full + excluded ----------------
    lines: list[str] = []
    res_full = evaluate_target("ALL urgent classes (as grouped)", p_soon_full_raw,
                               platt_full, is_pos_full,
                               [classes[i] for i in soon], lines)

    results = {"with_melanoma_folder": res_full}
    if mel_i is not None:
        keep = yt != mel_i                              # drop those rows entirely
        soon_x = [i for i in soon if i != mel_i]
        soon_xt = torch.tensor(soon_x)
        keep_v = yv != mel_i
        platt_x = Platt().fit(pv[keep_v][:, soon_xt].sum(1),
                              torch.isin(yv[keep_v], soon_xt))
        res_x = evaluate_target(
            "EXCLUDING 'Melanoma ... Nevi and Moles' (contaminated folder)",
            pt[keep][:, soon_xt].sum(1), platt_x, torch.isin(yt[keep], soon_xt),
            [classes[i] for i in soon_x], lines)
        results["without_melanoma_folder"] = res_x

    print("\n".join(lines))

    pcr = per_class_recall(p_full, yt, classes, soon, REPORT_THRESHOLDS)
    print("\n=== PER-CLASS RECALL (urgent classes, worst first) ===")
    print(f"{'class':<52} {'n':>5} " + " ".join(f"{'r@'+str(t_):>8}" for t_ in REPORT_THRESHOLDS)
          + " " + " ".join(f"{'miss@'+str(t_):>10}" for t_ in REPORT_THRESHOLDS))
    for r in pcr:
        print(f"{r['class'][:52]:<52} {r['n']:>5} "
              + " ".join(f"{r[f'recall_at_{t_}']:>8.3f}" for t_ in REPORT_THRESHOLDS)
              + " " + " ".join(f"{r[f'missed_at_{t_}']:>10}" for t_ in REPORT_THRESHOLDS))

    qo = queue_ordering(p_full, is_pos_full)
    print("\n" + "=" * 78)
    print("PRIMARY METRIC — QUEUE ORDERING (model as prioritiser, nobody excluded)")
    print("=" * 78)
    print("  review top   n      urgent found   capture   random   lift")
    for r in qo:
        print(f"   {r['review_fraction']:>6.0%}   {r['n_reviewed']:>5}   "
              f"{r['urgent_found']:>10}   {r['capture_rate']:>7.3f}   "
              f"{r['random_baseline']:>6.2f}   {r['lift_over_random']:>4.2f}x")
    print("  Miss cost collapses here: a low-ranked urgent case is seen LATER, not never.")

    KS = (0.05, 0.10, 0.20, 0.30)
    pcq = per_class_queue_capture(p_full, yt, classes, soon, KS)
    print("\n  PER-CLASS CAPTURE (worst at top-10% first) — [95% CI]")
    print(f"  {'class':<50} {'n':>4} " + "  ".join(f"{'top'+str(int(k*100))+'%':>20}" for k in KS))
    agg10 = next(r["capture_rate"] for r in qo if r["review_fraction"] == 0.10)
    for r in pcq:
        cells = []
        for k in KS:
            ci = r.get(f"ci95_at_{k}", [float('nan')] * 2)
            cells.append(f"{r[f'capture_at_{k}']:.3f}[{ci[0]:.2f}-{ci[1]:.2f}]".rjust(20))
        print(f"  {r['class'][:50]:<50} {r['n']:>4} " + "  ".join(cells))
    worst = pcq[0]
    gap = worst["capture_at_0.1"] - agg10
    print(f"\n  aggregate capture @top-10% = {agg10:.3f}; worst class "
          f"'{worst['class'][:40]}' = {worst['capture_at_0.1']:.3f} ({gap:+.3f})")
    cell = next((r for r in pcq if "Cellulitis" in r["class"]), None)
    if cell:
        cgap = cell["capture_at_0.1"] - agg10
        flag = ("BLOCKER" if cgap < -0.05 else "within aggregate")
        print(f"  CELLULITIS (progresses in hours): capture@10% = "
              f"{cell['capture_at_0.1']:.3f} ({cgap:+.3f} vs aggregate) -> {flag}")

    low_conf = [c for c in classes if tiers[c][1] == "low"]
    print(f"\nLOW-CONFIDENCE GROUPINGS (metrics from these are contaminated): {low_conf}")

    out = {**meta, "temperature_23way": temp.T, "per_class_audit": audit,
           "per_class_recall": pcr, "queue_ordering": qo,
           "per_class_queue_capture": pcq,
           "primary_metric": "queue_ordering",
           "targets": results, "low_confidence_classes": low_conf,
           "assumed_prevalences": ASSUMED_PREVALENCE,
           "caveat": "Grouping is [VERIFY] — reviewed_by is null for every class. "
                     "Binary accuracy is not reported: with 26% prevalence the "
                     "always-routine baseline scores 0.74, so accuracy is misleading. "
                     "Use AUPRC against the no-skill baseline."}
    p = ckpt.parent / "urgency_report.json"
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
