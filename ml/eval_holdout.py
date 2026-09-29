"""ml/eval_holdout.py
=========================================================================
Evaluates a checkpoint on REAL user-supplied phone photos, and measures how far
off-distribution they are.

WHY THIS IS NOW THE BINDING UNCERTAINTY
---------------------------------------
Every number we have — macro-F1 0.555, AUROC 0.878, the whole threshold table — is
measured on curated atlas photography: good lighting, framed lesions, clinical
settings, a mostly light-skinned population. The app's real input is phone photos:
uneven light, blur, off-angle, mostly acne and fungal complaints, on brown skin.

That gap is not a caveat to note and move past. It is the largest remaining source
of error in every claim we have made, and the direction is almost certainly
optimistic: models degrade under distribution shift, they do not improve.

This harness makes the size of the gap measurable instead of assumed. It emits the
SAME report schema as the frozen-split evaluation so the two sit side by side, and
adds a shift diagnostic:

  feature distance   Mahalanobis distance of each holdout image to the DermNet TRAIN
                     feature distribution, vs the same statistic for DermNet test.
                     If holdout distances sit far above, the model is extrapolating
                     and its confidences mean less than they claim.
  capture stats      blur (variance of Laplacian) and exposure vs the train set —
                     separates "different disease mix" from "worse photography".

INPUT FORMAT
------------
    <folder>/labels.csv     image_path,true_class[,urgency][,fitzpatrick][,notes]
    <folder>/<images...>

`image_path` is relative to the folder. `true_class` should use the checkpoint's
label space; anything unrecognised is reported rather than silently dropped. If
`urgency` (soon|routine) is given it is used directly, otherwise it is derived from
true_class via urgency_groups.yaml. `fitzpatrick` (I..VI) is optional but is the
ONLY way any per-skin-tone number becomes computable — with it, this harness
produces the first fairness measurement this project can honestly report.

    ml/.venv/bin/python ml/eval_holdout.py data/ml/runs_final/best.pt --dir data/holdout
=========================================================================
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import runmeta  # noqa: E402
from dataset import ManifestImageDataset, build_transforms  # noqa: E402
from ood import Mahalanobis, collect, penultimate  # noqa: E402
from pipeline import image_quality  # noqa: E402
from train import load_checkpoint  # noqa: E402
from urgency import (GROUPS_YAML, REPORT_THRESHOLDS, Platt, Temperature, auprc,  # noqa: E402
                     auroc, load_groups, logits_labels, threshold_table, tier_of)


def read_labels(folder: Path) -> list[dict]:
    f = folder / "labels.csv"
    if not f.exists():
        raise SystemExit(
            f"{f} not found.\n\nExpected format:\n"
            "  image_path,true_class,urgency,fitzpatrick,notes\n"
            "  IMG_0001.jpg,Tinea Ringworm Candidiasis and other Fungal Infections,routine,V,\n"
            "  IMG_0002.jpg,Acne and Rosacea Photos,routine,IV,\n\n"
            "Only image_path and true_class are required. `fitzpatrick` is optional but "
            "is the only route to a real fairness number."
        )
    with f.open(newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r.get("image_path")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--dir", required=True, help="folder with labels.csv + images")
    ap.add_argument("--manifest"), ap.add_argument("--root"), ap.add_argument("--device")
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()

    ckpt = Path(a.checkpoint).resolve()
    folder = Path(a.dir).resolve()
    device = config.resolve_device(a.device)
    model, ck = load_checkpoint(ckpt, device)
    classes = ck["classes"]
    lc = ck.get("label_column", "original_class")
    manifest_dir = Path(a.manifest) if a.manifest else config.out_dir() / "dermnet"
    root = config.dataset_root(a.root)
    groups = load_groups(GROUPS_YAML)

    rows = read_labels(folder)
    print(f"holdout: {len(rows)} rows from {folder}")

    meta = runmeta.manifest(tool="eval_holdout.py", checkpoint=ckpt,
                            files={"groups": str(GROUPS_YAML), "labels": folder / "labels.csv"},
                            extra={"holdout_dir": str(folder), "n_rows": len(rows)})
    runmeta.print_header(meta)

    # ---- reference distributions from the DermNet corpus ----
    tr = ManifestImageDataset(manifest_dir, root, "train", lc, img_size=ck["img_size"])
    te = ManifestImageDataset(manifest_dir, root, "test", lc, img_size=ck["img_size"])
    f_tr, _ = collect(model, DataLoader(tr, batch_size=64, shuffle=True), device, 40)
    maha = Mahalanobis(f_tr)
    f_te, _ = collect(model, DataLoader(te, batch_size=64), device, 20)
    d_te = maha(f_te)

    val = ManifestImageDataset(manifest_dir, root, "val", lc, img_size=ck["img_size"])
    lv, yv = logits_labels(model, DataLoader(val, batch_size=64), device)
    temp = Temperature().fit(lv, yv)
    soon_idx = torch.tensor([i for i, c in enumerate(classes) if tier_of(groups, c)[0] == "soon"])
    platt = Platt().fit(temp(lv).softmax(1)[:, soon_idx].sum(1), torch.isin(yv, soon_idx))

    # ---- score the holdout ----
    tf = build_transforms(ck["img_size"], train=False)
    unknown_labels, per_image = Counter(), []
    feats, logits = [], []
    quality = []
    ci = {c: i for i, c in enumerate(classes)}

    for r in rows:
        p = folder / r["image_path"]
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:  # noqa: BLE001
            print(f"  unreadable: {r['image_path']} ({type(e).__name__})")
            continue
        quality.append(image_quality(img))
        x = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits.append(model(x).cpu())
            feats.append(penultimate(model, x).cpu())
        tc = (r.get("true_class") or "").strip()
        if tc and tc not in ci:
            unknown_labels[tc] += 1
        urg = (r.get("urgency") or "").strip().lower()
        if urg not in ("soon", "routine"):
            urg = tier_of(groups, tc)[0] if tc in ci else ""
        per_image.append({"image_path": r["image_path"], "true_class": tc,
                          "urgency": urg, "fitzpatrick": (r.get("fitzpatrick") or "").strip()})

    if not per_image:
        raise SystemExit("no readable images")
    logits, feats = torch.cat(logits), torch.cat(feats)
    p_cal = temp(logits).softmax(1)
    score = platt(p_cal[:, soon_idx].sum(1))
    d_ho = maha(feats)

    # ---- distribution shift ----
    print("\n=== DISTRIBUTION SHIFT vs DermNet train ===")
    print(f"  feature distance (Mahalanobis)")
    print(f"    DermNet test  mean {d_te.mean():.1f}  p50 {d_te.median():.1f}  "
          f"p95 {d_te.quantile(0.95):.1f}")
    print(f"    holdout       mean {d_ho.mean():.1f}  p50 {d_ho.median():.1f}  "
          f"p95 {d_ho.quantile(0.95):.1f}")
    ratio = float(d_ho.mean() / d_te.mean())
    frac_beyond = float((d_ho > d_te.quantile(0.95)).float().mean())
    print(f"    ratio {ratio:.2f}x   ·  {frac_beyond:.1%} of holdout beyond DermNet test p95")
    if ratio > 1.15 or frac_beyond > 0.25:
        print("    >>> SUBSTANTIAL SHIFT. The model is extrapolating; treat every metric "
              "below\n        as an optimistic bound and its confidences as unreliable.")

    sharp = [q["sharpness"] for q in quality]
    bright = [q["brightness"] for q in quality]
    n_fail_q = sum(1 for q in quality if not q["ok"])
    print(f"  capture quality (holdout)")
    print(f"    sharpness  mean {np.mean(sharp):.0f}  p10 {np.percentile(sharp, 10):.0f}")
    print(f"    brightness mean {np.mean(bright):.0f}")
    print(f"    would fail the app's quality gate: {n_fail_q}/{len(quality)} "
          f"({n_fail_q / len(quality):.0%})")

    # ---- accuracy, if labels were given ----
    out_metrics = {}
    labelled = [i for i, e in enumerate(per_image) if e["true_class"] in ci]
    if labelled:
        idx = torch.tensor(labelled)
        y = torch.tensor([ci[per_image[i]["true_class"]] for i in labelled])
        pred = p_cal[idx].argmax(1)
        k = min(3, len(classes))
        top3 = p_cal[idx].topk(k, dim=1).indices
        t1 = float((pred == y).float().mean())
        t3 = float((top3 == y.unsqueeze(1)).any(1).float().mean())
        print(f"\n=== CLASSIFICATION on {len(labelled)} labelled holdout images ===")
        print(f"  top-1 {t1:.3f}   top-3 {t3:.3f}")
        print(f"  (frozen-split reference: top-1 0.571  top-3 0.772 — any drop is the "
              f"real-world gap)")
        out_metrics["classification"] = {"n": len(labelled), "top1": t1, "top3": t3}

    urg_rows = [i for i, e in enumerate(per_image) if e["urgency"] in ("soon", "routine")]
    if urg_rows and len({per_image[i]["urgency"] for i in urg_rows}) == 2:
        idx = torch.tensor(urg_rows)
        is_pos = torch.tensor([per_image[i]["urgency"] == "soon" for i in urg_rows])
        s = score[idx]
        print(f"\n=== URGENCY on {len(urg_rows)} holdout images ===")
        print(f"  prevalence {float(is_pos.float().mean()):.3f}")
        print(f"  AUROC {auroc(s, is_pos):.4f}  AUPRC {auprc(s, is_pos):.4f}  "
              f"[no-skill {float(is_pos.float().mean()):.4f}]")
        print(f"  (frozen-split reference: AUROC 0.878  AUPRC 0.745)")
        tbl = threshold_table(s, is_pos, REPORT_THRESHOLDS)
        print("  thresh  recall  precision  flagged  missed")
        for r in tbl:
            print(f"   {r['threshold']:.2f}   {r['recall']:.3f}     {r['precision']:.3f}"
                  f"     {r['flagged_rate']:.3f}    {r['missed']}")
        out_metrics["urgency"] = {"n": len(urg_rows), "auroc": auroc(s, is_pos),
                                  "auprc": auprc(s, is_pos), "thresholds": tbl}

    # ---- the first fairness number this project can honestly report ----
    # ONLY reader-assigned fitzpatrick counts here. self_reported_tone is read
    # separately below for the three-way tone comparison, and cannot enable this block.
    fitz = defaultdict(list)
    for i, e in enumerate(per_image):
        if e["fitzpatrick"]:
            fitz[e["fitzpatrick"].upper()].append(i)
    if fitz:
        print("\n=== BY FITZPATRICK (first measurable fairness number) ===")
        fair = {}
        for band, ids in sorted(fitz.items()):
            ids_t = torch.tensor(ids)
            lab = [i for i in ids if per_image[i]["true_class"] in ci]
            acc = float("nan")
            if lab:
                yy = torch.tensor([ci[per_image[i]["true_class"]] for i in lab])
                acc = float((p_cal[torch.tensor(lab)].argmax(1) == yy).float().mean())
            fair[band] = {"n": len(ids), "top1": acc,
                          "mean_urgent_score": float(score[ids_t].mean()),
                          "mean_feature_distance": float(d_ho[ids_t].mean())}
            print(f"  {band:<4} n={len(ids):<4} top-1 {acc:.3f}  "
                  f"mean urgent {fair[band]['mean_urgent_score']:.3f}  "
                  f"feat-dist {fair[band]['mean_feature_distance']:.1f}")
        out_metrics["by_fitzpatrick"] = fair
        # fairness_measurable means A TRAINED READER ASSIGNED THE TONE. Self-report
        # (self_reported_tone, Monk swatch) must never flip it: it is noisy, and a
        # flag that quietly downgrades to "someone ticked a colour" would let an
        # unvalidated fairness claim through exactly the check meant to stop it.
        meta["fairness_measurable"] = True
        meta["fitzpatrick"] = "reader-assigned"
        print("  NOTE: small n per band — treat as directional, not a claim.")
    else:
        print("\n=== BY FITZPATRICK ===\n  No fitzpatrick column supplied — still no "
              "measurable fairness number.")

    if unknown_labels:
        print(f"\nlabels not in the model's label space (reported, not dropped silently):")
        for k_, v in unknown_labels.most_common():
            print(f"  {v:>3}  {k_}")

    out = {**meta, "n_scored": len(per_image),
           "distribution_shift": {
               "dermnet_test_maha_mean": float(d_te.mean()),
               "holdout_maha_mean": float(d_ho.mean()),
               "ratio": ratio, "frac_beyond_test_p95": frac_beyond,
               "substantial_shift": bool(ratio > 1.15 or frac_beyond > 0.25)},
           "capture_quality": {"sharpness_mean": float(np.mean(sharp)),
                               "brightness_mean": float(np.mean(bright)),
                               "n_failing_quality_gate": n_fail_q,
                               "frac_failing_quality_gate": n_fail_q / len(quality)},
           "metrics": out_metrics,
           "unknown_labels": dict(unknown_labels),
           "per_image": [{**e, "urgent_score": float(score[i]),
                          "pred_top1": classes[int(p_cal[i].argmax())],
                          "feature_distance": float(d_ho[i])}
                         for i, e in enumerate(per_image)],
           "caveat": "Holdout metrics are the ones that matter for deployment. Where they "
                     "disagree with frozen-split metrics, believe these."}
    p = ckpt.parent / "holdout_report.json"
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
