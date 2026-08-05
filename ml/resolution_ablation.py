"""ml/resolution_ablation.py
=========================================================================
Two questions the mimicry finding raises.

ITEM 1 — IS SK <-> AK/BCC CONFUSION RESOLUTION-SENSITIVE?
Seborrheic keratosis vs pigmented BCC is separated clinically by FINE TEXTURE:
milia-like cysts, comedo-like openings, a stuck-on waxy surface versus arborizing
vessels and a pearly edge. If our confusion is that mimicry, it should shrink as
input resolution rises — and if it does, the fix is capture and preprocessing
(resolution, crop-to-lesion, multi-crop), not architecture.

The backbone is fully convolutional with global pooling, so it accepts inputs larger
than the 224 it was trained at without retraining. Test-time resolution increase is
a known free win (the train/test resolution discrepancy), so this is a fair
ablation, though a model FINE-TUNED at the larger size would do better still.

ITEM 4 — WHERE DOES NON-PIGMENTED AK/BCC ACTUALLY GO?
Non-pigmented AK/BCC is erythematous and scaly, which is the same visual territory
as eczema, psoriasis and tinea — the highest-volume conditions in an Indian consumer
app. If our worst failure on the most consequential class lands exactly where our
traffic concentrates, that matters more than the aggregate number.

    ml/.venv/bin/python ml/resolution_ablation.py data/ml/runs_final/best.pt
=========================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
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
from shortcut_test import darkness_stats  # noqa: E402
from train import load_checkpoint  # noqa: E402
from urgency import (GROUPS_YAML, Platt, Temperature, load_groups, logits_labels,  # noqa: E402
                     tier_of)

RESOLUTIONS = [224, 320, 448]
SK = "Seborrheic Keratoses and other Benign Tumors"
AKBCC = "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions"
INFLAMMATORY = ["Eczema Photos", "Atopic Dermatitis Photos",
                "Psoriasis pictures Lichen Planus and related diseases",
                "Tinea Ringworm Candidiasis and other Fungal Infections",
                "Poison Ivy Photos and other Contact Dermatitis"]


def source_image_sizes(root: Path, rows, n=300):
    ws, hs = [], []
    for r in rows[:n]:
        try:
            with Image.open(root / r["image_path"]) as im:
                ws.append(im.width)
                hs.append(im.height)
        except Exception:  # noqa: BLE001
            pass
    return {"n_sampled": len(ws), "median_w": float(np.median(ws)),
            "median_h": float(np.median(hs)),
            "median_short_side": float(np.median(np.minimum(ws, hs)))}


def run_at(model, manifest_dir, root, lc, size, device, bs=32):
    val = ManifestImageDataset(manifest_dir, root, "val", lc, img_size=size)
    te = ManifestImageDataset(manifest_dir, root, "test", lc, img_size=size)
    lv, yv = logits_labels(model, DataLoader(val, batch_size=bs), device)
    lt, yt = logits_labels(model, DataLoader(te, batch_size=bs), device)
    return lv, yv, lt, yt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--manifest"), ap.add_argument("--root"), ap.add_argument("--device")
    ap.add_argument("--resolutions", default=",".join(map(str, RESOLUTIONS)))
    ap.add_argument("--batch-size", type=int, default=32)
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
    soon = torch.tensor([i for i, c in enumerate(classes) if tier_of(groups, c)[0] == "soon"])
    pig_idx = [ci[c] for c in BENIGN_PIGMENTED if c in ci]

    te_rows = ManifestImageDataset(manifest_dir, root, "test", lc, img_size=224).rows
    frozen = runmeta.freeze(manifest_dir, "test", te_rows)
    meta = runmeta.manifest(tool="resolution_ablation.py", checkpoint=ckpt,
                            files={"groups": str(GROUPS_YAML)},
                            extra={"frozen_test": frozen, "trained_at": ck["img_size"]})
    runmeta.print_header(meta, frozen)

    src = source_image_sizes(root, te_rows)
    print(f"\nSOURCE IMAGES (DermNet): median {src['median_w']:.0f}x{src['median_h']:.0f}, "
          f"short side {src['median_short_side']:.0f}px")
    print(f"  Trained/evaluated at {ck['img_size']}px. Upsampling past the source short "
          f"side adds no\n  real detail — so on THIS corpus the ablation is limited by the "
          f"atlas images themselves.\n  A 12MP phone photo has far more headroom "
          f"(currently 0.41% of pixels survive preprocessing).")

    results = {}
    for size in [int(s) for s in a.resolutions.split(",")]:
        print(f"\n=== {size}px ===")
        lv, yv, lt, yt = run_at(model, manifest_dir, root, lc, size, device, a.batch_size)
        temp = Temperature().fit(lv, yv)
        platt = Platt().fit(temp(lv).softmax(1)[:, soon].sum(1), torch.isin(yv, soon))
        p = temp(lt).softmax(1)
        score = platt(p[:, soon].sum(1))
        pred = p.argmax(1)
        ytn = yt.numpy()

        pig_mask = torch.isin(yt, torch.tensor(pig_idx))
        flag12 = float((score[pig_mask] >= 0.12).float().mean())
        n = len(score)
        ranked = torch.empty(n, dtype=torch.long)
        ranked[score.argsort(descending=True)] = torch.arange(n)
        top10 = ranked < int(round(0.10 * n))
        pig_cap = float(top10[pig_mask].float().mean())

        sk_i, ak_i = ci.get(SK), ci.get(AKBCC)
        sk2ak = int(((yt == sk_i) & (pred == ak_i)).sum())
        ak2sk = int(((yt == ak_i) & (pred == sk_i)).sum())
        top1 = float((pred == yt).float().mean())

        results[size] = {"top1": top1, "benign_pigmented_flag_rate_0.12": flag12,
                         "benign_pigmented_top10_capture": pig_cap,
                         "SK_to_AKBCC": sk2ak, "AKBCC_to_SK": ak2sk,
                         "SK_AKBCC_total_confusion": sk2ak + ak2sk,
                         "temperature": temp.T}
        print(f"  top-1 {top1:.4f}   benign-pigmented flag@0.12 {flag12:.4f}   "
              f"pigmented top-10% capture {pig_cap:.4f}")
        print(f"  SK->AK/BCC {sk2ak}   AK/BCC->SK {ak2sk}   total {sk2ak + ak2sk}")

        if size == int(a.resolutions.split(",")[0]):
            base_pred, base_score, base_yt = pred, score, yt

    print("\n=== RESOLUTION SUMMARY ===")
    print(f"  {'px':>5} {'top1':>7} {'pig flag@.12':>13} {'pig cap@10%':>12} "
          f"{'SK<->AKBCC':>11}")
    for s, r in results.items():
        print(f"  {s:>5} {r['top1']:>7.4f} {r['benign_pigmented_flag_rate_0.12']:>13.4f} "
              f"{r['benign_pigmented_top10_capture']:>12.4f} "
              f"{r['SK_AKBCC_total_confusion']:>11}")
    sizes = sorted(results)
    d_conf = results[sizes[-1]]["SK_AKBCC_total_confusion"] - results[sizes[0]]["SK_AKBCC_total_confusion"]
    d_flag = results[sizes[-1]]["benign_pigmented_flag_rate_0.12"] - results[sizes[0]]["benign_pigmented_flag_rate_0.12"]
    material = d_conf <= -0.15 * max(results[sizes[0]]["SK_AKBCC_total_confusion"], 1) or d_flag <= -0.05
    print(f"\n  {sizes[0]} -> {sizes[-1]}: SK<->AK/BCC confusion {d_conf:+d}, "
          f"pigmented flag rate {d_flag:+.4f}")
    print("  " + ("RESOLUTION-SENSITIVE: the fix is capture/preprocessing (higher res, "
                  "crop-to-lesion,\n  multi-crop), not architecture."
                  if material else
                  "NOT resolution-sensitive on this corpus. Note the caveat above: DermNet's "
                  "own\n  images cap the achievable detail, so this does NOT rule out a gain "
                  "on real phone\n  photos, where 99.6% of the pixels are currently discarded."))

    # ---------------- item 4: where non-pigmented AK/BCC goes ----------------
    print("\n=== NON-PIGMENTED AK/BCC vs HIGH-VOLUME INFLAMMATORY CLASSES ===")
    dL = np.array([darkness_stats(root / r["image_path"])["delta_L"] for r in te_rows])
    cut = float(np.median(dL))
    ak_mask = (base_yt.numpy() == ci[AKBCC])
    npig = ak_mask & (dL < cut)
    pig = ak_mask & (dL >= cut)
    print(f"  AK/BCC: {int(npig.sum())} non-pigmented, {int(pig.sum())} pigmented "
          f"(split at median delta_L {cut:.2f})")

    infl_idx = [ci[c] for c in INFLAMMATORY if c in ci]
    bp = base_pred.numpy()
    to_infl = int(np.isin(bp[npig], infl_idx).sum())
    print(f"  non-pigmented AK/BCC predicted as an inflammatory class: "
          f"{to_infl}/{int(npig.sum())} = {to_infl / max(npig.sum(), 1):.1%}")
    for c in INFLAMMATORY:
        if c not in ci:
            continue
        k = int((bp[npig] == ci[c]).sum())
        rev = int(((base_yt.numpy() == ci[c]) & (bp == ci[AKBCC])).sum())
        print(f"    -> {c[:46]:<46} {k:>3}   (reverse {c[:20]}->AK/BCC: {rev})")
    sc = base_score.numpy()
    print(f"  urgent score: non-pigmented AK/BCC mean {sc[npig].mean():.3f} "
          f"vs pigmented {sc[pig].mean():.3f}")
    for c in INFLAMMATORY:
        if c in ci:
            m = base_yt.numpy() == ci[c]
            print(f"    {c[:46]:<46} mean urgent {sc[m].mean():.3f}  n={int(m.sum())}")

    out = {**meta, "source_image_sizes": src, "by_resolution": results,
           "resolution_sensitive": bool(material),
           "non_pigmented_akbcc": {
               "n_non_pigmented": int(npig.sum()), "n_pigmented": int(pig.sum()),
               "predicted_as_inflammatory": to_infl,
               "frac_predicted_as_inflammatory": to_infl / max(int(npig.sum()), 1),
               "mean_urgent_non_pigmented": float(sc[npig].mean()),
               "mean_urgent_pigmented": float(sc[pig].mean())},
           "caveat": "DermNet source images cap achievable resolution; a null result here "
                     "does not rule out a gain on full-resolution phone photos."}
    p = ckpt.parent / "resolution_ablation.json"
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
