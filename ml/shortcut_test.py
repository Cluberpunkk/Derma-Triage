"""ml/shortcut_test.py
=========================================================================
Tests one hypothesis: has the backbone learned PIGMENTATION as a proxy for
malignancy?

If true it explains three findings at once, which is why it is worth testing before
anything downstream:

  Blocker 1   71% of benign pigmented lesions flagged urgent — they are dark
  melanoma folder is easy   capture 0.526 at top-10%, double everything else —
                            melanoma is the darkest urgent class
  AK/BCC captures poorly    0.108 at top-5%, the worst — AK/BCC is typically
                            erythematous/scaly, NOT pigmented

A model that ranked by "how dark is the lesion" would produce exactly this pattern.

WHAT IS MEASURED
----------------
1. Lesion darkness/saturation in CIELAB, lesion region vs surrounding skin.
   No segmentation masks exist, so the lesion proxy is the central disc and the
   skin proxy is the outer ring — DermNet images are lesion-centred. This is a
   proxy and is stated as one; it is noisy per-image but unbiased in aggregate.
2. Correlation with the calibrated urgent score — overall, and WITHIN each class
   group. The within-group correlations are the load-bearing ones: an overall
   correlation could simply reflect that urgent classes happen to be darker,
   whereas a correlation inside the routine group alone means darkness moves the
   score regardless of disease.
3. Top-10% capture for pigmented vs non-pigmented halves of each urgent class.
   If the shortcut is real, the pigmented half of every class captures better.
4. Grad-CAM, quantified rather than eyeballed: correlation between the saliency map
   and the darkness map. "Is it attending to morphology, or to the dark region?"
   becomes a number.

VERDICT RULE
------------
Strong within-routine correlation + a consistent pigmented/non-pigmented capture gap
means the shortcut is real, the downstream heads cannot fix it (they consume the
same features), and the priority becomes a dermatology-pretrained backbone.

    ml/.venv/bin/python ml/shortcut_test.py data/ml/runs_final/best.pt
=========================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import runmeta  # noqa: E402
from dataset import ManifestImageDataset, build_transforms  # noqa: E402
from pigmentation_audit import BENIGN_PIGMENTED  # noqa: E402
from train import load_checkpoint  # noqa: E402
from urgency import (GROUPS_YAML, Platt, Temperature, load_groups, logits_labels,  # noqa: E402
                     tier_of)

SIZE = 224
R_LESION = 0.34        # central disc radius (fraction of half-width)
R_SKIN = 0.46          # outer ring starts here


def _lab(arr: np.ndarray) -> np.ndarray:
    """sRGB [0,1] -> CIELAB. Small, dependency-free, adequate for relative stats."""
    m = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    lin = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def darkness_stats(path: Path) -> dict:
    """Lesion-vs-surround lightness and chroma. Positive delta_L = lesion darker."""
    im = Image.open(path).convert("RGB").resize((SIZE, SIZE))
    lab = _lab(np.asarray(im, dtype=np.float32) / 255.0)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    r = np.sqrt((xx - SIZE / 2) ** 2 + (yy - SIZE / 2) ** 2) / (SIZE / 2)
    les, skin = r < R_LESION, r > R_SKIN
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = np.sqrt(a ** 2 + b ** 2)
    return {
        "lesion_L": float(L[les].mean()), "skin_L": float(L[skin].mean()),
        "delta_L": float(L[skin].mean() - L[les].mean()),
        "lesion_chroma": float(chroma[les].mean()),
        "delta_chroma": float(chroma[les].mean() - chroma[skin].mean()),
        "lesion_L_std": float(L[les].std()),
    }


def darkness_map(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB").resize((SIZE, SIZE))
    L = _lab(np.asarray(im, dtype=np.float32) / 255.0)[..., 0]
    d = (L.max() - L)
    return d / (d.max() + 1e-8)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d else float("nan")


def pearson(x, y):
    x, y = x - x.mean(), y - y.mean()
    d = np.sqrt((x ** 2).sum() * (y ** 2).sum())
    return float((x * y).sum() / d) if d else float("nan")


# ----------------------------------------------------------------- Grad-CAM

def last_conv(model):
    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    return convs[-1]


def gradcam(model, x, target_logits_idx, layer):
    feats, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: feats.__setitem__("v", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("v", go[0]))
    try:
        model.zero_grad(set_to_none=True)
        out = model(x)
        out[0, target_logits_idx].sum().backward()
    finally:
        h1.remove()
        h2.remove()
    w = grads["v"].mean(dim=(2, 3), keepdim=True)
    cam = (w * feats["v"]).sum(1).relu()[0].detach().cpu().numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((SIZE, SIZE))) / 255.0


def cam_sheet(entries, title, out_path, root, model, layer, soon_idx, tf, device):
    cols, cell = 5, 150
    rows = (len(entries) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (cell * 2 + 8) + 8, rows * (cell + 34) + 26), "white")
    d = ImageDraw.Draw(sheet)
    d.text((6, 6), title, fill="black")
    corrs = []
    for i, e in enumerate(entries):
        p = root / e["image_path"]
        try:
            img = Image.open(p).convert("RGB").resize((cell, cell))
            x = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            cam = gradcam(model, x, soon_idx, layer)
            dm = darkness_map(p)
            c = pearson(cam.ravel(), dm.ravel())
            corrs.append(c)
            heat = Image.fromarray((np.stack([cam, np.zeros_like(cam),
                                              1 - cam], -1) * 255).astype(np.uint8)
                                   ).resize((cell, cell))
            blend = Image.blend(img, heat, 0.45)
            cx = 8 + (i % cols) * (cell * 2 + 8)
            cy = 26 + (i // cols) * (cell + 34)
            sheet.paste(img, (cx, cy))
            sheet.paste(blend, (cx + cell + 2, cy))
            d.text((cx, cy + cell + 2), f"cam~dark r={c:+.2f}", fill="black")
            d.text((cx, cy + cell + 14), e["true"][:34], fill="darkred")
        except Exception:  # noqa: BLE001
            continue
    sheet.save(out_path)
    return corrs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--manifest"), ap.add_argument("--root"), ap.add_argument("--device")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-cam", type=int, default=20)
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
    meta = runmeta.manifest(tool="shortcut_test.py", checkpoint=ckpt,
                            files={"groups": str(GROUPS_YAML)}, extra={"frozen_test": frozen})
    runmeta.print_header(meta, frozen)

    lv, yv = logits_labels(model, DataLoader(val, batch_size=a.batch_size), device)
    lt, yt = logits_labels(model, DataLoader(te, batch_size=a.batch_size), device)
    temp = Temperature().fit(lv, yv)
    soon = torch.tensor([i for i, c in enumerate(classes) if tier_of(groups, c)[0] == "soon"])
    platt = Platt().fit(temp(lv).softmax(1)[:, soon].sum(1), torch.isin(yv, soon))
    score = platt(temp(lt).softmax(1)[:, soon].sum(1)).numpy()
    is_pos = torch.isin(yt, soon).numpy()

    print(f"\ncomputing LAB lesion/skin stats for {len(te.rows)} test images "
          f"(centre disc = lesion proxy, outer ring = skin proxy)…")
    stats = []
    for i, r in enumerate(te.rows):
        stats.append(darkness_stats(root / r["image_path"]))
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(te.rows)}")
    dL = np.array([s["delta_L"] for s in stats])
    dC = np.array([s["delta_chroma"] for s in stats])

    # ---- item 1: correlations ----
    print("\n=== CORRELATION: lesion darkness vs calibrated urgent score ===")
    print("  delta_L = surrounding skin L minus lesion L (positive = lesion darker)")
    groups_corr = {}
    for name, m in (("overall", np.ones_like(is_pos, dtype=bool)),
                    ("urgent only", is_pos), ("routine only", ~is_pos)):
        sp = spearman(dL[m], score[m])
        pe = pearson(dL[m], score[m])
        spc = spearman(dC[m], score[m])
        groups_corr[name] = {"n": int(m.sum()), "spearman_delta_L": sp,
                             "pearson_delta_L": pe, "spearman_delta_chroma": spc}
        print(f"  {name:<14} n={int(m.sum()):<5} rho(delta_L)={sp:+.3f}  "
              f"r={pe:+.3f}   rho(chroma)={spc:+.3f}")
    print("  The 'routine only' row is load-bearing: a correlation there means darkness")
    print("  moves the urgent score WITHIN non-urgent disease, i.e. regardless of class.")

    # ---- item 1: pigmented vs non-pigmented halves of each urgent class ----
    cut = float(np.median(dL))
    n = len(score)
    order = np.argsort(-score)
    ranked = np.zeros(n, dtype=int)
    ranked[order] = np.arange(n)
    top10 = ranked < int(round(0.10 * n))
    print(f"\n=== TOP-10% CAPTURE, pigmented vs non-pigmented halves "
          f"(split at median delta_L = {cut:.2f}) ===")
    print(f"  {'class':<50} {'pigmented':>22} {'non-pigmented':>22} {'gap':>7}")
    halves = []
    yt_np = yt.numpy()
    for i in soon.tolist():
        m = yt_np == i
        if m.sum() < 20:
            continue
        pig, non = m & (dL >= cut), m & (dL < cut)
        if pig.sum() < 5 or non.sum() < 5:
            continue
        cp = top10[pig].mean()
        cn = top10[non].mean()
        halves.append({"class": classes[i], "n_pigmented": int(pig.sum()),
                       "n_non_pigmented": int(non.sum()),
                       "capture_pigmented": float(cp), "capture_non_pigmented": float(cn),
                       "gap": float(cp - cn)})
        print(f"  {classes[i][:50]:<50} {cp:>10.3f} (n={int(pig.sum()):<4}) "
              f"{cn:>10.3f} (n={int(non.sum()):<4}) {cp - cn:>+7.3f}")
    if halves:
        mean_gap = float(np.mean([h["gap"] for h in halves]))
        n_pos_gap = sum(1 for h in halves if h["gap"] > 0)
        print(f"\n  mean gap {mean_gap:+.3f}; pigmented half captures better in "
              f"{n_pos_gap}/{len(halves)} classes")

    # ---- item 1: Grad-CAM, quantified ----
    layer = last_conv(model)
    tf = build_transforms(ck["img_size"], train=False)
    correct_urgent = [i for i in range(n) if is_pos[i] and score[i] >= 0.5][:a.n_cam]
    pig_idx = [classes.index(c) for c in BENIGN_PIGMENTED if c in classes]
    fa = [i for i in range(n) if yt_np[i] in pig_idx and score[i] >= 0.5]
    fa = sorted(fa, key=lambda i: -score[i])[:a.n_cam]

    out_dir = ckpt.parent / "shortcut"
    out_dir.mkdir(exist_ok=True)
    ent = lambda ids: [{"image_path": te.rows[i]["image_path"],  # noqa: E731
                        "true": classes[int(yt_np[i])]} for i in ids]
    print(f"\n=== GRAD-CAM: does saliency track morphology or the dark region? ===")
    c1 = cam_sheet(ent(correct_urgent), "Correct urgent predictions",
                   out_dir / "cam_correct_urgent.png", root, model, layer, soon, tf, device)
    c2 = cam_sheet(ent(fa), "Benign-pigmented FALSE ALARMS",
                   out_dir / "cam_pigmented_false_alarms.png", root, model, layer, soon,
                   tf, device)
    print(f"  correct urgent      n={len(c1):<3} mean corr(CAM, darkness) = "
          f"{np.mean(c1):+.3f}")
    print(f"  pigmented false alarms n={len(c2):<3} mean corr(CAM, darkness) = "
          f"{np.mean(c2):+.3f}")
    print(f"  sheets in {out_dir}")

    # ---- verdict ----
    rho_routine = groups_corr["routine only"]["spearman_delta_L"]
    gap_ok = halves and mean_gap > 0.03 and n_pos_gap >= 0.7 * len(halves)
    strong = abs(rho_routine) >= 0.25 and gap_ok
    if strong:
        verdict = (
            f"SHORTCUT CONFIRMED. Within routine disease alone, darkness correlates with "
            f"the urgent score at rho={rho_routine:+.3f}, and the pigmented half of urgent "
            f"classes captures {mean_gap:+.3f} better in {n_pos_gap}/{len(halves)} classes. "
            f"The backbone is using pigmentation as a proxy for malignancy. Downstream heads "
            f"consume these same features and CANNOT fix this. Priority becomes a "
            f"dermatology-pretrained backbone, not any head, threshold or calibration work.")
    elif abs(rho_routine) >= 0.15 or gap_ok:
        verdict = (
            f"PARTIAL. rho(routine)={rho_routine:+.3f}, mean pigmented capture gap "
            f"{mean_gap:+.3f} ({n_pos_gap}/{len(halves)} classes). Darkness contributes but "
            f"does not obviously dominate. Heads are worth running, with pigmented-half "
            f"capture tracked as a success criterion.")
    else:
        verdict = (
            f"NOT SUPPORTED. rho(routine)={rho_routine:+.3f} and the pigmented/non-pigmented "
            f"capture gap is {mean_gap:+.3f}. Darkness does not explain the blocker; the "
            f"pigmentation false alarms have another cause and the heads remain worth running.")
    print("\n=== VERDICT ===\n  " + verdict.replace(". ", ".\n  "))

    out = {**meta, "correlations": groups_corr, "delta_L_median": cut,
           "pigmented_vs_non_capture": halves,
           "mean_capture_gap": mean_gap if halves else None,
           "gradcam_darkness_corr": {"correct_urgent_mean": float(np.mean(c1)) if c1 else None,
                                     "pigmented_false_alarm_mean": float(np.mean(c2)) if c2 else None},
           "shortcut_confirmed": bool(strong), "verdict": verdict,
           "method_caveat": "Lesion region is a centre-disc proxy, not a segmentation. "
                            "Noisy per image, unbiased in aggregate."}
    p = ckpt.parent / "shortcut_test.json"
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
