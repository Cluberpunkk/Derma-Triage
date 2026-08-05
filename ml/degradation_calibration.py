"""ml/degradation_calibration.py
=========================================================================
Sets the quality thresholds from MEASURED HARM instead of guessed constants.

The problem with calibrating a quality gate: atlas photos are all sharp, so DermNet
shows a 0.000 blur-rejection rate at any sensible threshold and tells us nothing.
And we cannot collect blurry real captures through a gate we have not calibrated.

Synthetic degradation breaks that circularity. Blur and mis-expose known-good images
by known amounts, and measure two things at each level:

  VoL(sigma)         what the metric reads
  accuracy(sigma)    what it costs the model

The threshold then answers a real question — "at what VoL does classification start
to degrade?" — instead of being an inherited constant (55, from nowhere in
particular).

THE TONE TEST THAT CAME BACK NaN
--------------------------------
Variance-of-Laplacian is a contrast-dependent statistic: the same optical blur
produces a LOWER VoL on a lower-contrast image. If darker-skinned captures have
lower local contrast, an absolute VoL threshold rejects them more often at identical
true sharpness — telling those users to retake photos that were fine.

On undegraded DermNet this was untestable (zero rejections in every bin, so the gap
was 0.000 by construction and the correlation was NaN). Synthetic degradation makes
it testable TODAY: apply the SAME sigma across ITA bins and compare the VoL each bin
reports. A spread means the threshold must be contrast-relative.

    ml/.venv/bin/python ml/degradation_calibration.py data/ml/runs_final/best.pt
=========================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import runmeta  # noqa: E402
from dataset import ManifestImageDataset, build_transforms  # noqa: E402
from ita_audit import ITA_BINS, boot_ci, ita_bin, ita_from_ring  # noqa: E402
from shortcut_test import _lab  # noqa: E402
from train import load_checkpoint  # noqa: E402

BLUR_SIGMAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
EXPOSURE_GAINS = [0.25, 0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2]
ACCEPTABLE_ACC_DROP = 0.05      # threshold = where accuracy falls this far below baseline


def vol(img: Image.Image) -> float:
    g = np.asarray(img.convert("L").resize((200, 200)), dtype=np.float32)
    lap = (-4 * g + np.roll(g, 1, 0) + np.roll(g, -1, 0)
           + np.roll(g, 1, 1) + np.roll(g, -1, 1))[1:-1, 1:-1]
    return float(lap.var())


def luma(img: Image.Image) -> float:
    return float(np.asarray(img.convert("L").resize((200, 200)), dtype=np.float32).mean())


def local_contrast(img: Image.Image) -> float:
    return float(np.asarray(img.convert("L").resize((200, 200)), dtype=np.float32).std())


def expose(img: Image.Image, gain: float) -> Image.Image:
    a = np.asarray(img, dtype=np.float32) * gain
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


@torch.no_grad()
def correct_flags(model, imgs, labels, tf, device, bs=32) -> np.ndarray:
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.stack([tf(im) for im in imgs[i:i + bs]]).to(device)
        out.append((model(x).argmax(1).cpu() == labels[i:i + bs]).numpy())
    return np.concatenate(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--manifest"), ap.add_argument("--root"), ap.add_argument("--device")
    ap.add_argument("--n-per-bin", type=int, default=110)
    a = ap.parse_args()

    ckpt = Path(a.checkpoint).resolve()
    device = config.resolve_device(a.device)
    model, ck = load_checkpoint(ckpt, device)
    classes = ck["classes"]
    ci = {c: i for i, c in enumerate(classes)}
    manifest_dir = Path(a.manifest) if a.manifest else config.out_dir() / "dermnet"
    root = config.dataset_root(a.root)
    tf = build_transforms(ck["img_size"], train=False)

    te = ManifestImageDataset(manifest_dir, root, "test", "original_class", img_size=224)
    frozen = runmeta.freeze(manifest_dir, "test", te.rows)
    meta = runmeta.manifest(tool="degradation_calibration.py", checkpoint=ckpt,
                            extra={"frozen_test": frozen, "blur_sigmas": BLUR_SIGMAS,
                                   "exposure_gains": EXPOSURE_GAINS,
                                   "acceptable_acc_drop": ACCEPTABLE_ACC_DROP})
    runmeta.print_header(meta, frozen)

    # stratified sample so darker bins are actually represented
    print(f"\nsampling up to {a.n_per_bin} per ITA bin…")
    picked, per_bin = [], {b[2]: 0 for b in ITA_BINS}
    rng = np.random.default_rng(0)
    for i in rng.permutation(len(te.rows)):
        r = te.rows[i]
        img = Image.open(root / r["image_path"]).convert("RGB")
        small = np.asarray(img.resize((224, 224)), dtype=np.float32) / 255.0
        b = ita_bin(ita_from_ring(_lab(small)))
        if b in per_bin and per_bin[b] < a.n_per_bin:
            per_bin[b] += 1
            picked.append({"img": img.resize((512, 512)), "bin": b,
                           "y": ci[r["original_class"]],
                           "contrast": local_contrast(img)})
        if all(v >= a.n_per_bin for v in per_bin.values()):
            break
    print("  " + "  ".join(f"{k}={v}" for k, v in per_bin.items()))

    labels = torch.tensor([p["y"] for p in picked])
    bins = np.array([p["bin"] for p in picked])
    order = [b[2] for b in ITA_BINS if per_bin[b[2]] >= 20]

    # ---------------- blur ----------------
    print(f"\n=== BLUR: VoL and accuracy vs sigma ===")
    print(f"  {'sigma':>6} {'acc':>7} {'dAcc':>7} {'VoL med':>9}   " +
          "  ".join(f"{b[:9]:>10}" for b in order))
    blur_rows, base_acc = [], None
    for s in BLUR_SIGMAS:
        imgs = [p["img"] if s == 0 else p["img"].filter(ImageFilter.GaussianBlur(s))
                for p in picked]
        acc = float(correct_flags(model, imgs, labels, tf, device).mean())
        if base_acc is None:
            base_acc = acc
        vs = np.array([vol(im) for im in imgs])
        per = {b: float(np.median(vs[bins == b])) for b in order}
        blur_rows.append({"sigma": s, "accuracy": acc, "acc_drop": base_acc - acc,
                          "vol_median": float(np.median(vs)), "vol_by_ita": per,
                          "vol_spread_across_bins": max(per.values()) - min(per.values())})
        print(f"  {s:>6.1f} {acc:>7.3f} {base_acc - acc:>7.3f} {np.median(vs):>9.1f}   " +
              "  ".join(f"{per[b]:>10.1f}" for b in order))

    hurt = next((r for r in blur_rows if r["acc_drop"] >= ACCEPTABLE_ACC_DROP), None)
    if hurt:
        thr = hurt["vol_median"]
        print(f"\n  accuracy first drops {ACCEPTABLE_ACC_DROP:.0%} at sigma={hurt['sigma']}, "
              f"where median VoL = {thr:.1f}")
        print(f"  -> MEASURED blur threshold ~{thr:.0f}  (current inherited constant: 55)")
    else:
        thr = None
        print(f"\n  accuracy never drops {ACCEPTABLE_ACC_DROP:.0%} across the tested range — "
              f"blur matters less than assumed")

    # the tone test that was NaN on undegraded images
    print(f"\n=== TONE-DEPENDENCE OF VoL AT IDENTICAL BLUR ===")
    print("  Same sigma applied to every bin. Spread => an absolute threshold is unfair.")
    worst = max(blur_rows[1:], key=lambda r: r["vol_spread_across_bins"])
    for r in blur_rows:
        lo = min(r["vol_by_ita"].values())
        hi = max(r["vol_by_ita"].values())
        ratio = hi / lo if lo > 0 else float("inf")
        print(f"  sigma {r['sigma']:>4.1f}:  VoL {lo:>7.1f} - {hi:>7.1f}   ratio {ratio:>5.2f}x")
    lo_b, hi_b = order[0], order[-1]
    rho = float(np.corrcoef([p["contrast"] for p in picked],
                            [vol(p["img"]) for p in picked])[0, 1])
    print(f"  corr(local_contrast, VoL) = {rho:+.3f}")
    if thr is not None:
        below = {b: float(np.mean([vol(p["img"].filter(ImageFilter.GaussianBlur(1.0)))
                                   < thr for p in picked if p["bin"] == b])) for b in order}
        print(f"\n  At sigma=1.0 (mild, identical blur), fraction falling below the "
              f"measured threshold {thr:.0f}:")
        for b in order:
            print(f"    {b:<13} {below[b]:.3f}")
        spread = max(below.values()) - min(below.values())
        print(f"  spread {spread:.3f} -> " +
              ("THRESHOLD IS TONE-DEPENDENT. Use a contrast-relative form "
               "(VoL / local_contrast^2)." if spread > 0.10 else
               "no material tone dependence at this threshold."))
    else:
        below, spread = {}, None

    # ---------------- exposure ----------------
    print(f"\n=== EXPOSURE: luma and accuracy vs gain ===")
    print(f"  {'gain':>6} {'acc':>7} {'dAcc':>7} {'luma med':>9}")
    exp_rows = []
    for gmul in EXPOSURE_GAINS:
        imgs = [expose(p["img"], gmul) for p in picked]
        acc = float(correct_flags(model, imgs, labels, tf, device).mean())
        lu = float(np.median([luma(im) for im in imgs]))
        exp_rows.append({"gain": gmul, "accuracy": acc,
                         "acc_drop": blur_rows[0]["accuracy"] - acc, "luma_median": lu})
        print(f"  {gmul:>6.2f} {acc:>7.3f} {blur_rows[0]['accuracy'] - acc:>7.3f} {lu:>9.1f}")
    dark = [r for r in exp_rows if r["gain"] < 1 and r["acc_drop"] >= ACCEPTABLE_ACC_DROP]
    bright = [r for r in exp_rows if r["gain"] > 1 and r["acc_drop"] >= ACCEPTABLE_ACC_DROP]
    lo_lum = max((r["luma_median"] for r in dark), default=None)
    hi_lum = min((r["luma_median"] for r in bright), default=None)
    print(f"\n  MEASURED exposure band: luma "
          f"{lo_lum if lo_lum is not None else '(no lower bound found)'} .. "
          f"{hi_lum if hi_lum is not None else '(no upper bound found)'}")
    print(f"  (current inherited constants: 42 .. 222)")

    out = {**meta, "blur": blur_rows, "measured_blur_threshold": thr,
           "inherited_blur_threshold": 55,
           "tone_dependence": {"below_threshold_by_bin": below, "spread": spread,
                               "corr_contrast_vol": rho,
                               "max_vol_spread_row": worst},
           "exposure": exp_rows, "measured_luma_low": lo_lum, "measured_luma_high": hi_lum,
           "inherited_luma_band": [42, 222],
           "caveat": "Synthetic degradation of atlas photos. Real capture blur is optical "
                     "and motion blur, not Gaussian, and real exposure failures include "
                     "colour casts these gains do not model. Treat as a principled "
                     "STARTING POINT, to be replaced by the pilot distribution."}
    p = ckpt.parent / "degradation_calibration.json"
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
