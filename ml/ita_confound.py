"""ml/ita_confound.py
=========================================================================
Is ITA measuring the PATIENT's skin tone, or the LESION's darkness?

`ml/ita_audit.py` samples ITA from a fixed perilesional annulus (r > 0.46). On a
tightly-framed close-up the lesion can reach into that annulus, and a dark lesion
would then drag ITA down — so the "dark skin" bin could actually be a "dark lesion"
bin, and the whole fairness finding would be a lesion-darkness finding wearing the
wrong label.

Three checks, per your spec:

  (a) class composition per ITA bin. If the dark bin is dominated by melanoma,
      pigmentation and vascular classes rather than spread across diseases, ITA is
      reading the lesion.
  (b) corr(ITA, lesion-area fraction). A strong negative correlation means bigger
      lesions produce darker apparent tone.
  (c) tone gradient restricted to CLEAN samples — images where the localised lesion
      box barely intersects the sampling annulus. If the gradient collapses there,
      relabel the finding.

Second half: ITA-STRATIFIED QUALITY GATE. The gate is the new gatekeeper for
collection, and a tone-dependent rejection rate silently excludes the users we most
need data from. Blur detection via gradient energy is contrast-dependent by
construction, so a fixed absolute threshold should be expected to reject darker
images more often. Each criterion is reported separately, with CIs, so a
contrast-relative threshold can be justified if the data demands it.

    ml/.venv/bin/python ml/ita_confound.py data/ml/runs_final/best.pt
=========================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import runmeta  # noqa: E402
from dataset import ManifestImageDataset  # noqa: E402
from ita_audit import ITA_BINS, boot_ci, find_lesion, gap_ci, ita_bin  # noqa: E402
from shortcut_test import _lab  # noqa: E402

# Mirrors src/lib/cameraQuality.js THRESHOLDS
Q = {"sharpness": 55.0, "brightness_low": 42.0, "brightness_high": 222.0, "fill": 0.40}
ANNULUS_R = 0.46
CLEAN_MAX_OVERLAP = 0.05     # lesion box may cover at most this share of the annulus


def analyse(path: Path) -> dict:
    img = Image.open(path).convert("RGB")
    small = img.resize((224, 224))
    arr = np.asarray(small, dtype=np.float32) / 255.0
    lab = _lab(arr)
    L = lab[..., 0]
    h, w = L.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (min(h, w) / 2)
    annulus = r > ANNULUS_R

    def ita_at(rmin):
        m = r > rmin
        if m.sum() < 20:
            return float("nan")
        b = float(np.median(lab[..., 2][m]))
        if abs(b) < 1e-6:
            return float("nan")
        return float(np.degrees(np.arctan((float(np.median(L[m])) - 50) / b)))

    ita = ita_at(ANNULUS_R)
    # A FAR more peripheral ring: much less likely to contain lesion, and — unlike the
    # "clean sample" filter — NOT selected on the localiser's own output, so it is not
    # circular. If the tone gradient survives here, ITA is not merely lesion darkness.
    ita_far = ita_at(0.80)

    loc = find_lesion(img, "edge")
    # lesion box in 224-space, to measure how much of the annulus it eats
    sf = loc.get("side_frac", 1.0)
    side = sf * min(h, w)
    box = (abs(xx - w / 2) <= side / 2) & (abs(yy - h / 2) <= side / 2)
    overlap = float((box & annulus).sum() / max(annulus.sum(), 1))

    g = np.asarray(small.convert("L"), dtype=np.float32)
    lap = (-4 * g + np.roll(g, 1, 0) + np.roll(g, -1, 0)
           + np.roll(g, 1, 1) + np.roll(g, -1, 1))[1:-1, 1:-1]
    gx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
    gy = np.abs(np.diff(g, axis=0, prepend=g[:1, :]))
    e = gx + gy
    centre = r < 0.5
    fill = float(e[centre].sum() / (e.sum() + 1e-8))

    return {"ita": ita, "ita_far": ita_far, "annulus_overlap": overlap,
            "lesion_area_frac": float(loc.get("side_frac", 1.0) ** 2),
            "fallback": bool(loc["fallback"]),
            "sharpness": float(lap.var()), "brightness": float(g.mean()),
            "fill": fill, "local_contrast": float(g.std())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--manifest"), ap.add_argument("--root")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    ckpt = Path(a.checkpoint).resolve()
    manifest_dir = Path(a.manifest) if a.manifest else config.out_dir() / "dermnet"
    root = config.dataset_root(a.root)
    te = ManifestImageDataset(manifest_dir, root, "test", "original_class", img_size=224)
    frozen = runmeta.freeze(manifest_dir, "test", te.rows)
    meta = runmeta.manifest(tool="ita_confound.py", checkpoint=ckpt,
                            extra={"frozen_test": frozen, "ita_is_a_proxy": True})
    runmeta.print_header(meta, frozen)

    rows = te.rows[:a.limit] if a.limit else te.rows
    print(f"\nanalysing {len(rows)} images…")
    recs = []
    for i, r in enumerate(rows):
        d = analyse(root / r["image_path"])
        d["cls"] = r["original_class"]
        recs.append(d)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(rows)}")

    ita = np.array([d["ita"] for d in recs])
    bins = np.array([ita_bin(v) for v in ita])
    overlap = np.array([d["annulus_overlap"] for d in recs])
    area = np.array([d["lesion_area_frac"] for d in recs])
    fb = np.array([d["fallback"] for d in recs], dtype=float)
    order = [b[2] for b in ITA_BINS]

    # ---------- (a) class composition ----------
    print("\n=== (a) CLASS COMPOSITION PER ITA BIN — is the dark bin disease-skewed? ===")
    comp = {}
    for b in order:
        m = bins == b
        if m.sum() < 25:
            continue
        cnt = Counter([recs[i]["cls"] for i in np.nonzero(m)[0]])
        top = cnt.most_common(3)
        comp[b] = {"n": int(m.sum()),
                   "top3": [{"cls": c, "n": k, "share": k / int(m.sum())} for c, k in top],
                   "n_distinct_classes": len(cnt)}
        share3 = sum(k for _, k in top) / int(m.sum())
        print(f"  {b:<13} n={int(m.sum()):<5} distinct={len(cnt):<3} top3={share3:.1%}  "
              + "; ".join(f"{c[:26]} {k / int(m.sum()):.0%}" for c, k in top))
    print("  A tone bin should look like the whole corpus. Heavy skew toward pigmented/")
    print("  vascular classes in the dark bin would mean ITA is reading the lesion.")

    # ---------- (b) ITA vs lesion area ----------
    ok = np.isfinite(ita)
    rho_area = float(np.corrcoef(ita[ok], area[ok])[0, 1])
    rho_ovl = float(np.corrcoef(ita[ok], overlap[ok])[0, 1])
    print(f"\n=== (b) ITA vs LESION SIZE ===")
    print(f"  corr(ITA, lesion_area_frac)  = {rho_area:+.3f}")
    print(f"  corr(ITA, annulus_overlap)   = {rho_ovl:+.3f}")
    print("  Strong NEGATIVE values would mean bigger lesions produce darker apparent tone.")

    # ---------- (c) clean-sample gradient ----------
    clean = ok & (overlap <= CLEAN_MAX_OVERLAP)
    print(f"\n=== (c) TONE GRADIENT ON CLEAN PERILESIONAL SAMPLES ===")
    print(f"  clean = lesion box covers <= {CLEAN_MAX_OVERLAP:.0%} of the annulus: "
          f"{int(clean.sum())}/{len(recs)} images")
    lo_b, hi_b = order[0], order[-1]
    res = {}
    for tag, m in (("all", ok), ("clean only", clean)):
        l = fb[m & (bins == lo_b)]
        d = fb[m & (bins == hi_b)]
        if len(l) < 20 or len(d) < 20:
            print(f"  {tag:<12} too few per bin (n={len(l)}, {len(d)})")
            continue
        gap = float(d.mean() - l.mean())
        ci = gap_ci(l, d)
        res[tag] = {"n_light": len(l), "n_dark": len(d), "fallback_light": float(l.mean()),
                    "fallback_dark": float(d.mean()), "gap": gap, "ci95": ci,
                    "includes_zero": bool(ci[0] <= 0 <= ci[1])}
        print(f"  {tag:<12} {lo_b} {l.mean():.3f} (n={len(l)}) -> {hi_b} {d.mean():.3f} "
              f"(n={len(d)})  gap {gap:+.4f} CI [{ci[0]:+.4f},{ci[1]:+.4f}]"
              f"{'  INCLUDES ZERO' if res[tag]['includes_zero'] else ''}")

    # --- non-circular replacement for (c): peripheral-annulus ITA ---
    ita_far = np.array([d["ita_far"] for d in recs])
    bins_far = np.array([ita_bin(v) for v in ita_far])
    okf = np.isfinite(ita_far)
    lf = fb[okf & (bins_far == lo_b)]
    df = fb[okf & (bins_far == hi_b)]
    print(f"\n  NON-CIRCULAR CHECK — ITA from a peripheral annulus (r>0.80):")
    print(f"    corr(ITA_r0.46, ITA_r0.80) = "
          f"{float(np.corrcoef(ita[ok & okf], ita_far[ok & okf])[0, 1]):+.3f}")
    if len(lf) >= 20 and len(df) >= 20:
        gapf = float(df.mean() - lf.mean())
        cif = gap_ci(lf, df)
        res["peripheral r>0.80"] = {"n_light": len(lf), "n_dark": len(df),
                                    "fallback_light": float(lf.mean()),
                                    "fallback_dark": float(df.mean()),
                                    "gap": gapf, "ci95": cif,
                                    "includes_zero": bool(cif[0] <= 0 <= cif[1])}
        print(f"    {lo_b} {lf.mean():.3f} (n={len(lf)}) -> {hi_b} {df.mean():.3f} "
              f"(n={len(df)})  gap {gapf:+.4f} CI [{cif[0]:+.4f},{cif[1]:+.4f}]"
              f"{'  INCLUDES ZERO' if res['peripheral r>0.80']['includes_zero'] else '  EXCLUDES ZERO'}")

    # NOTE: the "clean only" row above is CIRCULAR and must not be read as evidence.
    # A fallback sets the crop box to the whole frame, which covers ~100% of the
    # annulus, so fallback cases can never satisfy the <=5% overlap filter. Its
    # fallback rate is 0 by construction in both bins. Kept in the output only so the
    # error is visible rather than silently removed.
    res["clean only"]["CIRCULAR_DO_NOT_USE"] = True

    per = res.get("peripheral r>0.80")
    collapsed = bool(per and per["includes_zero"])
    v1 = ("LESION CONTAMINATION CONFIRMED. The tone gradient disappears once the "
          "perilesional annulus is clear of the lesion. The ITA finding is a "
          "LESION-DARKNESS finding and must be relabelled in the record."
          if collapsed else
          "GRADIENT SURVIVES the clean-sample restriction. ITA is not simply reading "
          "lesion darkness — though it remains a proxy, and the holdout should sample "
          "outside the user-placed reticle rather than a heuristic annulus.")
    print("\n  " + v1.replace(". ", ".\n  "))

    # ---------- (2) ITA-stratified quality gate ----------
    print(f"\n=== ITA-STRATIFIED QUALITY GATE (per criterion, with CIs) ===")
    sharp = np.array([d["sharpness"] for d in recs])
    bright = np.array([d["brightness"] for d in recs])
    fill = np.array([d["fill"] for d in recs])
    contrast = np.array([d["local_contrast"] for d in recs])
    crit = {"blur": sharp < Q["sharpness"],
            "too dark": bright < Q["brightness_low"],
            "too bright": bright > Q["brightness_high"],
            "poor fill": fill < Q["fill"]}
    crit["ANY"] = np.any(np.stack(list(crit.values())), axis=0)

    print(f"  {'criterion':<12} " + " ".join(f"{b[:9]:>18}" for b in order))
    gate = {}
    for name, mask in crit.items():
        cells, gate[name] = [], {}
        for b in order:
            m = bins == b
            if m.sum() < 25:
                cells.append(f"{'-':>18}")
                continue
            v = mask[m].astype(float)
            ci = boot_ci(v)
            gate[name][b] = {"n": int(m.sum()), "rate": float(v.mean()), "ci95": ci}
            cells.append(f"{v.mean():.3f}[{ci[0]:.2f}-{ci[1]:.2f}]".rjust(18))
        print(f"  {name:<12} " + " ".join(cells))

    blur_gap = None
    if lo_b in gate["blur"] and hi_b in gate["blur"]:
        bl = crit["blur"][bins == lo_b].astype(float)
        bd = crit["blur"][bins == hi_b].astype(float)
        blur_gap = {"gap": float(bd.mean() - bl.mean()), "ci95": gap_ci(bl, bd)}
        inc0 = blur_gap["ci95"][0] <= 0 <= blur_gap["ci95"][1]
        print(f"\n  BLUR rejection {lo_b} -> {hi_b}: {blur_gap['gap']:+.4f} "
              f"CI [{blur_gap['ci95'][0]:+.4f},{blur_gap['ci95'][1]:+.4f}]"
              f"{'  includes zero' if inc0 else '  EXCLUDES ZERO'}")
        print(f"  corr(local_contrast, blur-reject) = "
              f"{float(np.corrcoef(contrast, crit['blur'].astype(float))[0, 1]):+.3f}")
        if not inc0 and blur_gap["gap"] > 0.03:
            print("  -> Blur threshold is TONE-DEPENDENT. Make it contrast-relative "
                  "(e.g. sharpness / local_contrast^2)\n     before collection: a fixed "
                  "absolute cut silently excludes darker-skinned users.")

    out = {**meta, "class_composition": comp,
           "corr_ita_lesion_area": rho_area, "corr_ita_annulus_overlap": rho_ovl,
           "clean_sample_gradient": res, "lesion_contamination": bool(collapsed),
           "verdict_contamination": v1,
           "quality_gate_by_ita": gate, "blur_gap": blur_gap,
           "fairness_measurable": False,
           "caveat": "ITA remains a proxy. For the holdout, sample ITA from an annulus "
                     "outside the USER-PLACED RETICLE, not a heuristic perilesional patch."}
    p = ckpt.parent / "ita_confound.json"
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
