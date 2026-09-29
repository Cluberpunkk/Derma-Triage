"""ml/skin_detect.py
=========================================================================
Is there any skin in this photo at all?

THE GAP THIS ADDRESSES
Nothing in the capture path checks. Measured: a flat grey wall gets localiser
confidence 0.306 — above the 0.25 fallback threshold — so the pipeline reports a
CONFIDENT lesion box on a wall. Worse, the ITA estimator returns +80.7 for that wall
("very light") and -22.1 for blue sky ("brown"), so a photo of the sky would be
binned as brown skin in the fairness stratification.

Users will photograph faces, clothes, walls and pets. Without a check, those become
valid-looking holdout rows and contaminate both the dataset and the ITA analysis.

THE CHEAPEST CHECK: SKIN-PIXEL FRACTION IN THE RETICLE
Classic YCbCr chrominance thresholding (Chai & Ngan): skin occupies a compact region
in Cb/Cr that is substantially INVARIANT TO LUMINANCE — which is precisely the
property we need, because a detector keyed on brightness would fail on darker skin
and repeat the localiser defect we just fixed.

That invariance is the reason to prefer it over an RGB rule. It is not a guarantee,
which is why this script measures the false-rejection rate stratified by ITA rather
than assuming it.

    ml/.venv/bin/python ml/skin_detect.py                    # measure on DermNet
    ml/.venv/bin/python ml/skin_detect.py --dir path/to/photos   # score real non-skin
=========================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import runmeta  # noqa: E402
from dataset import ManifestImageDataset  # noqa: E402
from ita_audit import ITA_BINS, boot_ci, ita_bin, ita_from_ring  # noqa: E402
from shortcut_test import _lab  # noqa: E402

# Chai & Ngan YCbCr skin locus. Luminance-invariant by construction.
CB_LO, CB_HI = 77, 127
CR_LO, CR_HI = 133, 173
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]


def skin_fraction(img: Image.Image, reticle_only: bool = True) -> float:
    """Fraction of pixels inside the reticle region falling in the YCbCr skin locus."""
    a = np.asarray(img.convert("RGB").resize((160, 160)), dtype=np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    m = (cb >= CB_LO) & (cb <= CB_HI) & (cr >= CR_LO) & (cr <= CR_HI)
    if reticle_only:
        h, w = m.shape
        yy, xx = np.mgrid[0:h, 0:w]
        rad = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (min(h, w) / 2)
        inside = rad <= 0.55
        return float(m[inside].mean())
    return float(m.mean())


def probes(n: int = 60) -> dict[str, list[Image.Image]]:
    """Synthetic non-skin. NOT a substitute for real photos of walls, pets and clothes —
    see the caveat printed with the results."""
    g = np.random.default_rng(0)
    out: dict[str, list[Image.Image]] = {}

    def make(base, jitter=10):
        ims = []
        for _ in range(n):
            a = np.zeros((240, 240, 3), np.int16)
            a[:] = base
            a += g.integers(-jitter, jitter, a.shape)
            ims.append(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)))
        return ims

    out["grey wall"] = make((150, 148, 145))
    out["blue sky"] = make((110, 160, 220))
    out["green foliage"] = make((70, 120, 60))
    out["dark fabric"] = make((40, 42, 55))
    out["white paper"] = make((235, 235, 232), 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="folder of REAL non-skin photos to score")
    ap.add_argument("--manifest"), ap.add_argument("--root")
    ap.add_argument("--limit", type=int, default=2000)
    a = ap.parse_args()

    meta = runmeta.manifest(tool="skin_detect.py",
                            checkpoint=config.out_dir() / "runs_final" / "best.pt",
                            extra={"ycbcr_locus": [CB_LO, CB_HI, CR_LO, CR_HI]})
    runmeta.print_header(meta)

    # ---------- real skin, stratified by ITA ----------
    manifest_dir = Path(a.manifest) if a.manifest else config.out_dir() / "dermnet"
    root = config.dataset_root(a.root)
    te = ManifestImageDataset(manifest_dir, root, "test", "original_class", img_size=224)
    rows = te.rows[:a.limit]
    print(f"\nscoring {len(rows)} REAL skin photos…")
    fr, bins = [], []
    for i, r in enumerate(rows):
        im = Image.open(root / r["image_path"]).convert("RGB")
        fr.append(skin_fraction(im))
        small = np.asarray(im.resize((224, 224)), dtype=np.float32) / 255.0
        bins.append(ita_bin(ita_from_ring(_lab(small))))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(rows)}")
    fr, bins = np.array(fr), np.array(bins)

    print("\n=== SKIN-PIXEL FRACTION ON REAL SKIN, BY ITA BIN ===")
    print(f"  {'bin':<14} {'n':>5} {'mean':>7} {'p10':>7} {'p25':>7} {'median':>8}")
    order = [b[2] for b in ITA_BINS]
    per_bin = {}
    for b in order:
        m = bins == b
        if m.sum() < 25:
            continue
        per_bin[b] = {"n": int(m.sum()), "mean": float(fr[m].mean()),
                      "p10": float(np.percentile(fr[m], 10)),
                      "p25": float(np.percentile(fr[m], 25)),
                      "median": float(np.median(fr[m]))}
        v = per_bin[b]
        print(f"  {b:<14} {v['n']:>5} {v['mean']:>7.3f} {v['p10']:>7.3f} "
              f"{v['p25']:>7.3f} {v['median']:>8.3f}")

    # ---------- non-skin probes ----------
    print("\n=== SYNTHETIC NON-SKIN PROBES ===")
    print(f"  {'probe':<16} {'mean skin fraction':>20}")
    probe_scores = {}
    for name, ims in probes().items():
        s = np.array([skin_fraction(im) for im in ims])
        probe_scores[name] = float(s.mean())
        print(f"  {name:<16} {s.mean():>20.3f}")

    real_dir = {}
    if a.dir:
        d = Path(a.dir)
        files = [p for p in d.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        print(f"\n=== REAL NON-SKIN PHOTOS from {d} ({len(files)}) ===")
        for p in files:
            try:
                s = skin_fraction(Image.open(p))
                real_dir[p.name] = s
                print(f"  {p.name[:40]:<40} {s:.3f}"
                      + ("   <- would PASS a 0.30 gate" if s >= 0.30 else ""))
            except Exception as e:  # noqa: BLE001
                print(f"  {p.name[:40]:<40} unreadable ({type(e).__name__})")

    # ---------- the cost: false rejections on real skin, by ITA ----------
    print("\n=== COST OF A SKIN GATE: FALSE REJECTION ON REAL SKIN, BY ITA BIN ===")
    print("  (a detector that rejects darker skin more often repeats the localiser defect)")
    hdr = "  " + f"{'threshold':>9}" + "".join(f"{b[:9]:>11}" for b in per_bin) + f"{'worst gap':>11}"
    print(hdr)
    table = []
    for t in THRESHOLDS:
        rates = {}
        for b in per_bin:
            m = bins == b
            rates[b] = float((fr[m] < t).mean())
        gap = max(rates.values()) - min(rates.values())
        table.append({"threshold": t, "reject_rate_by_bin": rates, "worst_gap": gap,
                      "overall_reject": float((fr < t).mean()),
                      "probes_blocked": {k: bool(v < t) for k, v in probe_scores.items()}})
        print("  " + f"{t:>9.2f}" + "".join(f"{rates[b]:>11.3f}" for b in per_bin)
              + f"{gap:>11.3f}")

    print("\n  probe blocking by threshold:")
    for row in table:
        blocked = sum(row["probes_blocked"].values())
        print(f"    t={row['threshold']:.2f}  blocks {blocked}/{len(probe_scores)} probes, "
              f"rejects {row['overall_reject']:.1%} of real skin "
              f"(worst per-tone gap {row['worst_gap']:.3f})")

    usable = [r for r in table
              if all(r["probes_blocked"].values()) and r["worst_gap"] <= 0.05]
    if usable:
        rec = min(usable, key=lambda r: r["overall_reject"])
        verdict = (f"RECOMMEND threshold {rec['threshold']:.2f}: blocks every synthetic "
                   f"probe, rejects {rec['overall_reject']:.1%} of real skin, worst "
                   f"per-tone gap {rec['worst_gap']:.3f} (<=0.05).")
    else:
        best = min(table, key=lambda r: r["worst_gap"])
        verdict = (f"NO CLEAN THRESHOLD on synthetic probes. Smallest per-tone gap is "
                   f"{best['worst_gap']:.3f} at t={best['threshold']:.2f}, blocking "
                   f"{sum(best['probes_blocked'].values())}/{len(probe_scores)} probes. "
                   f"Decide on REAL non-skin photos, not these.")
    print("\n=== VERDICT ===\n  " + verdict.replace(". ", ".\n  "))

    print("\n  CAVEAT: the probes are flat synthetic colours. Real non-skin photos —")
    print("  faces (which ARE skin), pets, wood, sand, brick — are much closer to the")
    print("  skin locus, so these numbers are an OPTIMISTIC bound. Re-run with --dir")
    print("  on 20-30 real phone photos before trusting any threshold.")

    out = {**meta, "skin_fraction_by_ita": per_bin, "probe_scores": probe_scores,
           "real_dir_scores": real_dir, "threshold_table": table, "verdict": verdict,
           "caveat": "Synthetic probes only; real non-skin photos are closer to the "
                     "skin locus and will lower the achievable separation."}
    p = config.out_dir() / "runs_final" / "skin_detect.json"
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
