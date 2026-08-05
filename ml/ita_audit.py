"""ml/ita_audit.py
=========================================================================
Does the CAPTURE PATH work equally well across skin tones?

`src/lib/lesionCrop.js` localises the lesion by colour distance from the modal skin
tone. On darker skin the LAB distance between lesion and surrounding skin is
generally SMALLER — less lightness headroom, and pigmented lesions sit closer to the
background. If that pushes localisation confidence below the fallback threshold more
often on darker skin, then dark-skinned users silently get a blind centre crop while
light-skinned users get a native-resolution lesion crop.

That would be a fairness defect in the capture path itself — upstream of the model,
affecting every downstream number — and it must be found BEFORE collection, not
after.

ITA IS A PROXY. READ THIS BEFORE QUOTING ANY NUMBER BELOW.
---------------------------------------------------------
Individual Typology Angle, ITA = atan((L* - 50) / b*) * 180/pi, computed here from
PERILESIONAL skin (the outer ring), not the lesion. It is the standard image-based
tone estimate, and it is confounded by:

  * white balance and colour profile of the camera
  * lighting temperature and intensity
  * lesional erythema bleeding into the perilesional ring
  * DermNet's own tone distribution, which is skewed light

So this is NOT a fairness evaluation and must not be reported as one.
`fairness_measurable` stays FALSE on the strength of it. It is the first tone signal
we can compute for free, and it is good enough to detect a large capture-path defect
— which is all it is used for here.

THE LOCALISATION CODE BELOW IS A PORT of src/lib/lesionCrop.js findLesion(). The two
must stay in step; the algorithm is documented in both files.

    ml/.venv/bin/python ml/ita_audit.py data/ml/runs_final/best.pt
    ml/.venv/bin/python ml/ita_audit.py best.pt --remedy edge   # test a remedy
=========================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
from shortcut_test import _lab  # noqa: E402
from train import load_checkpoint  # noqa: E402
from urgency import (GROUPS_YAML, Platt, Temperature, load_groups, logits_labels,  # noqa: E402
                     tier_of)

ANALYSIS_W = 256          # must match lesionCrop.js ANALYSIS_W
MIN_CONFIDENCE = 0.25     # must match lesionCrop.js captureNative minConfidence
PAD = 1.6
MODEL_SIZE = 224

# Standard ITA bins (Chardon / Del Bino). Ordered light -> dark.
ITA_BINS = [(55, 1e9, "very light"), (41, 55, "light"), (28, 41, "intermediate"),
            (10, 28, "tan"), (-30, 10, "brown"), (-1e9, -30, "dark")]


def ita_from_ring(lab: np.ndarray) -> float:
    """ITA of PERILESIONAL skin — the outer ring, not the lesion."""
    h, w = lab.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (min(h, w) / 2)
    ring = r > 0.46
    if ring.sum() < 20:
        return float("nan")
    L = float(np.median(lab[..., 0][ring]))
    b = float(np.median(lab[..., 2][ring]))
    if abs(b) < 1e-6:
        return float("nan")
    return float(np.degrees(np.arctan((L - 50) / b)))


def ita_bin(v: float) -> str:
    if not np.isfinite(v):
        return "unknown"
    for lo, hi, name in ITA_BINS:
        if lo < v <= hi:
            return name
    return "unknown"


def boot_ci(v: np.ndarray, n_boot: int = 800, seed: int = 0):
    if len(v) == 0:
        return [float("nan")] * 2
    g = np.random.default_rng(seed)
    b = [v[g.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return [round(float(np.quantile(b, 0.025)), 4), round(float(np.quantile(b, 0.975)), 4)]


def gap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 800, seed: int = 1):
    """CI on the DIFFERENCE of two rates — resample both groups independently."""
    if len(a) == 0 or len(b) == 0:
        return [float("nan")] * 2
    g = np.random.default_rng(seed)
    d = [b[g.integers(0, len(b), len(b))].mean() - a[g.integers(0, len(a), len(a))].mean()
         for _ in range(n_boot)]
    return [round(float(np.quantile(d, 0.025)), 4), round(float(np.quantile(d, 0.975)), 4)]


def frame_stats(img: Image.Image) -> dict:
    """Confounds that are INDEPENDENT of the localiser's own adaptive threshold.

    non_skin_frac  framing tightness: fraction of frame further than an ABSOLUTE LAB
                   distance from modal skin. A tightly-framed close-up has a high
                   value; a wide shot of a limb has a low one.
    clip_frac      exposure: fraction of pixels with any channel at 0 or 255. ITA is
                   confounded by exposure, so the tone gradient must be re-checked
                   with clipped images excluded.
    """
    a = np.asarray(img.resize((128, 128)).convert("RGB"))
    clip = float(((a == 0) | (a == 255)).any(-1).mean())
    lab = _lab(a.astype(np.float32) / 255.0)
    keys = (np.round(lab[..., 0] / 12).astype(int) * 10000
            + np.round(lab[..., 1] / 6).astype(int) * 100
            + np.round(lab[..., 2] / 6).astype(int))
    mode = Counter(keys.ravel()).most_common(1)[0][0]
    m = keys == mode
    skin = np.array([lab[..., 0][m].mean(), lab[..., 1][m].mean(), lab[..., 2][m].mean()])
    d = np.sqrt(((lab - skin) ** 2).sum(-1))
    return {"non_skin_frac": float((d > 15).mean()), "clip_frac": clip}


def find_lesion(img: Image.Image, remedy: str = "none") -> dict:
    """Python port of src/lib/lesionCrop.js findLesion(). Keep in sync."""
    w0, h0 = img.size
    sc = ANALYSIS_W / max(w0, h0)
    aw, ah = max(32, round(w0 * sc)), max(32, round(h0 * sc))
    arr = np.asarray(img.resize((aw, ah)).convert("RGB"), dtype=np.float32) / 255.0
    lab = _lab(arr)

    # modal skin colour from a coarse LAB histogram
    keys = (np.round(lab[..., 0] / 12).astype(int) * 10000
            + np.round(lab[..., 1] / 6).astype(int) * 100
            + np.round(lab[..., 2] / 6).astype(int))
    mode = Counter(keys.ravel()).most_common(1)[0][0]
    m = keys == mode
    skin = np.array([lab[..., 0][m].mean(), lab[..., 1][m].mean(), lab[..., 2][m].mean()])

    dist = np.sqrt(((lab - skin) ** 2).sum(-1))

    if remedy == "edge":
        # colour-independent second channel: local edge energy, normalised.
        g = lab[..., 0]
        gx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
        gy = np.abs(np.diff(g, axis=0, prepend=g[:1, :]))
        e = gx + gy
        e = e / (e.max() + 1e-8)
        dist = 0.5 * (dist / (dist.max() + 1e-8)) + 0.5 * e
    elif remedy == "adaptive":
        # contrast-adaptive: rescale by the achievable contrast in THIS image rather
        # than an absolute LAB distance, so low-contrast (darker) images are not
        # penalised for having less lightness headroom.
        p99 = np.percentile(dist, 99)
        dist = dist / (p99 + 1e-8)

    maxd = float(dist.max())
    if maxd <= 1e-6:
        return {"confidence": 0.0, "fallback": True, "side_frac": 1.0}
    thr = maxd * 0.55
    sel = dist >= thr
    n = int(sel.sum())
    if n < 12:
        return {"confidence": 0.0, "fallback": True, "side_frac": 1.0}

    yy, xx = np.nonzero(sel)
    cx, cy = xx.mean(), yy.mean()
    spread = float(np.sqrt(max(((xx - cx) ** 2).sum(), ((yy - cy) ** 2).sum()) / n))
    frac = n / (aw * ah)
    conf = max(0.0, min(1.0, (1 - frac / 0.45) * min(1.0, 24 / (spread + 1e-6))))

    short = min(w0, h0)
    side = (spread * PAD * 2) / sc
    side = max(short * 0.25, min(short * 0.95, side))
    return {"confidence": conf, "fallback": conf < MIN_CONFIDENCE,
            "side_frac": side / short,
            # source pixels per model pixel — LOWER IS BETTER (a wide fallback crop
            # downsamples far more than a tight lesion crop)
            "effective_scale": (short * 0.95 if conf < MIN_CONFIDENCE else side) / MODEL_SIZE}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--manifest"), ap.add_argument("--root"), ap.add_argument("--device")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--remedy", default="none", choices=["none", "edge", "adaptive"])
    ap.add_argument("--limit", type=int)
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
    meta = runmeta.manifest(tool="ita_audit.py", checkpoint=ckpt,
                            files={"groups": str(GROUPS_YAML)},
                            extra={"frozen_test": frozen, "remedy": a.remedy,
                                   "ita_is_a_proxy": True})
    runmeta.print_header(meta, frozen)
    print(f"\nITA is a PROXY (white balance, lighting, erythema confounded). "
          f"fairness_measurable stays FALSE.\nremedy = {a.remedy}\n")

    lv, yv = logits_labels(model, DataLoader(val, batch_size=a.batch_size), device)
    lt, yt = logits_labels(model, DataLoader(te, batch_size=a.batch_size), device)
    temp = Temperature().fit(lv, yv)
    soon = torch.tensor([i for i, c in enumerate(classes) if tier_of(groups, c)[0] == "soon"])
    platt = Platt().fit(temp(lv).softmax(1)[:, soon].sum(1), torch.isin(yv, soon))
    p = temp(lt).softmax(1)
    score = platt(p[:, soon].sum(1)).numpy()
    ent = (-(p.clamp(min=1e-12).log() * p).sum(1) / np.log(len(classes))).numpy()
    y = yt.numpy()

    rows = te.rows[:a.limit] if a.limit else te.rows
    print(f"computing ITA + localisation for {len(rows)} images…")
    itas, confs, fbs, escales, nsf, clipf = [], [], [], [], [], []
    for i, r in enumerate(rows):
        img = Image.open(root / r["image_path"]).convert("RGB")
        small = np.asarray(img.resize((224, 224)), dtype=np.float32) / 255.0
        itas.append(ita_from_ring(_lab(small)))
        loc = find_lesion(img, a.remedy)
        confs.append(loc["confidence"])
        fbs.append(loc["fallback"])
        escales.append(loc.get("effective_scale", np.nan))
        fs_ = frame_stats(img)
        nsf.append(fs_["non_skin_frac"])
        clipf.append(fs_["clip_frac"])
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(rows)}")

    itas = np.array(itas)
    confs, fbs, escales = np.array(confs), np.array(fbs), np.array(escales)
    nsf, clipf = np.array(nsf), np.array(clipf)
    n = len(rows)
    y, score, ent = y[:n], score[:n], ent[:n]
    is_pos = np.isin(y, soon.numpy())
    pig = np.isin(y, [ci[c] for c in BENIGN_PIGMENTED if c in ci])
    ranked = np.empty(n, dtype=int)
    ranked[np.argsort(-score)] = np.arange(n)
    top10 = ranked < int(round(0.10 * n))

    bins = np.array([ita_bin(v) for v in itas])
    print(f"\nITA distribution (proxy): " + "  ".join(
        f"{k}={v}" for k, v in Counter(bins).most_common()))

    print(f"\n=== CAPTURE PATH BY ITA BIN (light -> dark) ===")
    print(f"  {'bin':<14} {'n':>5} {'medITA':>7} {'conf':>7} {'FALLBACK':>9} "
          f"{'dnsmpl':>6} | {'cap@10':>7} {'urgRec':>7} {'pigFlag':>8} {'entropy':>8}")
    out_rows = []
    order = [b[2] for b in ITA_BINS] + ["unknown"]
    for b in order:
        m = bins == b
        if m.sum() < 25:
            continue
        mu, mp = m & is_pos, m & pig
        r = {"bin": b, "n": int(m.sum()), "median_ita": float(np.nanmedian(itas[m])),
             "mean_confidence": float(confs[m].mean()),
             "fallback_rate": float(fbs[m].mean()),
             "mean_downsample_factor": float(np.nanmean(escales[m])),  # LOWER IS BETTER
             "top10_capture": float(top10[mu].mean()) if mu.sum() else float("nan"),
             "urgent_recall_0.32": float((score[mu] >= 0.32).mean()) if mu.sum() else float("nan"),
             "pigmented_flag_0.32": float((score[mp] >= 0.32).mean()) if mp.sum() else float("nan"),
             "mean_entropy": float(ent[m].mean()),
             "fallback_ci95": boot_ci(fbs[m].astype(float)),
             "mean_non_skin_frac": float(nsf[m].mean()),
             "mean_clip_frac": float(clipf[m].mean())}
        out_rows.append(r)
        print(f"  {b:<14} {r['n']:>5} {r['median_ita']:>7.1f} {r['mean_confidence']:>7.3f} "
              f"{r['fallback_rate']:>9.3f} [{r['fallback_ci95'][0]:.2f}-{r['fallback_ci95'][1]:.2f}] "
              f"{r['mean_downsample_factor']:>6.2f} | "
              f"{r['top10_capture']:>7.3f} {r['urgent_recall_0.32']:>7.3f} "
              f"{r['pigmented_flag_0.32']:>8.3f} {r['mean_entropy']:>8.3f}")

    # ---- defect test: does fallback rise as ITA falls? ----
    valid = [r for r in out_rows if r["bin"] != "unknown"]
    verdict = "insufficient bins"
    defect = False
    if len(valid) >= 2:
        lightest, darkest = valid[0], valid[-1]
        d_fb = darkest["fallback_rate"] - lightest["fallback_rate"]
        d_conf = darkest["mean_confidence"] - lightest["mean_confidence"]
        rho = float(np.corrcoef(itas[np.isfinite(itas)],
                                fbs[np.isfinite(itas)].astype(float))[0, 1])
        gci = gap_ci(fbs[bins == lightest["bin"]].astype(float),
                     fbs[bins == darkest["bin"]].astype(float))
        includes_zero = gci[0] <= 0 <= gci[1]
        print(f"  gap 95% CI [{gci[0]:+.4f}, {gci[1]:+.4f}] -> "
              f"{'INCLUDES ZERO — not distinguishable from zero on this corpus' if includes_zero else 'excludes zero'}")

        # --- confound (a): framing tightness ---
        q = np.nanquantile(nsf, 0.25)
        wide = nsf <= q          # widest-framed quartile = least non-skin
        rho_tight = float(np.corrcoef(nsf, fbs.astype(float))[0, 1])
        print(f"\n  CONFOUND (a) framing tightness: corr(non_skin_frac, fallback) = {rho_tight:+.3f}")
        wl = fbs[wide & (bins == lightest["bin"])].astype(float)
        wd = fbs[wide & (bins == darkest["bin"])].astype(float)
        if len(wl) >= 20 and len(wd) >= 20:
            wgap = wd.mean() - wl.mean()
            wci = gap_ci(wl, wd)
            print(f"    widest-framed quartile only: fallback {wl.mean():.3f} (n={len(wl)}) -> "
                  f"{wd.mean():.3f} (n={len(wd)}), gap {wgap:+.4f} CI [{wci[0]:+.4f},{wci[1]:+.4f}]")
        else:
            wgap, wci = float("nan"), [float("nan")] * 2
            print(f"    widest-framed quartile: too few per bin (n={len(wl)}, {len(wd)})")

        # --- confound (b): exposure ---
        ok = clipf <= 0.02
        el = fbs[ok & (bins == lightest["bin"])].astype(float)
        ed = fbs[ok & (bins == darkest["bin"])].astype(float)
        print(f"\n  CONFOUND (b) exposure: {int((~ok).sum())}/{len(ok)} images have >2% clipped pixels")
        if len(el) >= 20 and len(ed) >= 20:
            egap = ed.mean() - el.mean()
            eci = gap_ci(el, ed)
            print(f"    unclipped only: fallback {el.mean():.3f} (n={len(el)}) -> "
                  f"{ed.mean():.3f} (n={len(ed)}), gap {egap:+.4f} CI [{eci[0]:+.4f},{eci[1]:+.4f}]")
        else:
            egap, eci = float("nan"), [float("nan")] * 2
            print(f"    unclipped: too few per bin (n={len(el)}, {len(ed)})")
        print(f"\n  lightest ({lightest['bin']}) -> darkest ({darkest['bin']}): "
              f"fallback {d_fb:+.3f}, confidence {d_conf:+.3f}")
        print(f"  corr(ITA, fallback) = {rho:+.3f}  (negative = more fallback on darker skin)")
        defect = (d_fb > 0.05 or rho < -0.10) and not includes_zero
        verdict = (
            f"CAPTURE-PATH DEFECT. Fallback to a blind centre crop is {d_fb:+.3f} more "
            f"likely on the darkest bin than the lightest (corr(ITA,fallback)={rho:+.3f}). "
            f"Darker-skinned users would systematically get a worse crop. Fix before "
            f"collection: try --remedy edge or --remedy adaptive and re-measure."
            if defect else
            f"NO DEFECT DETECTED. Fallback differs by {d_fb:+.3f} across bins and "
            f"corr(ITA,fallback)={rho:+.3f}. The colour-distance localiser does not "
            f"obviously degrade on darker skin IN THIS CORPUS — whose tone range is "
            f"narrow and light-skewed, so this is weak evidence, not clearance.")
    print("\n=== VERDICT ===\n  " + verdict.replace(". ", ".\n  "))

    out = {**meta, "ita_bins": out_rows, "remedy": a.remedy,
           "defect_detected": bool(defect), "verdict": verdict,
           "gap_ci95": gci, "gap_includes_zero": bool(includes_zero),
           "confound_framing": {"corr_nonskin_fallback": rho_tight,
                                "widest_quartile_gap": wgap, "widest_quartile_ci95": wci},
           "confound_exposure": {"n_clipped": int((~ok).sum()),
                                 "unclipped_gap": egap, "unclipped_ci95": eci},
           "fairness_measurable": False,
           "proxy_caveat": "ITA from perilesional skin is confounded by white balance, "
                           "lighting and erythema; DermNet's tone range is narrow and "
                           "light-skewed. Detects a large capture-path defect only. Not a "
                           "fairness evaluation."}
    pth = ckpt.parent / f"ita_audit_{a.remedy}.json"
    pth.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
