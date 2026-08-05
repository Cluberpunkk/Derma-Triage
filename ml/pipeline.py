"""ml/pipeline.py
=========================================================================
THE FULL TRIAGE DECISION CHAIN — every gate we built, in the order they must fire.

    image (+ optional clinical metadata)
      |
      1. QUALITY GATE      too blurry / dark / bright / lesion not framed?
      |                    -> RETAKE  (mirrors src/lib/cameraQuality.js thresholds)
      2. OOD GATE          is this even skin?  (Mahalanobis + max-softmax)
      |                    -> NOT_SKIN
      3. CLASSIFY          23-way coarse categories (+ metadata if supplied)
      |
      4. ABSTENTION        confident enough to show anything?
      |                    -> UNCERTAIN -> see a clinician
      5. ESCALATION        can't-miss category anywhere in top-3, at ANY confidence
      |                    -> ESCALATE, overrides everything above except retake
      v
    decision + top-3 shortlist + explicit caveats

WHY THE ORDER MATTERS
---------------------
Quality first: a blurry photo makes every downstream score meaningless. OOD before
classification: a 23-way softmax has no "none of these" option, so a wall gets a
skin disease. Abstention before display: ~10% of predictions are confidently wrong.
Escalation LAST and overriding: a possible malignancy must survive every earlier
gate that would otherwise have quietly suppressed it — a missed cancer routed to
"uncertain" is acceptable, a missed cancer never surfaced is not.

WHAT THIS DELIBERATELY NEVER DOES
---------------------------------
No treatment, medicine, cream or product is ever emitted — the output schema has no
field for one. No single diagnosis is ever presented: only a top-3 shortlist with
confidence, or an honest refusal. Nothing here suppresses the app's own urgency
assessment or red-flag path; escalation is additive only.

    ml/.venv/bin/python ml/pipeline.py --fit data/ml/runs_final/best.pt
    ml/.venv/bin/python ml/pipeline.py data/ml/runs_final/best.pt photo.jpg
    ml/.venv/bin/python ml/pipeline.py best.pt photo.jpg --meta body_site=Groin,prior_steroid_use=yes
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
import metadata as md  # noqa: E402
from dataset import ManifestImageDataset, build_transforms  # noqa: E402
from evaluate import CANT_MISS_SUBSTRINGS  # noqa: E402
from ood import Mahalanobis, collect, penultimate  # noqa: E402
from train import load_checkpoint  # noqa: E402

# Mirrors THRESHOLDS in src/lib/cameraQuality.js so the server agrees with the client.
QUALITY = {"sharpness": 55.0, "brightness_low": 42.0, "brightness_high": 222.0}

# Abstention: below this max-softmax the model says "unclear" instead of guessing.
# From abstain.py — 0.70 cuts confidently-missed malignancies 145 -> 33 on the 192px
# model. Refit per checkpoint; --fit records the value actually used.
ABSTAIN_BELOW = 0.70

DECISIONS = ("RETAKE", "NOT_SKIN", "UNCERTAIN", "SHORTLIST", "ESCALATE")


def image_quality(img: Image.Image) -> dict:
    """Variance-of-Laplacian sharpness + mean luma, on a downscaled grey copy.

    Same measures and thresholds as the browser-side gate, so a photo accepted in
    the app is not silently rejected here (or vice versa).
    """
    g = np.asarray(img.convert("L").resize((200, 200)), dtype=np.float32)
    lap = (-4 * g
           + np.roll(g, 1, 0) + np.roll(g, -1, 0)
           + np.roll(g, 1, 1) + np.roll(g, -1, 1))[1:-1, 1:-1]
    sharp, bright = float(lap.var()), float(g.mean())
    issues = []
    if sharp < QUALITY["sharpness"]:
        issues.append("too blurry — hold steady and move closer")
    if bright < QUALITY["brightness_low"]:
        issues.append("too dark — find better light")
    if bright > QUALITY["brightness_high"]:
        issues.append("too bright / glare — avoid direct light")
    return {"sharpness": sharp, "brightness": bright, "issues": issues,
            "ok": not issues}


class TriagePipeline:
    def __init__(self, ckpt_path: Path, device: str, gates: dict | None = None):
        self.model, self.ck = load_checkpoint(ckpt_path, device)
        self.device = device
        self.classes = self.ck["classes"]
        self.tf = build_transforms(self.ck["img_size"], train=False)
        self.cant_miss = {i for i, c in enumerate(self.classes)
                          if any(s in c.lower() for s in CANT_MISS_SUBSTRINGS)}
        self.gates = gates
        self.maha = None
        if gates and "maha_mu" in gates:
            self.maha = Mahalanobis.__new__(Mahalanobis)
            self.maha.mu = torch.tensor(gates["maha_mu"])
            self.maha.prec = torch.tensor(gates["maha_prec"])

    # ---------------------------------------------------------------- fit
    @classmethod
    def fit_gates(cls, ckpt_path: Path, device: str, manifest_dir: Path, root: Path,
                  fit_batches: int = 40) -> dict:
        """Fit the OOD reference distribution and pick thresholds from held-out data."""
        model, ck = load_checkpoint(ckpt_path, device)
        lc = ck.get("label_column", "original_class")
        tr = ManifestImageDataset(manifest_dir, root, "train", lc, img_size=ck["img_size"])
        f_tr, _ = collect(model, DataLoader(tr, batch_size=64, shuffle=True), device, fit_batches)
        maha = Mahalanobis(f_tr)

        val = ManifestImageDataset(manifest_dir, root, "val", lc, img_size=ck["img_size"])
        f_v, l_v = collect(model, DataLoader(val, batch_size=64), device)
        d = maha(f_v)
        return {
            "maha_mu": maha.mu.tolist(), "maha_prec": maha.prec.tolist(),
            # reject the most-distant 5% of genuine skin photos: the tunable cost of
            # never letting a wall through as a diagnosis
            "maha_threshold": float(d.quantile(0.95)),
            "abstain_below": ABSTAIN_BELOW,
            "fitted_on": {"n_fit": len(f_tr), "n_val": len(f_v)},
            "in_distribution_maha_mean": float(d.mean()),
            "val_maxprob_mean": float(l_v.softmax(1).max(1).values.mean()),
        }

    # ---------------------------------------------------------------- run
    @torch.no_grad()
    def __call__(self, image_path: str, meta: dict | None = None) -> dict:
        out = {"image": str(image_path), "decision": None, "shortlist": [],
               "escalate": False, "caveats": CAVEATS}

        img = Image.open(image_path).convert("RGB")

        # 1 -- quality
        q = image_quality(img)
        out["quality"] = q
        if not q["ok"]:
            out["decision"] = "RETAKE"
            out["message"] = "Photo quality too low to assess. " + "; ".join(q["issues"])
            return out

        x = self.tf(img).unsqueeze(0).to(self.device)
        logits = self.model(x).cpu()
        probs = logits.softmax(1)[0]

        # 2 -- out of distribution
        if self.maha is not None:
            dist = float(self.maha(penultimate(self.model, x).cpu()))
            out["ood"] = {"mahalanobis": dist, "threshold": self.gates["maha_threshold"]}
            if dist > self.gates["maha_threshold"]:
                out["decision"] = "NOT_SKIN"
                out["message"] = ("This doesn't look like a photo of a skin problem. "
                                  "Please photograph the affected area directly.")
                return out

        # 3 -- classify
        k = min(3, len(self.classes))
        top = probs.topk(k)
        out["shortlist"] = [{"category": self.classes[int(i)], "confidence": round(float(p), 4)}
                            for p, i in zip(top.values, top.indices)]
        top1 = float(top.values[0])

        # 5 -- escalation is evaluated BEFORE abstention can hide it
        hit = [self.classes[int(i)] for i in top.indices if int(i) in self.cant_miss]
        if hit:
            out["escalate"] = True
            out["decision"] = "ESCALATE"
            out["message"] = ("This may need prompt in-person assessment. Please see a "
                              "doctor or dermatologist soon.")
            out["escalation_reason"] = f"possible serious category in shortlist: {hit}"
            return out

        # 4 -- abstain
        thr = (self.gates or {}).get("abstain_below", ABSTAIN_BELOW)
        if top1 < thr:
            out["decision"] = "UNCERTAIN"
            out["message"] = ("Not clear enough from this photo to suggest possibilities. "
                              "Please see a clinician.")
            out["max_confidence"] = round(top1, 4)
            return out

        out["decision"] = "SHORTLIST"
        out["message"] = ("Possible categories to discuss with a clinician — not a "
                          "diagnosis.")
        return out


CAVEATS = [
    "Coarse category, NOT a diagnosis — source folders bundle multiple diseases.",
    "Trained on public-atlas images with NO skin-tone labels; reliability across skin "
    "tones is unmeasured.",
    "Not a validated dermatology model. Triage + education only; no treatment guidance.",
]


def parse_meta(s: str | None) -> dict:
    if not s:
        return {}
    out = {}
    for pair in s.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            out[k.strip()] = v.strip()
    bad = set(out) - set(md.FIELD_NAMES)
    if bad:
        print(f"  warning: unknown metadata fields ignored: {sorted(bad)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("images", nargs="*")
    ap.add_argument("--fit", action="store_true", help="fit + save the OOD/abstain gates")
    ap.add_argument("--meta", help="k=v,k=v clinical fields")
    ap.add_argument("--manifest"), ap.add_argument("--root"), ap.add_argument("--device")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    ckpt = Path(a.checkpoint).resolve()
    device = config.resolve_device(a.device)
    gates_path = ckpt.parent / "gates.json"

    if a.fit:
        manifest_dir = Path(a.manifest) if a.manifest else config.out_dir() / "dermnet"
        root = config.dataset_root(a.root)
        print(f"fitting gates for {ckpt.name} …")
        g = TriagePipeline.fit_gates(ckpt, device, manifest_dir, root)
        gates_path.write_text(json.dumps(g))
        print(f"  mahalanobis threshold {g['maha_threshold']:.1f} "
              f"(in-dist mean {g['in_distribution_maha_mean']:.1f})")
        print(f"  abstain below max-softmax {g['abstain_below']}")
        print(f"  wrote {gates_path}")
        if not a.images:
            return 0

    if not a.images:
        print("no images given. Use --fit to build gates, or pass image paths.")
        return 1

    gates = json.loads(gates_path.read_text()) if gates_path.exists() else None
    if gates is None:
        print(f"  warning: no {gates_path.name} — OOD gate DISABLED. Run --fit first.")
    pipe = TriagePipeline(ckpt, device, gates)

    results = []
    for p in a.images:
        r = pipe(p, parse_meta(a.meta))
        results.append(r)
        if a.json:
            continue
        print(f"\n{'=' * 64}\n{p}\n  DECISION: {r['decision']}"
              + ("   [ESCALATE]" if r["escalate"] else ""))
        print(f"  {r.get('message', '')}")
        if r.get("quality"):
            q = r["quality"]
            print(f"  quality: sharpness {q['sharpness']:.0f} brightness {q['brightness']:.0f}")
        if r.get("ood"):
            print(f"  ood: mahalanobis {r['ood']['mahalanobis']:.0f} "
                  f"(reject above {r['ood']['threshold']:.0f})")
        for s in r["shortlist"]:
            bar = "█" * max(1, round(s["confidence"] * 24))
            print(f"    {s['confidence']:6.1%} {bar:<24} {s['category']}")
    if a.json:
        print(json.dumps(results, indent=2))
    else:
        print("\n" + "\n".join(f"NOTE: {c}" for c in CAVEATS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
