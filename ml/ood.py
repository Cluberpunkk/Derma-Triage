"""ml/ood.py
=========================================================================
OUT-OF-DISTRIBUTION DETECTION — "is this even a photo of skin?"

THE EDGE CASE NOTHING ELSE COVERS
---------------------------------
A 23-way classifier must output one of 23 skin conditions for ANY input. Photograph
a wall, a shoe, a plate of food, or a face with no lesion, and it will return a
dermatological condition — often confidently. Confidence thresholding (abstain.py)
does not catch this: OOD inputs frequently produce HIGH max-probability, because
the softmax is normalised over 23 classes that all happen to be wrong.

This is a real product failure, not a theoretical one. It is also the failure most
likely to reach a user, because people test new apps by pointing them at whatever
is nearby.

TWO SCORES, NEITHER NEEDING RETRAINING
--------------------------------------
  energy       -logsumexp(logits). Uses the raw logit magnitudes rather than the
               normalised softmax, so it does not get fooled by a confident-looking
               distribution over wrong classes.
  mahalanobis  distance from the training feature distribution in the backbone's
               penultimate space. Directly answers "does this look like anything
               the model was trained on?" — which is the actual question.

Both are fitted from one pass over the training split; the checkpoint is untouched.

HONESTY ABOUT THE VALIDATION
----------------------------
Real OOD means real non-skin photographs, which we do not have locally. So this
validates against SYNTHETIC probes (noise, flat colours, patch-scrambled images,
extreme blur/inversion). Those are legitimate OOD, and a detector failing on them
is definitely broken — but passing them is NOT proof it handles photos of walls and
faces. Treat the reported AUROC as a lower bound and re-validate on real non-skin
images before shipping. This limitation is printed with the results.

    ml/.venv/bin/python ml/ood.py data/ml/runs_final/best.pt
    ml/.venv/bin/python ml/ood.py best.pt --image somephoto.jpg   # score one image
=========================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from dataset import ManifestImageDataset, build_transforms  # noqa: E402
from train import load_checkpoint  # noqa: E402


# ---------------------------------------------------------------- features

def penultimate(model, x: torch.Tensor) -> torch.Tensor:
    """Pooled pre-logit features, for timm or torchvision backbones alike."""
    if hasattr(model, "forward_features") and hasattr(model, "forward_head"):
        return model.forward_head(model.forward_features(x), pre_logits=True)
    # torchvision fallback: hook whatever feeds the final Linear
    feats = {}
    last_linear = [m for m in model.modules() if isinstance(m, torch.nn.Linear)][-1]
    h = last_linear.register_forward_hook(lambda _m, inp, _o: feats.setdefault("v", inp[0]))
    try:
        model(x)
    finally:
        h.remove()
    return feats["v"]


@torch.no_grad()
def collect(model, loader, device, limit_batches: int | None = None):
    F, L = [], []
    for i, (x, _y) in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        x = x.to(device)
        F.append(penultimate(model, x).cpu())
        L.append(model(x).cpu())
    return torch.cat(F), torch.cat(L)


# ---------------------------------------------------------------- scores

def energy(logits: torch.Tensor) -> torch.Tensor:
    """Higher = more out-of-distribution."""
    return -torch.logsumexp(logits, dim=1)


class Mahalanobis:
    """Single-Gaussian fit of the training features; distance = OOD score."""

    def __init__(self, feats: torch.Tensor, shrink: float = 1e-3):
        self.mu = feats.mean(0)
        d = feats - self.mu
        cov = (d.T @ d) / max(len(feats) - 1, 1)
        cov += torch.eye(cov.shape[0]) * (shrink * cov.diag().mean())  # ridge for stability
        self.prec = torch.linalg.inv(cov.double()).float()

    def __call__(self, feats: torch.Tensor) -> torch.Tensor:
        d = feats - self.mu
        return ((d @ self.prec) * d).sum(1).clamp(min=0).sqrt()


def auroc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """P(score(OOD) > score(in-dist)). 1.0 = perfect separation, 0.5 = useless."""
    all_s = torch.cat([pos, neg])
    ranks = all_s.argsort().argsort().float() + 1
    r_pos = ranks[:len(pos)].sum()
    n1, n2 = len(pos), len(neg)
    return float((r_pos - n1 * (n1 + 1) / 2) / (n1 * n2))


# ---------------------------------------------------------------- OOD probes

def probes(reference: torch.Tensor, n: int = 200) -> dict[str, torch.Tensor]:
    """Synthetic non-skin inputs, built to mimic distinct real failure modes."""
    g = torch.Generator().manual_seed(42)
    b, c, h, w = n, *reference.shape[1:]
    out = {}

    # camera pointed at nothing / sensor noise
    out["gaussian_noise"] = torch.randn(b, c, h, w, generator=g)

    # flat surfaces: a wall, a table, a screen
    flat = torch.rand(b, c, 1, 1, generator=g) * 4 - 2
    out["flat_colour"] = flat.expand(b, c, h, w).contiguous()

    # right texture, destroyed structure — the hardest probe, and the one that
    # separates a real detector from one keyed on colour statistics alone
    src = reference[torch.randint(0, len(reference), (b,), generator=g)]
    k = 8
    patches = src.unfold(2, h // k, h // k).unfold(3, w // k, w // k).contiguous()
    shape = patches.shape
    patches = patches.view(b, c, -1, *shape[-2:])
    idx = torch.stack([torch.randperm(patches.shape[2], generator=g) for _ in range(b)])
    patches = torch.stack([patches[i, :, idx[i]] for i in range(b)])
    out["scrambled_patches"] = patches.permute(0, 2, 1, 3, 4).reshape(
        b, c, k, k, h // k, w // k).permute(0, 1, 2, 4, 3, 5).reshape(b, c, h, w)

    # heavy defocus — a plausible real capture failure
    blur = src.clone()
    for _ in range(6):
        blur = torch.nn.functional.avg_pool2d(blur, 3, 1, 1)
    out["extreme_blur"] = blur

    # inverted / wrong colour space
    out["inverted"] = -src
    return out


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--manifest"), ap.add_argument("--root")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--fit-batches", type=int, default=40, help="train batches used to fit")
    ap.add_argument("--device")
    ap.add_argument("--image", help="score a single image instead of running the benchmark")
    a = ap.parse_args()

    ckpt_path = Path(a.checkpoint).resolve()
    device = config.resolve_device(a.device)
    model, ck = load_checkpoint(ckpt_path, device)
    manifest_dir = Path(a.manifest) if a.manifest else config.out_dir() / "dermnet"
    root = config.dataset_root(a.root)
    lc = ck.get("label_column", "original_class")

    print(f"checkpoint {ckpt_path}\n  {ck['model_name']} · {ck['img_size']}px · {device}\n")

    tr = ManifestImageDataset(manifest_dir, root, "train", lc, img_size=ck["img_size"])
    tr_loader = DataLoader(tr, batch_size=a.batch_size, shuffle=True, num_workers=0)
    print(f"fitting reference distribution on {a.fit_batches} train batches…")
    f_tr, _ = collect(model, tr_loader, device, a.fit_batches)
    maha = Mahalanobis(f_tr)
    print(f"  feature dim {f_tr.shape[1]}, n={len(f_tr)}")

    if a.image:
        from PIL import Image
        tf = build_transforms(ck["img_size"], train=False)
        x = tf(Image.open(a.image).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            lg = model(x).cpu()
            ft = penultimate(model, x).cpu()
        print(f"\n{a.image}\n  energy      {float(energy(lg)):.3f}"
              f"\n  mahalanobis {float(maha(ft)):.3f}"
              f"\n  max softmax {float(lg.softmax(1).max()):.3f}"
              f"\n  top class   {ck['classes'][int(lg.argmax())]}")
        print("\nCompare against the in-distribution ranges printed by a full run.")
        return 0

    te = ManifestImageDataset(manifest_dir, root, "test", lc, img_size=ck["img_size"])
    te_loader = DataLoader(te, batch_size=a.batch_size, shuffle=False, num_workers=0)
    print("scoring in-distribution test images…")
    f_id, l_id = collect(model, te_loader, device)
    e_id, m_id, s_id = energy(l_id), maha(f_id), l_id.softmax(1).max(1).values

    print(f"\nin-distribution (n={len(f_id)}):")
    print(f"  energy       mean {e_id.mean():.3f}  p95 {e_id.quantile(0.95):.3f}")
    print(f"  mahalanobis  mean {m_id.mean():.3f}  p95 {m_id.quantile(0.95):.3f}")
    print(f"  max softmax  mean {s_id.mean():.3f}")

    ref = next(iter(te_loader))[0]
    results = {}
    print(f"\n{'probe':<20} {'maxprob':>8} {'energy':>8} {'maha':>8} | "
          f"{'AUROC energy':>13} {'AUROC maha':>11} {'AUROC 1-maxp':>13}")
    for name, batch in probes(ref).items():
        with torch.no_grad():
            lg, ft = [], []
            for i in range(0, len(batch), a.batch_size):
                xb = batch[i:i + a.batch_size].to(device)
                lg.append(model(xb).cpu())
                ft.append(penultimate(model, xb).cpu())
            lg, ft = torch.cat(lg), torch.cat(ft)
        e_o, m_o, s_o = energy(lg), maha(ft), lg.softmax(1).max(1).values
        r = {"mean_maxprob": float(s_o.mean()), "mean_energy": float(e_o.mean()),
             "mean_maha": float(m_o.mean()),
             "auroc_energy": auroc(e_o, e_id), "auroc_maha": auroc(m_o, m_id),
             "auroc_maxprob": auroc(-s_o, -s_id)}
        results[name] = r
        print(f"{name:<20} {r['mean_maxprob']:>8.3f} {r['mean_energy']:>8.3f} "
              f"{r['mean_maha']:>8.3f} | {r['auroc_energy']:>13.3f} "
              f"{r['auroc_maha']:>11.3f} {r['auroc_maxprob']:>13.3f}")

    # operating threshold: flag OOD above the 95th percentile of in-distribution
    thr = {"energy_p95": float(e_id.quantile(0.95)), "maha_p95": float(m_id.quantile(0.95))}
    print(f"\nsuggested gate (flags 5% of genuine skin photos as 'unclear'):")
    print(f"  reject if energy > {thr['energy_p95']:.3f} OR mahalanobis > {thr['maha_p95']:.3f}")

    mean_auroc = {k: sum(r[k] for r in results.values()) / len(results)
                  for k in ("auroc_energy", "auroc_maha", "auroc_maxprob")}
    print(f"\nmean AUROC across probes: energy {mean_auroc['auroc_energy']:.3f}  "
          f"mahalanobis {mean_auroc['auroc_maha']:.3f}  "
          f"max-softmax {mean_auroc['auroc_maxprob']:.3f}")
    print("  (max-softmax is the baseline every competitor uses — compare against it)")

    out = ckpt_path.parent / "ood_report.json"
    out.write_text(json.dumps({
        "checkpoint": str(ckpt_path), "n_in_distribution": len(f_id),
        "in_distribution": {"energy_mean": float(e_id.mean()),
                            "maha_mean": float(m_id.mean()),
                            "maxprob_mean": float(s_id.mean())},
        "probes": results, "thresholds": thr, "mean_auroc": mean_auroc,
        "limitation": "Validated on SYNTHETIC probes only. Real OOD (walls, faces, food) "
                      "must be tested with real non-skin photographs before shipping; "
                      "treat these AUROCs as a lower bound.",
    }, indent=2))
    print(f"\nWrote {out}")
    print("\nLIMITATION: synthetic probes only. Re-validate on real non-skin photos "
          "before relying on this gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
