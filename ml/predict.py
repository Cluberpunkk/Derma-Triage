"""ml/predict.py
=========================================================================
Offline inspection of a trained checkpoint. Point it at images and it prints the
top-k coarse CATEGORY scores.

    ml/.venv/bin/python ml/predict.py data/ml/runs/best.pt path/to/image.jpg
    ml/.venv/bin/python ml/predict.py best.pt img1.jpg img2.jpg --topk 5 --json

WHAT THIS IS FOR: checking that a checkpoint trained on Kaggle actually loads and
behaves sanely on a few images, and eyeballing where it is confidently wrong.

WHAT IT IS NOT: a diagnosis, and not a component of the app. It is deliberately a
standalone CLI with no import path into src/ or server/. Wiring it into the triage
flow would be a product decision with real safety consequences — the labels are
coarse atlas buckets (several bundle multiple diseases), the training corpus has no
skin-tone labels, and nothing here has been clinically validated. It prints
categories and confidence only: never a treatment, medicine, cream, or product, and
it does not replace the app's urgency assessment or red-flag escalation.

The checkpoint carries its own `classes`, `img_size` and `label_column`, so the
label order can never drift from what it was trained with.
=========================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from dataset import build_transforms  # noqa: E402
from train import load_checkpoint  # noqa: E402  (tolerates timm- or torchvision-built ckpts)

CAVEATS = [
    "Coarse category, NOT a diagnosis — several source folders bundle multiple diseases.",
    "Trained on public-atlas images with NO skin-tone labels; reliability across skin "
    "tones is unmeasured.",
    "Not a validated dermatology model. Triage + education only; no treatment guidance.",
]





def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--device")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args()

    device = config.resolve_device(a.device)
    model, ckpt = load_checkpoint(Path(a.checkpoint), device)
    tf = build_transforms(ckpt["img_size"], train=False)
    classes = ckpt["classes"]
    k = min(a.topk, len(classes))

    if not a.json:
        print(f"checkpoint {a.checkpoint}\n  {ckpt['model_name']} · {len(classes)} classes · "
              f"{ckpt['img_size']}px · label_column={ckpt.get('label_column', '?')} · "
              f"val_macro_f1={ckpt.get('val_macro_f1', float('nan')):.3f}\n  device {device}")

    out = []
    for img_path in a.images:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:  # noqa: BLE001
            print(f"\n{img_path}\n  ERROR unreadable: {type(e).__name__}: {e}")
            continue
        with torch.no_grad():
            probs = model(tf(img).unsqueeze(0).to(device)).softmax(dim=1)[0].cpu()
        top = probs.topk(k)
        preds = [{"category": classes[i], "confidence": round(float(p), 4)}
                 for p, i in zip(top.values, top.indices)]
        out.append({"image": img_path, "top_k": preds, "caveats": CAVEATS})
        if not a.json:
            print(f"\n{img_path}")
            for r in preds:
                bar = "█" * max(1, round(r["confidence"] * 30))
                print(f"  {r['confidence']:6.1%}  {bar:<30} {r['category']}")

    if a.json:
        print(json.dumps(out, indent=2))
    else:
        print("\n" + "\n".join(f"NOTE: {c}" for c in CAVEATS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
