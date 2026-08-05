"""ml/train.py
=========================================================================
Trains a coarse skin-condition CATEGORY classifier from a pipeline manifest.
Same code path locally (CPU/MPS) and on Kaggle (CUDA) — device and paths come
from ml/config.py, nothing is machine-specific.

    python3 ml/train.py --smoke                 # tiny CPU slice, minutes
    python3 ml/train.py                         # full run (use Kaggle GPU)
    EPOCHS=20 MODEL=convnext_tiny python3 ml/train.py

Metrics are macro-averaged (macro-F1, balanced accuracy) plus top-3, because the
class distribution is skewed 6:1 and plain accuracy would flatter the head classes.
Top-3 is the triage-relevant number: the product surfaces a shortlist, not a
diagnosis.

Class imbalance is handled with inverse-frequency-weighted cross-entropy.

This is a research artifact. It does not make the app diagnostic: the model gives
coarse categories from atlas-labelled data with no skin-tone labels, so the app's
placeholder-engine notice and skin-tone reliability caveat both still apply.
=========================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from dataset import ManifestImageDataset  # noqa: E402


def _torchvision_pretrained(name: str, num_classes: int, weights="DEFAULT") -> nn.Module | None:
    """Same architecture via torchvision, whose weights come from download.pytorch.org.

    Returns None if torchvision has no such model or an unrecognised head layout.
    `weights` is a seam for tests: pass None to build the architecture without a
    network fetch and still exercise the head-swap logic below.
    """
    import torchvision.models as tvm
    fn = getattr(tvm, name, None)
    if fn is None or not callable(fn):
        return None
    m = fn(weights=weights)
    if isinstance(getattr(m, "fc", None), nn.Linear):                 # resnet family
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if isinstance(getattr(m, "head", None), nn.Linear):               # some vits
        m.head = nn.Linear(m.head.in_features, num_classes)
        return m
    head = getattr(m, "classifier", None)                             # convnext/efficientnet
    if isinstance(head, nn.Linear):
        m.classifier = nn.Linear(head.in_features, num_classes)
        return m
    if isinstance(head, nn.Sequential):
        for i in range(len(head) - 1, -1, -1):
            if isinstance(head[i], nn.Linear):
                head[i] = nn.Linear(head[i].in_features, num_classes)
                return m
    return None


def build_model(name: str, num_classes: int, pretrained: bool) -> tuple[nn.Module, str]:
    """Returns (model, weight_source) so the report records where weights came from.

    timm is preferred, but it fetches pretrained weights from HuggingFace, which fails
    intermittently inside the Kaggle notebook image ("Cannot send a request, as the
    client has been closed"). torchvision fetches from download.pytorch.org, so it is
    a genuinely independent path rather than a retry of the same one. Training from
    scratch is the last resort and is reported loudly — it materially changes results,
    so it must never happen silently.
    """
    import timm
    if not pretrained:
        return timm.create_model(name, pretrained=False, num_classes=num_classes), "scratch"

    try:
        return timm.create_model(name, pretrained=True, num_classes=num_classes), "timm_hf"
    except Exception as e:  # noqa: BLE001 — network/hub failures come in many shapes
        print(f"  WARNING timm pretrained weights unavailable ({type(e).__name__}: "
              f"{str(e)[:120]}); trying torchvision")

    try:
        m = _torchvision_pretrained(name, num_classes)
        if m is not None:
            print("  using torchvision pretrained weights (download.pytorch.org)")
            return m, "torchvision"
        print(f"  WARNING torchvision has no usable '{name}'")
    except Exception as e:  # noqa: BLE001
        print(f"  WARNING torchvision pretrained failed ({type(e).__name__}: {str(e)[:120]})")

    print("  WARNING >>> falling back to RANDOM INIT. Expect substantially worse results; "
          "weight_source=scratch is recorded in report.json.")
    return timm.create_model(name, pretrained=False, num_classes=num_classes), "scratch"


def load_checkpoint(path, device: str):
    """Rebuild a model from a checkpoint written by this script. Returns (model, ckpt).

    A checkpoint may have been built by timm OR by the torchvision fallback, and their
    state_dict key layouts differ for some architectures. Try timm, then torchvision
    (architecture only — the weights come from the checkpoint), so a checkpoint is
    loadable regardless of which backend produced it. Shared by evaluate.py/predict.py
    so there is exactly one loader to keep correct.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    for k in ("classes", "model_name", "img_size"):
        if k not in ck:
            raise SystemExit(f"{path} is missing '{k}' — not a checkpoint from ml/train.py")
    import timm
    try:
        m = timm.create_model(ck["model_name"], pretrained=False, num_classes=len(ck["classes"]))
        m.load_state_dict(ck["model"])
    except Exception:                                   # noqa: BLE001
        m = _torchvision_pretrained(ck["model_name"], len(ck["classes"]), weights=None)
        if m is None:
            raise
        m.load_state_dict(ck["model"])
    return m.eval().to(device), ck


def _loaders(manifest_dir, root, label_column, img_size, batch_size, workers, limit):
    ds = {}
    for split in ("train", "val", "test"):
        try:
            ds[split] = ManifestImageDataset(manifest_dir, root, split, label_column,
                                             img_size=img_size, limit_per_class=limit)
        except SystemExit as e:
            print(f"  ({split}: {e})")
    loaders = {
        s: DataLoader(d, batch_size=batch_size, shuffle=(s == "train"),
                      num_workers=workers, pin_memory=False, drop_last=False)
        for s, d in ds.items()
    }
    return ds, loaders


@torch.no_grad()
def evaluate(model, loader, device, num_classes) -> dict:
    model.eval()
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long)
    top1 = top3 = total = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += crit(out, y).item() * y.size(0)
        k = min(3, num_classes)
        topk = out.topk(k, dim=1).indices
        pred = topk[:, 0]
        top1 += (pred == y).sum().item()
        top3 += (topk == y.unsqueeze(1)).any(dim=1).sum().item()
        total += y.size(0)
        for t, p in zip(y.cpu().tolist(), pred.cpu().tolist()):
            conf[t, p] += 1

    tp = conf.diag().float()
    support = conf.sum(1).float()
    predicted = conf.sum(0).float()
    recall = torch.where(support > 0, tp / support.clamp(min=1), torch.zeros_like(tp))
    precision = torch.where(predicted > 0, tp / predicted.clamp(min=1), torch.zeros_like(tp))
    f1 = torch.where((precision + recall) > 0,
                     2 * precision * recall / (precision + recall).clamp(min=1e-9),
                     torch.zeros_like(tp))
    present = support > 0
    return {
        "loss": loss_sum / max(total, 1),
        "top1": top1 / max(total, 1),
        "top3": top3 / max(total, 1),
        "macro_f1": f1[present].mean().item() if present.any() else 0.0,
        "balanced_acc": recall[present].mean().item() if present.any() else 0.0,
        "n": total,
        "confusion": conf.tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="dir holding manifest.csv + classes.json")
    ap.add_argument("--root", help="dataset root for relative image paths")
    ap.add_argument("--out", help="where checkpoints/reports go")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU-friendly run that asserts the pipeline trains")
    ap.add_argument("--limit-per-class", type=int)
    ap.add_argument("--model"), ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int), ap.add_argument("--img-size", type=int)
    ap.add_argument("--lr", type=float), ap.add_argument("--label-column")
    ap.add_argument("--device"), ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args()

    root = config.dataset_root(a.root)
    manifest_dir = Path(a.manifest) if a.manifest else config.out_dir() / "dermnet"
    # smoke artifacts go to their own dir so a sanity run never clobbers a real
    # run's best.pt / report.json
    out = Path(a.out) if a.out else config.out_dir() / ("runs_smoke" if a.smoke else "runs")

    # Smoke defaults: small model, small images, one epoch, no weight download.
    if a.smoke:
        model_name = a.model or "resnet18"
        epochs, batch_size, img_size = 1, 8, 96
        limit = a.limit_per_class or 8
        pretrained, workers = False, 0
        device = a.device or "cpu"
    else:
        model_name = config.setting("model", a.model)
        epochs = config.setting("epochs", a.epochs)
        batch_size = config.setting("batch_size", a.batch_size)
        img_size = config.setting("img_size", a.img_size)
        limit = a.limit_per_class
        pretrained = False if a.no_pretrained else config.setting("pretrained")
        workers = config.setting("num_workers")
        device = config.resolve_device(a.device)

    lr = config.setting("lr", a.lr)
    label_column = config.setting("label_column", a.label_column)
    seed = config.setting("seed")
    torch.manual_seed(seed)

    print(f"manifest {manifest_dir}\nroot     {root}\ndevice   {device}  "
          f"model {model_name} (pretrained={pretrained})\n"
          f"epochs {epochs}  batch {batch_size}  img {img_size}  lr {lr}  "
          f"label_column {label_column}" + ("  [SMOKE]" if a.smoke else ""))

    ds, loaders = _loaders(manifest_dir, root, label_column, img_size, batch_size, workers, limit)
    if "train" not in loaders:
        print("FAIL no train split")
        return 1
    num_classes = len(ds["train"].classes)
    print(f"classes  {num_classes}  |  " +
          "  ".join(f"{s}={len(d)}" for s, d in ds.items()))

    model, weight_source = build_model(model_name, num_classes, pretrained)
    model = model.to(device)
    print(f"weights  {weight_source}")

    counts = torch.tensor(ds["train"].class_counts(), dtype=torch.float)
    weights = torch.where(counts > 0, counts.sum() / (counts.clamp(min=1) * num_classes),
                          torch.zeros_like(counts)).to(device)
    crit = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=config.setting("weight_decay"))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    out.mkdir(parents=True, exist_ok=True)
    history, best_f1, first_loss, last_loss = [], -1.0, None, None

    for epoch in range(1, epochs + 1):
        model.train()
        t0, seen, running = time.time(), 0, 0.0
        for step, (x, y) in enumerate(loaders["train"], 1):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            if not torch.isfinite(loss):
                print(f"FAIL non-finite loss at epoch {epoch} step {step}")
                return 1
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
            seen += y.size(0)
            if first_loss is None:
                first_loss = loss.item()
            last_loss = loss.item()
            if step % 50 == 0:
                print(f"  e{epoch} step {step} loss {running / seen:.4f}")
        sched.step()
        train_loss = running / max(seen, 1)

        rec = {"epoch": epoch, "train_loss": train_loss, "secs": round(time.time() - t0, 1)}
        if "val" in loaders:
            val = evaluate(model, loaders["val"], device, num_classes)
            rec["val"] = {k: v for k, v in val.items() if k != "confusion"}
            print(f"  e{epoch} train_loss {train_loss:.4f}  val top1 {val['top1']:.3f} "
                  f"top3 {val['top3']:.3f} macroF1 {val['macro_f1']:.3f} ({rec['secs']}s)")
            if val["macro_f1"] > best_f1:
                best_f1 = val["macro_f1"]
                torch.save({"model": model.state_dict(), "classes": ds["train"].classes,
                            "model_name": model_name, "img_size": img_size,
                            "label_column": label_column, "epoch": epoch,
                            "val_macro_f1": best_f1, "weight_source": weight_source},
                           out / "best.pt")
        else:
            print(f"  e{epoch} train_loss {train_loss:.4f} ({rec['secs']}s)")
        history.append(rec)

    report = {"model": model_name, "weight_source": weight_source,
              "epochs": epochs, "img_size": img_size,
              "batch_size": batch_size, "lr": lr, "seed": seed,
              "label_column": label_column, "num_classes": num_classes,
              "device": device, "smoke": a.smoke, "history": history,
              "sizes": {s: len(d) for s, d in ds.items()},
              "caveats": [
                  "Trained on public-atlas folder labels — coarse categories, not diagnoses.",
                  "No skin-tone labels in this corpus: no per-skin-tone fairness metric "
                  "can be reported, so the app's skin-tone reliability caveat still stands.",
                  "Not a validated dermatology model; triage + education only.",
              ]}
    if "test" in loaders and not a.smoke:
        report["test"] = evaluate(model, loaders["test"], device, num_classes)
        t = report["test"]
        print(f"\nTEST top1 {t['top1']:.3f}  top3 {t['top3']:.3f}  "
              f"macroF1 {t['macro_f1']:.3f}  balAcc {t['balanced_acc']:.3f}  n={t['n']}")
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out / 'report.json'}" + ("" if best_f1 < 0 else f" + best.pt (macroF1 {best_f1:.3f})"))

    if a.smoke:
        ok = first_loss is not None and last_loss is not None and \
            all(map(torch.isfinite, map(torch.tensor, [first_loss, last_loss])))
        print(f"\n{'PASS' if ok else 'FAIL'}  smoke: forward+backward ran, "
              f"loss {first_loss:.4f} -> {last_loss:.4f}, finite")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
