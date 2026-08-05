"""ml/evaluate.py
=========================================================================
Reads a checkpoint and produces an HONEST evaluation on the held-out test split —
the numbers you actually need before deciding whether a model is worth anything,
rather than a single headline accuracy.

    ml/.venv/bin/python ml/evaluate.py data/ml/kaggle_out/out/runs/best.pt
    ml/.venv/bin/python ml/evaluate.py best.pt --split val --limit-per-class 20

Writes eval_report.md + eval_report.json next to the checkpoint, and prints a
summary. What it reports and why:

  per-class P/R/F1 + support   a 6:1 skewed dataset hides tail-class failure behind
                               a decent overall accuracy
  top-1 vs top-3              the product surfaces a shortlist, so top-3 recall is
                               the number that maps to the actual use
  worst confusion pairs       tells you WHICH categories collapse into each other;
                               some pairs are the atlas's fault (folders that bundle
                               several diseases), not the model's
  confidently wrong           high-confidence errors are the dangerous ones for
                               triage — a wrong answer stated firmly is worse than
                               an uncertain one
  accuracy by map_kind        whether the model does better on folders that map
                               cleanly to the triage vocabulary (`direct`) than on
                               coarse buckets — a check on label noise, not skill
  can't-miss recall           recall on the malignancy-ish folders specifically,
                               because missing those is the costly direction

NO per-skin-tone breakdown appears here, and that is deliberate: this corpus has no
skin-tone labels, so any such number would be fabricated.
=========================================================================
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from dataset import ManifestImageDataset  # noqa: E402
from train import load_checkpoint  # noqa: E402  (one shared, backend-tolerant loader)

# Folders whose misses are the expensive direction (referral, not reassurance).
CANT_MISS_SUBSTRINGS = ("malignant", "melanoma", "carcinoma")





def map_kinds(manifest_dir: Path) -> dict[str, str]:
    """original_class -> map_kind, read from the manifest the model was trained on."""
    out: dict[str, str] = {}
    f = Path(manifest_dir) / "manifest.csv"
    if not f.exists():
        return out
    with f.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r.get("original_class", ""), r.get("map_kind", ""))
    return out


@torch.no_grad()
def collect(model, loader, device, n_classes):
    conf = torch.zeros(n_classes, n_classes, dtype=torch.long)
    top3_hit = Counter()
    per_class_total = Counter()
    conf_right, conf_wrong, confidently_wrong = [], [], 0
    for x, y in loader:
        x = x.to(device)
        probs = model(x).softmax(dim=1).cpu()
        k = min(3, n_classes)
        top = probs.topk(k, dim=1)
        for row, t in zip(range(len(y)), y.tolist()):
            pred = int(top.indices[row, 0])
            p = float(top.values[row, 0])
            conf[t, pred] += 1
            per_class_total[t] += 1
            if t in top.indices[row].tolist():
                top3_hit[t] += 1
            if pred == t:
                conf_right.append(p)
            else:
                conf_wrong.append(p)
                if p >= 0.70:
                    confidently_wrong += 1
    return conf, top3_hit, per_class_total, conf_right, conf_wrong, confidently_wrong


def prf(conf: torch.Tensor):
    tp = conf.diag().float()
    support = conf.sum(1).float()
    predicted = conf.sum(0).float()
    rec = torch.where(support > 0, tp / support.clamp(min=1), torch.zeros_like(tp))
    pre = torch.where(predicted > 0, tp / predicted.clamp(min=1), torch.zeros_like(tp))
    f1 = torch.where((pre + rec) > 0, 2 * pre * rec / (pre + rec).clamp(min=1e-9),
                     torch.zeros_like(tp))
    return pre, rec, f1, support


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--manifest", help="dir with manifest.csv + classes.json")
    ap.add_argument("--root", help="dataset root for relative image paths")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit-per-class", type=int)
    ap.add_argument("--device")
    a = ap.parse_args()

    ckpt_path = Path(a.checkpoint).resolve()
    device = config.resolve_device(a.device)
    model, ck = load_checkpoint(ckpt_path, device)
    classes = ck["classes"]
    n = len(classes)

    manifest_dir = Path(a.manifest) if a.manifest else config.out_dir() / "dermnet"
    root = config.dataset_root(a.root)
    ds = ManifestImageDataset(manifest_dir, root, a.split, ck.get("label_column", "original_class"),
                             img_size=ck["img_size"], limit_per_class=a.limit_per_class)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=0)

    print(f"checkpoint  {ckpt_path}")
    print(f"  {ck['model_name']} · weights={ck.get('weight_source', 'unknown')} · "
          f"{n} classes · {ck['img_size']}px · trained epoch {ck.get('epoch', '?')}")
    print(f"  eval on split={a.split}  n={len(ds)}  device={device}\n")

    conf, top3_hit, totals, c_right, c_wrong, conf_wrong_n = collect(model, loader, device, n)
    pre, rec, f1, support = prf(conf)
    present = support > 0
    total = int(support.sum())
    top1 = int(conf.diag().sum())
    top3 = sum(top3_hit.values())

    kinds = map_kinds(manifest_dir)
    by_kind = defaultdict(lambda: [0, 0])
    for i, cls in enumerate(classes):
        k = kinds.get(cls, "unknown") or "unknown"
        by_kind[k][0] += int(conf[i].sum())
        by_kind[k][1] += int(conf[i, i])

    rows = sorted(
        ({"class": classes[i], "support": int(support[i]), "precision": float(pre[i]),
          "recall": float(rec[i]), "f1": float(f1[i]),
          "top3_recall": top3_hit[i] / max(totals[i], 1), "map_kind": kinds.get(classes[i], "")}
         for i in range(n) if support[i] > 0),
        key=lambda r: r["f1"])

    pairs = sorted(((int(conf[i, j]), classes[i], classes[j])
                    for i in range(n) for j in range(n)
                    if i != j and conf[i, j] > 0), reverse=True)[:10]

    cm_idx = [i for i, c in enumerate(classes)
              if any(s in c.lower() for s in CANT_MISS_SUBSTRINGS)]
    cm = {classes[i]: {"recall": float(rec[i]), "top3_recall": top3_hit[i] / max(totals[i], 1),
                       "support": int(support[i])} for i in cm_idx}

    summary = {
        "checkpoint": str(ckpt_path), "split": a.split, "n": total,
        "model": ck["model_name"], "weight_source": ck.get("weight_source", "unknown"),
        "top1": top1 / max(total, 1), "top3": top3 / max(total, 1),
        "macro_f1": float(f1[present].mean()), "balanced_acc": float(rec[present].mean()),
        "mean_conf_correct": sum(c_right) / max(len(c_right), 1),
        "mean_conf_wrong": sum(c_wrong) / max(len(c_wrong), 1),
        "confidently_wrong_ge_0.70": conf_wrong_n,
        "confidently_wrong_rate": conf_wrong_n / max(total, 1),
        "per_class": rows, "worst_confusions": pairs,
        "by_map_kind": {k: {"n": v[0], "top1": v[1] / max(v[0], 1)} for k, v in sorted(by_kind.items())},
        "cant_miss": cm,
        "skin_tone_breakdown": None,
        "skin_tone_note": "Not computable: this corpus has no skin-tone labels.",
    }

    out_json = ckpt_path.parent / "eval_report.json"
    out_json.write_text(json.dumps(summary, indent=2))
    (ckpt_path.parent / "eval_report.md").write_text(_md(summary))

    print(f"top-1 {summary['top1']:.3f}   top-3 {summary['top3']:.3f}   "
          f"macro-F1 {summary['macro_f1']:.3f}   balanced-acc {summary['balanced_acc']:.3f}")
    print(f"mean confidence: correct {summary['mean_conf_correct']:.3f} / "
          f"wrong {summary['mean_conf_wrong']:.3f}")
    print(f"confidently wrong (p>=0.70 and incorrect): {conf_wrong_n} "
          f"({summary['confidently_wrong_rate']:.1%})\n")
    print("weakest 5 classes by F1:")
    for r in rows[:5]:
        print(f"  {r['f1']:.3f} F1  {r['recall']:.3f} rec  {r['top3_recall']:.3f} top3  "
              f"n={r['support']:<4} [{r['map_kind']}] {r['class']}")
    print("\nstrongest 3:")
    for r in rows[-3:][::-1]:
        print(f"  {r['f1']:.3f} F1  {r['recall']:.3f} rec  n={r['support']:<4} {r['class']}")
    print("\ntop confusions (true -> predicted):")
    for c, t, p in pairs[:6]:
        print(f"  {c:<5} {t}  ->  {p}")
    print("\naccuracy by vocabulary map_kind:")
    for k, v in summary["by_map_kind"].items():
        print(f"  {k:<14} n={v['n']:<5} top1 {v['top1']:.3f}")
    if cm:
        print("\ncan't-miss folders (missing these is the costly direction):")
        for c, v in cm.items():
            print(f"  recall {v['recall']:.3f}  top3 {v['top3_recall']:.3f}  n={v['support']}  {c}")
    print(f"\nWrote {out_json.name} + eval_report.md -> {ckpt_path.parent}")
    return 0


def _md(s: dict) -> str:
    lines = [
        f"# Evaluation — {s['model']} on `{s['split']}` (n={s['n']})", "",
        f"**weight_source:** `{s['weight_source']}`", "",
        "| metric | value |", "|---|---|",
        f"| top-1 | {s['top1']:.3f} |", f"| top-3 | {s['top3']:.3f} |",
        f"| macro-F1 | {s['macro_f1']:.3f} |", f"| balanced accuracy | {s['balanced_acc']:.3f} |",
        f"| mean confidence when correct | {s['mean_conf_correct']:.3f} |",
        f"| mean confidence when wrong | {s['mean_conf_wrong']:.3f} |",
        f"| confidently wrong (p≥0.70) | {s['confidently_wrong_ge_0.70']} "
        f"({s['confidently_wrong_rate']:.1%}) |", "",
        "## Per class (worst F1 first)", "",
        "| class | map_kind | n | precision | recall | F1 | top-3 recall |", "|---|---|---|---|---|---|---|",
    ]
    for r in s["per_class"]:
        lines.append(f"| {r['class']} | {r['map_kind']} | {r['support']} | {r['precision']:.3f} | "
                     f"{r['recall']:.3f} | {r['f1']:.3f} | {r['top3_recall']:.3f} |")
    lines += ["", "## Worst confusions (true → predicted)", ""]
    lines += [f"- {c} × `{t}` → `{p}`" for c, t, p in s["worst_confusions"]]
    lines += ["", "## Accuracy by vocabulary map_kind", "",
              "| map_kind | n | top-1 |", "|---|---|---|"]
    for k, v in s["by_map_kind"].items():
        lines.append(f"| {k} | {v['n']} | {v['top1']:.3f} |")
    lines += ["", "`direct` folders map cleanly onto the triage vocabulary; `coarse` bundle "
              "several diseases; `none` have no vocabulary equivalent. A gap between them is "
              "label noise, not model skill.", ""]
    if s["cant_miss"]:
        lines += ["## Can't-miss folders", "", "| folder | recall | top-3 recall | n |", "|---|---|---|---|"]
        for c, v in s["cant_miss"].items():
            lines.append(f"| {c} | {v['recall']:.3f} | {v['top3_recall']:.3f} | {v['support']} |")
        lines += ["", "Missing these is the expensive direction — a false reassurance costs more "
                  "than a false referral.", ""]
    lines += ["## Skin tone", "",
              "**Not reported, and not reportable.** " + s["skin_tone_note"] +
              " Any per-skin-tone figure for this model would be fabricated. The app's "
              "skin-tone reliability caveat therefore stands unchanged.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
