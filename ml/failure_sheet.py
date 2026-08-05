"""ml/failure_sheet.py
=========================================================================
Contact sheets for the cases the urgency model gets wrong.

Aggregate metrics have taken us as far as they can. A recall of 0.946 does not say
whether the 56 misses are borderline lesions a dermatologist would also hesitate
over, or obvious ones that indicate a systematic blind spot. Only looking does.

Produces, for a given threshold:

  misses_<t>.png / .csv        urgent cases scored BELOW threshold — the dangerous
                               failures — grouped and sorted by true class
  false_alarms_<t>.png / .csv  the N highest-scoring routine cases — the cost side
                               of the trade

Each cell shows the image, its true class, the top-3 predicted classes with
calibrated probabilities, and the calibrated urgent score. The CSV carries the same
data plus the image path, so any case can be pulled up individually.

    ml/.venv/bin/python ml/failure_sheet.py data/ml/runs_final/best.pt --threshold 0.12
=========================================================================
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import runmeta  # noqa: E402
from dataset import ManifestImageDataset  # noqa: E402
from train import load_checkpoint  # noqa: E402
from urgency import GROUPS_YAML, Platt, Temperature, load_groups, logits_labels, tier_of  # noqa: E402

CELL = 190          # thumbnail edge
TEXT_H = 74         # caption strip under each thumbnail
COLS = 6
PAD = 6


def short(name: str, n: int = 26) -> str:
    return name if len(name) <= n else name[:n - 1] + "…"


def build_sheet(entries, title: str, out_path: Path, root: Path) -> None:
    """Compose a labelled grid. Entries are dicts with image_path/true/top3/score."""
    if not entries:
        out_path.write_bytes(b"")
        print(f"  (nothing to draw for {out_path.name})")
        return
    cols = min(COLS, len(entries))
    rows = (len(entries) + cols - 1) // cols
    cw, chh = CELL + PAD, CELL + TEXT_H + PAD
    W, H = cols * cw + PAD, rows * chh + PAD + 26

    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    d.text((PAD, 6), title, fill="black")

    for i, e in enumerate(entries):
        cx = PAD + (i % cols) * cw
        cy = 26 + PAD + (i // cols) * chh
        try:
            im = Image.open(root / e["image_path"]).convert("RGB")
            im.thumbnail((CELL, CELL))
            sheet.paste(im, (cx + (CELL - im.width) // 2, cy + (CELL - im.height) // 2))
        except Exception:  # noqa: BLE001 — a broken file must not kill the sheet
            d.rectangle([cx, cy, cx + CELL, cy + CELL], outline="red")
            d.text((cx + 8, cy + CELL // 2), "unreadable", fill="red")

        ty = cy + CELL + 2
        d.text((cx, ty), f"urg={e['urgent_score']:.3f}", fill="black")
        d.text((cx, ty + 12), f"TRUE {short(e['true'], 24)}", fill="darkred")
        for j, (c, p) in enumerate(e["top3"]):
            d.text((cx, ty + 26 + j * 12), f"{p:.2f} {short(c, 24)}", fill="black")

    sheet.save(out_path)
    print(f"  wrote {out_path.name}  ({len(entries)} cases, {cols}x{rows})")


def write_csv(entries, path: Path) -> None:
    cols = ["urgent_score", "true", "pred1", "p1", "pred2", "p2", "pred3", "p3",
            "image_path", "case_id"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for e in entries:
            t = e["top3"] + [("", 0.0)] * (3 - len(e["top3"]))
            w.writerow([round(e["urgent_score"], 4), e["true"],
                        t[0][0], round(t[0][1], 4), t[1][0], round(t[1][1], 4),
                        t[2][0], round(t[2][1], 4), e["image_path"], e["case_id"]])
    print(f"  wrote {path.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--threshold", type=float, default=0.12)
    ap.add_argument("--n-false-alarms", type=int, default=50)
    ap.add_argument("--manifest"), ap.add_argument("--root"), ap.add_argument("--device")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out")
    a = ap.parse_args()

    ckpt = Path(a.checkpoint).resolve()
    device = config.resolve_device(a.device)
    model, ck = load_checkpoint(ckpt, device)
    classes = ck["classes"]
    lc = ck.get("label_column", "original_class")
    manifest_dir = Path(a.manifest) if a.manifest else config.out_dir() / "dermnet"
    root = config.dataset_root(a.root)
    out_dir = Path(a.out) if a.out else ckpt.parent / "failures"
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = load_groups(GROUPS_YAML)
    val = ManifestImageDataset(manifest_dir, root, "val", lc, img_size=ck["img_size"])
    te = ManifestImageDataset(manifest_dir, root, "test", lc, img_size=ck["img_size"])
    frozen = runmeta.freeze(manifest_dir, "test", te.rows)
    meta = runmeta.manifest(tool="failure_sheet.py", checkpoint=ckpt,
                            files={"groups": str(GROUPS_YAML)},
                            extra={"threshold": a.threshold, "frozen_test": frozen})
    runmeta.print_header(meta, frozen)

    lv, yv = logits_labels(model, DataLoader(val, batch_size=a.batch_size), device)
    lt, yt = logits_labels(model, DataLoader(te, batch_size=a.batch_size), device)
    temp = Temperature().fit(lv, yv)
    pv, pt = temp(lv).softmax(1), temp(lt).softmax(1)

    soon = torch.tensor([i for i, c in enumerate(classes)
                         if tier_of(groups, c)[0] == "soon"])
    platt = Platt().fit(pv[:, soon].sum(1), torch.isin(yv, soon))
    score = platt(pt[:, soon].sum(1))
    is_pos = torch.isin(yt, soon)

    def entry(i: int) -> dict:
        top = pt[i].topk(3)
        return {"case_id": te.rows[i]["case_id"], "image_path": te.rows[i]["image_path"],
                "true": classes[int(yt[i])], "urgent_score": float(score[i]),
                "top3": [(classes[int(j)], float(p)) for p, j in zip(top.values, top.indices)]}

    # --- misses: urgent, scored below threshold ---
    miss_idx = torch.nonzero((score < a.threshold) & is_pos).flatten().tolist()
    misses = sorted((entry(i) for i in miss_idx), key=lambda e: (e["true"], e["urgent_score"]))
    by_class = defaultdict(list)
    for e in misses:
        by_class[e["true"]].append(e)

    print(f"\nMISSES at threshold {a.threshold} — urgent cases scored below it: {len(misses)}")
    for c, es in sorted(by_class.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(es):>3}  {c}")
    build_sheet(misses, f"MISSES @ {a.threshold} — urgent cases the model scored low "
                        f"(n={len(misses)}), grouped by true class",
                out_dir / f"misses_{a.threshold}.png", root)
    write_csv(misses, out_dir / f"misses_{a.threshold}.csv")

    # --- false alarms: routine, highest scoring ---
    fa_idx = torch.nonzero(~is_pos).flatten()
    fa_sorted = fa_idx[score[fa_idx].argsort(descending=True)][:a.n_false_alarms].tolist()
    alarms = [entry(i) for i in fa_sorted]
    fa_by_class = defaultdict(int)
    for e in alarms:
        fa_by_class[e["true"]] += 1
    print(f"\nTOP {len(alarms)} FALSE ALARMS — routine cases scored highest:")
    for c, n in sorted(fa_by_class.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>3}  {c}")
    build_sheet(alarms, f"TOP {len(alarms)} FALSE ALARMS — routine cases with the "
                        f"highest urgent score", out_dir / f"false_alarms_{a.threshold}.png", root)
    write_csv(alarms, out_dir / f"false_alarms_{a.threshold}.csv")

    print(f"\nSheets in {out_dir}")
    print("Look at them. Aggregate metrics cannot tell you whether the misses are "
          "borderline\ncases a clinician would also hesitate over, or a systematic blind spot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
