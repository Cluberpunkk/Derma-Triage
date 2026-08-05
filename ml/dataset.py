"""ml/dataset.py
=========================================================================
Torch Dataset over a pipeline manifest (from ml/dermnet_adapter.py, or the
clinical console export — same first 11 columns).

Two invariants worth stating, because both are easy to get wrong and silently
produce a model that scores well and means nothing:

1. `image_path` in the manifest is RELATIVE to the dataset root. The root is
   supplied at load time, so the identical manifest works with ./data locally and
   /kaggle/input/... on Kaggle.
2. The class index comes from classes.json (frozen by the adapter), NOT from the
   rows present in a given split. Deriving it per-split would give train and val
   different label spaces.

Rows whose split starts with `excluded` are never loaded.
=========================================================================
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_manifest(manifest_dir: Path) -> list[dict]:
    with (Path(manifest_dir) / "manifest.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_classes(manifest_dir: Path, label_column: str) -> list[str]:
    data = json.loads((Path(manifest_dir) / "classes.json").read_text())
    if label_column not in data:
        raise SystemExit(f"classes.json has no label column {label_column!r}; "
                         f"available: {sorted(data)}")
    return data[label_column]


def build_transforms(img_size: int, train: bool):
    """Augmentation kept conservative: lesion morphology and colour ARE the signal, so
    no aggressive colour jitter that would fabricate skin tones the data doesn't have."""
    from torchvision import transforms as T
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ManifestImageDataset(Dataset):
    def __init__(self, manifest_dir, dataset_root, split, label_column="original_class",
                 img_size=224, transform=None, limit_per_class=None):
        self.root = Path(dataset_root)
        self.label_column = label_column
        self.classes = read_classes(manifest_dir, label_column)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform if transform is not None else build_transforms(
            img_size, train=(split == "train"))

        rows = [r for r in read_manifest(manifest_dir)
                if r["split"] == split and not r["split"].startswith("excluded")]
        unknown = {r[label_column] for r in rows} - set(self.class_to_idx)
        if unknown:
            raise SystemExit(f"manifest has labels absent from classes.json: {sorted(unknown)}")

        if limit_per_class:
            kept, seen = [], {}
            for r in sorted(rows, key=lambda r: r["image_path"]):
                n = seen.get(r[label_column], 0)
                if n < limit_per_class:
                    kept.append(r)
                    seen[r[label_column]] = n + 1
            rows = kept

        self.rows = rows
        if not rows:
            raise SystemExit(f"no rows for split={split!r} in {manifest_dir}")

    def __len__(self) -> int:
        return len(self.rows)

    def class_counts(self) -> list[int]:
        counts = [0] * len(self.classes)
        for r in self.rows:
            counts[self.class_to_idx[r[self.label_column]]] += 1
        return counts

    def __getitem__(self, i: int):
        r = self.rows[i]
        img = Image.open(self.root / r["image_path"]).convert("RGB")
        return self.transform(img), self.class_to_idx[r[self.label_column]]
