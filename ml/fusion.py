"""ml/fusion.py
=========================================================================
Image + clinical-metadata fusion model, and a harness that proves it works before
any real metadata exists.

    image --> backbone --------> [img_feat]
                                            concat --> head --> 23 logits
    metadata (60-d) --> MLP --> [meta_feat]

THE DESIGN PROBLEM
------------------
Metadata is usually ABSENT. DermNet has none; real users skip questions. A naive
fusion model learns to lean on metadata and then collapses when it is missing —
which would be worse than the image-only model we already have.

Two mechanisms prevent that:

  modality dropout   during training, the metadata vector is randomly replaced by
                     the all-missing vector (p=meta_dropout). The model is forced
                     to stay competent image-only, and treats metadata as a bonus
                     rather than a crutch.
  present-mask bits  metadata.py marks which fields were actually answered, so
                     "not asked" is distinguishable from "answered unknown".

VALIDATING WITHOUT DATA
-----------------------
We cannot yet measure a real gain: DermNet carries no metadata. So `--simulate`
generates synthetic metadata of CONTROLLED informativeness — each case gets fields
that agree with its true label with probability `--signal`, else random. That does
not prove a real-world gain; it proves (a) the plumbing is correct end to end and
(b) how much headroom metadata of a given quality would buy. Simulated numbers are
labelled as such everywhere and must never be quoted as model performance.

    ml/.venv/bin/python ml/fusion.py --self-test
    ml/.venv/bin/python ml/fusion.py --simulate --signal 0.7 --epochs 3
=========================================================================
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import metadata as md  # noqa: E402
from dataset import ManifestImageDataset  # noqa: E402


class FusionModel(nn.Module):
    """Backbone image features + metadata MLP -> joint classifier.

    `meta_dropout` is applied at TRAINING time only: with that probability the
    metadata vector is replaced by the all-missing encoding, so the model must
    remain accurate without it.
    """

    def __init__(self, backbone_name: str, num_classes: int, pretrained: bool = True,
                 meta_dim: int = md.FEATURE_DIM, meta_hidden: int = 64,
                 meta_dropout: float = 0.5):
        super().__init__()
        import timm
        # num_classes=0 -> timm returns pooled features instead of logits
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        img_dim = self.backbone.num_features

        self.meta = nn.Sequential(
            nn.Linear(meta_dim, meta_hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(meta_hidden, meta_hidden), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(img_dim + meta_hidden, num_classes),
        )
        self.meta_dropout = meta_dropout
        # the all-missing vector, kept on the right device automatically
        self.register_buffer("missing", torch.tensor(md.empty(), dtype=torch.float32))

    def apply_modality_dropout(self, meta: torch.Tensor) -> torch.Tensor:
        """Randomly blank each row's metadata to the all-missing vector (train only).

        Split out from forward() so it can be tested directly — testing it through
        forward() would compare train-mode against eval-mode BatchNorm/Dropout and
        measure the wrong thing.
        """
        if not self.training or self.meta_dropout <= 0:
            return meta
        drop = torch.rand(meta.size(0), device=meta.device) < self.meta_dropout
        return torch.where(drop.unsqueeze(1), self.missing.expand_as(meta), meta)

    def forward(self, img: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        meta = self.apply_modality_dropout(meta)
        return self.head(torch.cat([self.backbone(img), self.meta(meta)], dim=1))

    @torch.no_grad()
    def predict_image_only(self, img: torch.Tensor) -> torch.Tensor:
        """Inference path when no metadata was collected — the common case today."""
        meta = self.missing.expand(img.size(0), -1)
        return self.forward(img, meta)


class MetaDataset(ManifestImageDataset):
    """ManifestImageDataset that also returns the encoded metadata vector.

    Returns ((image, meta), label) so the training loop can stay generic.

    `simulate_signal` fabricates metadata for benchmarking only (see module docstring):
    with probability `signal` the fields are drawn from that class's characteristic
    profile, otherwise uniformly at random. Deterministic given `seed`.
    """

    def __init__(self, *a, simulate_signal: float | None = None, seed: int = 42, **kw):
        super().__init__(*a, **kw)
        self.simulate_signal = simulate_signal
        self.seed = seed

    def _simulated(self, idx: int, label: str) -> dict:
        rnd = random.Random(f"{self.seed}:{idx}")
        informative = rnd.random() < (self.simulate_signal or 0.0)
        profile = _CLASS_PROFILE.get(label)
        if not informative or profile is None:
            return {"body_site": rnd.choice(md.BODY_SITES),
                    "prior_steroid_use": rnd.choice(md.PRIOR_STEROID),
                    "duration": rnd.choice(md.DURATION),
                    "symptoms": rnd.choice(md.SYMPTOMS),
                    "recent_change": rnd.choice(md.RECENT_CHANGE)}
        return {k: (rnd.choice(v) if isinstance(v, list) else v) for k, v in profile.items()}

    def __getitem__(self, i: int):
        row = self.rows[i]
        img, y = super().__getitem__(i)
        src = self._simulated(i, row[self.label_column]) if self.simulate_signal else row
        return (img, torch.tensor(md.encode(src), dtype=torch.float32)), y


# Rough clinical profiles, used ONLY by --simulate. Not clinical guidance; they
# exist so the synthetic metadata has realistic structure rather than noise.
_CLASS_PROFILE = {
    "Tinea Ringworm Candidiasis and other Fungal Infections": {
        "body_site": ["Groin", "Foot", "Trunk"], "prior_steroid_use": ["yes", "unknown"],
        "duration": ["weeks", "months"], "symptoms": "itch", "recent_change": "spreading"},
    "Psoriasis pictures Lichen Planus and related diseases": {
        "body_site": ["Arm", "Leg", "Back", "Scalp"], "prior_steroid_use": ["no", "unknown"],
        "duration": ["months", "years"], "symptoms": "none", "recent_change": "same"},
    "Eczema Photos": {
        "body_site": ["Hand", "Arm", "Leg"], "prior_steroid_use": ["no", "yes"],
        "duration": ["months", "years"], "symptoms": "itch", "recent_change": "same"},
    "Atopic Dermatitis Photos": {
        "body_site": ["Arm", "Leg", "Neck"], "prior_steroid_use": ["no", "yes"],
        "duration": ["years"], "symptoms": "itch", "recent_change": "same"},
    "Acne and Rosacea Photos": {
        "body_site": ["Face", "Back", "Chest"], "prior_steroid_use": ["no"],
        "duration": ["months", "years"], "symptoms": "none", "recent_change": "same"},
    "Nail Fungus and other Nail Disease": {
        "body_site": "Nails", "prior_steroid_use": ["no", "unknown"],
        "duration": ["months", "years"], "symptoms": "none", "recent_change": "same"},
    "Hair Loss Photos Alopecia and other Hair Diseases": {
        "body_site": "Scalp", "prior_steroid_use": ["no", "unknown"],
        "duration": ["months", "years"], "symptoms": "none", "recent_change": "spreading"},
    "Urticaria Hives": {
        "body_site": ["Trunk", "Arm", "Leg"], "prior_steroid_use": ["no"],
        "duration": ["days"], "symptoms": "itch", "recent_change": "new"},
    "Scabies Lyme Disease and other Infestations and Bites": {
        "body_site": ["Hand", "Abdomen", "Groin"], "prior_steroid_use": ["no", "unknown"],
        "duration": ["weeks"], "symptoms": "itch", "recent_change": "spreading"},
}


def self_test() -> int:
    """Prove the plumbing without touching the dataset: shapes, missing-metadata
    robustness, and that metadata actually reaches the output."""
    torch.manual_seed(0)
    m = FusionModel("resnet18", num_classes=23, pretrained=False, meta_dropout=0.0)
    img = torch.zeros(4, 3, 224, 224)
    meta = torch.stack([torch.tensor(md.encode(
        {"body_site": "Groin", "prior_steroid_use": "yes", "symptoms": "itch"}),
        dtype=torch.float32)] * 4)
    m.eval()

    out = m(img, meta)
    assert out.shape == (4, 23), out.shape
    print(f"PASS  forward with metadata -> {tuple(out.shape)}")

    out_missing = m(img, m.missing.expand(4, -1))
    assert out_missing.shape == (4, 23)
    print(f"PASS  forward with ALL-MISSING metadata -> {tuple(out_missing.shape)}")

    out_img_only = m.predict_image_only(img)
    assert torch.allclose(out_img_only, out_missing, atol=1e-6)
    print("PASS  predict_image_only == all-missing path (image-only inference works)")

    delta = (out - out_missing).abs().max().item()
    assert delta > 1e-4, "metadata had no effect on the output — fusion is not wired"
    print(f"PASS  metadata changes the output (max logit delta {delta:.4f})")

    # test the dropout mechanism directly — through forward() we would be comparing
    # train-mode BatchNorm against eval-mode and measuring the wrong thing
    m.train()
    m.meta_dropout = 1.0
    assert torch.allclose(m.apply_modality_dropout(meta), m.missing.expand(4, -1))
    m.meta_dropout = 0.0
    assert torch.allclose(m.apply_modality_dropout(meta), meta)
    m.meta_dropout = 0.5
    m.eval()
    assert torch.allclose(m.apply_modality_dropout(meta), meta), "must not drop at eval time"
    print("PASS  modality dropout: p=1 blanks metadata, p=0 keeps it, eval never drops")

    # and it must actually fire at a realistic rate during training
    m.train(); m.meta_dropout = 0.5
    big = torch.stack([meta[0]] * 400)
    blanked = (m.apply_modality_dropout(big) == m.missing).all(dim=1).float().mean().item()
    assert 0.4 < blanked < 0.6, blanked
    print(f"PASS  p=0.5 blanks {blanked:.1%} of rows in training")
    m.eval()

    n_meta = sum(p.numel() for p in m.meta.parameters())
    n_back = sum(p.numel() for p in m.backbone.parameters())
    print(f"\nparams: backbone {n_back:,}  metadata branch {n_meta:,} "
          f"({n_meta / n_back:.2%} overhead)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--simulate", action="store_true",
                    help="benchmark with SYNTHETIC metadata (not real performance)")
    ap.add_argument("--signal", type=float, default=0.7,
                    help="probability a simulated case gets informative metadata")
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--img-size", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit-per-class", type=int, default=120)
    ap.add_argument("--meta-dropout", type=float, default=0.5)
    ap.add_argument("--device")
    a = ap.parse_args()

    if a.self_test or not a.simulate:
        return self_test()

    device = config.resolve_device(a.device)
    manifest_dir = config.out_dir() / "dermnet"
    root = config.dataset_root()
    print("*** SIMULATED METADATA — measures headroom, NOT real model performance ***")
    print(f"signal={a.signal}  meta_dropout={a.meta_dropout}  device={device}\n")

    ds = {s: MetaDataset(manifest_dir, root, s, "original_class", img_size=a.img_size,
                         limit_per_class=a.limit_per_class, simulate_signal=a.signal)
          for s in ("train", "val")}
    ld = {s: DataLoader(d, batch_size=a.batch_size, shuffle=(s == "train"), num_workers=0)
          for s, d in ds.items()}
    n_classes = len(ds["train"].classes)

    model = FusionModel(a.model, n_classes, pretrained=True,
                        meta_dropout=a.meta_dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        for (img, meta), y in ld["train"]:
            img, meta, y = img.to(device), meta.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(img, meta), y)
            loss.backward()
            opt.step()
            tot += loss.item() * y.size(0)

        model.eval()
        hits = {"with_meta": 0, "image_only": 0}
        n = 0
        with torch.no_grad():
            for (img, meta), y in ld["val"]:
                img, meta, y = img.to(device), meta.to(device), y.to(device)
                hits["with_meta"] += (model(img, meta).argmax(1) == y).sum().item()
                hits["image_only"] += (model.predict_image_only(img).argmax(1) == y).sum().item()
                n += y.size(0)
        print(f"  e{ep} loss {tot / len(ds['train']):.4f}  "
              f"val top1 WITH metadata {hits['with_meta'] / n:.3f}  "
              f"IMAGE-ONLY {hits['image_only'] / n:.3f}")

    print("\nThe gap between the two columns is what metadata of this quality buys.")
    print("Image-only must stay usable — that is what modality dropout protects.")
    print("*** Simulated. Real numbers require console cases with real metadata. ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
