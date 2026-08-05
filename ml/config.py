"""ml/config.py
=========================================================================
Single source of truth for PATHS and HYPERPARAMS so the exact same code runs:

  locally   DERMNET_DIR=./data                      (repo checkout, gitignored)
  Kaggle    DERMNET_DIR=/kaggle/input/dermnet       (read-only input mount)

Resolution order for every setting:  explicit argument > env var > config.json > default.
Nothing is hardcoded to one machine; `python3 ml/config.py` prints what resolved.

WHY a separate out_dir: on Kaggle the dataset mount is READ-ONLY, so manifests and
checkpoints must go to /kaggle/working. Locally they go to data/ml/ (gitignored).
=========================================================================
"""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Optional file-based overrides (env still wins). Committed example: config.example.json
_CONFIG_JSON = Path(__file__).resolve().parent / "config.json"

# Candidate dataset roots, tried in order when DERMNET_DIR is unset. A candidate is
# accepted only if it actually contains the DermNet class-folder layout (see
# find_split_dirs), so a stale path never silently wins.
_CANDIDATE_ROOTS = [
    REPO_ROOT / "data",
    Path("/kaggle/input/dermnet"),
    Path("/kaggle/input/dermnet-skin-disease-images"),
    Path("/kaggle/input/skin-diseases-image-dataset"),
]

# Directory names DermNet ships. `val` is CARVED from train by the adapter (DermNet
# has no official val split), so it is not expected on disk.
TRAIN_DIRNAMES = ("train", "Train", "TRAIN")
TEST_DIRNAMES = ("test", "Test", "TEST")

DEFAULTS = {
    # --- model / optimisation (used by train.py) ---
    "model": "resnet50",
    "pretrained": True,
    "img_size": 224,
    "batch_size": 32,
    "epochs": 12,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "num_workers": 4,
    "seed": 42,               # matches console/lib/export.js SEED — one seed repo-wide
    "val_fraction": 0.15,     # carved out of DermNet's train/ split
    "label_column": "original_class",  # or "confirmed_diagnosis" (triage-vocab mapping)
    "device": "auto",
}


def _file_config() -> dict:
    if _CONFIG_JSON.exists():
        try:
            return json.loads(_CONFIG_JSON.read_text())
        except json.JSONDecodeError as e:
            raise SystemExit(f"ml/config.json is not valid JSON: {e}") from e
    return {}


def is_kaggle() -> bool:
    return bool(os.getenv("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/input").exists()


def find_split_dirs(root: Path) -> tuple[Path, Path] | None:
    """Locate (train_dir, test_dir) under `root`, tolerating one level of nesting.

    Kaggle mirrors of DermNet vary: sometimes /kaggle/input/<ds>/train, sometimes
    /kaggle/input/<ds>/dermnet/train. Returns None if neither shape is present.
    """
    if not root.is_dir():
        return None

    def pick(base: Path, names) -> Path | None:
        for n in names:
            if (base / n).is_dir():
                return base / n
        return None

    for base in [root, *sorted(p for p in root.iterdir() if p.is_dir())]:
        tr, te = pick(base, TRAIN_DIRNAMES), pick(base, TEST_DIRNAMES)
        if tr and te and any(p.is_dir() for p in tr.iterdir()):
            return tr, te
    return None


def dataset_root(explicit: str | Path | None = None) -> Path:
    """The DermNet root containing train/ and test/ class folders."""
    tried: list[str] = []
    for cand in [explicit, os.getenv("DERMNET_DIR"), _file_config().get("dermnet_dir"), *_CANDIDATE_ROOTS]:
        if not cand:
            continue
        p = Path(cand).expanduser()
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        tried.append(str(p))
        if find_split_dirs(p):
            return p
    raise SystemExit(
        "Could not find the DermNet dataset (a directory containing train/ and test/ "
        "class folders).\nTried:\n  " + "\n  ".join(tried or ["<nothing>"]) +
        "\nSet it explicitly:  DERMNET_DIR=/path/to/dermnet python3 ml/<script>.py"
    )


def out_dir(explicit: str | Path | None = None) -> Path:
    """Writable output root for manifests, checkpoints and reports."""
    cand = explicit or os.getenv("DERMA_ML_OUT") or _file_config().get("out_dir")
    if cand:
        p = Path(cand).expanduser()
        p = p if p.is_absolute() else (REPO_ROOT / p).resolve()
    else:
        p = Path("/kaggle/working/ml") if is_kaggle() else REPO_ROOT / "data" / "ml"

    # Guard: the clinical export dir (dermatologist-confirmed cases) is a different
    # corpus with different consent + ground-truth rules. Never write atlas data there.
    clinical = (REPO_ROOT / "data" / "exports").resolve()
    if p.resolve() == clinical or clinical in p.resolve().parents:
        raise SystemExit(
            f"Refusing to write public-atlas ML artifacts into the clinical export dir ({clinical}).\n"
            "DermNet is public-atlas pretraining data and must stay separate from "
            "dermatologist-confirmed cases. Pick a different DERMA_ML_OUT."
        )
    return p


def setting(name: str, explicit=None):
    """One hyperparameter, resolved explicit > env > config.json > default.

    Env var name is the upper-cased setting (IMG_SIZE, BATCH_SIZE, EPOCHS, LR, ...).
    """
    if name not in DEFAULTS:
        raise KeyError(f"unknown setting {name!r}; known: {sorted(DEFAULTS)}")
    if explicit is not None:
        return explicit
    raw = os.getenv(name.upper())
    if raw is None:
        raw = _file_config().get(name)
    if raw is None:
        return DEFAULTS[name]
    default = DEFAULTS[name]
    if isinstance(default, bool):
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


def resolve_device(explicit: str | None = None) -> str:
    """'auto' -> cuda if available (Kaggle GPU), else mps (Apple), else cpu."""
    want = explicit or setting("device")
    if want != "auto":
        return want
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    print(f"is_kaggle      : {is_kaggle()}")
    try:
        root = dataset_root()
        print(f"dataset_root   : {root}")
        tr, te = find_split_dirs(root)
        print(f"  train dir    : {tr}  ({sum(1 for p in tr.iterdir() if p.is_dir())} classes)")
        print(f"  test dir     : {te}  ({sum(1 for p in te.iterdir() if p.is_dir())} classes)")
    except SystemExit as e:
        print(f"dataset_root   : UNRESOLVED\n{e}")
    print(f"out_dir        : {out_dir()}")
    print(f"device         : {resolve_device()}")
    for k in DEFAULTS:
        print(f"  {k:<14}: {setting(k)}")
