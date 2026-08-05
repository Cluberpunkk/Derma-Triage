"""Kaggle kernel entrypoint — runs headlessly on Kaggle's GPU.

Pushed by ml/kaggle/launch.py together with two attached inputs:
  /kaggle/input/<code-dataset>   the ml/ scripts (this repo)
  /kaggle/input/<dermnet>        the DermNet image folders

Everything it prints lands in the kernel log, which launch.py fetches. Outputs go
to /kaggle/working/out and are retrievable via `kaggle kernels output`.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
CODE = WORK / "ml"
OUT = WORK / "out"


def sh(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def find_code_dir() -> Path:
    """The attached code dataset: whichever input dir contains dermnet_adapter.py."""
    for cand in sorted(INPUT.rglob("dermnet_adapter.py")):
        return cand.parent
    sys.exit("FATAL: dermnet_adapter.py not found in /kaggle/input — is the code dataset attached?")


def find_dermnet_root() -> Path:
    """Whichever input dir directly contains train/ and test/ class folders."""
    candidates = []
    for train in INPUT.rglob("train"):
        if not train.is_dir():
            continue
        root = train.parent
        test = root / "test"
        if test.is_dir() and any(p.is_dir() for p in train.iterdir()):
            n = sum(1 for p in train.iterdir() if p.is_dir())
            candidates.append((n, root))
    if not candidates:
        sys.exit("FATAL: no DermNet-shaped input found (a dir with train/ and test/ "
                 "class folders). Attach the DermNet dataset.")
    candidates.sort(reverse=True)          # most classes wins
    return candidates[0][1]


def main() -> int:
    print("=" * 70)
    print("DermNet training kernel")
    print("=" * 70)
    print("inputs present:", [p.name for p in sorted(INPUT.iterdir())] if INPUT.is_dir() else "NONE")

    code_src = find_code_dir()
    CODE.mkdir(parents=True, exist_ok=True)
    for f in code_src.glob("*.py"):
        shutil.copy(f, CODE / f.name)
    print(f"code from {code_src} -> {CODE}: {sorted(p.name for p in CODE.glob('*.py'))}")

    root = find_dermnet_root()
    n_classes = sum(1 for p in (root / "train").iterdir() if p.is_dir())
    print(f"dermnet root: {root}  ({n_classes} train classes)")

    os.environ["DERMNET_DIR"] = str(root)
    os.environ["DERMA_ML_OUT"] = str(OUT)
    os.chdir(WORK)

    # Kaggle preinstalls timm. Reinstalling it upgrades huggingface_hub, whose HF
    # download path is broken in this image, so only install when it is missing.
    try:
        import timm  # noqa: F401
        print("timm already present — not reinstalling (avoids huggingface_hub churn)")
    except ImportError:
        if sh([sys.executable, "-m", "pip", "install", "-q", "timm"]) != 0:
            print("WARNING: timm install failed — is Internet enabled on this kernel?")

    # HARD GATE: a kernel whose accelerator was not actually enabled will happily train
    # on CPU for hours and then hit the session limit. Kaggle's UI reports this as
    # "Accelerator: None". Fail in seconds instead. Set ALLOW_CPU=1 to train on CPU
    # deliberately (e.g. a tiny slice).
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU OK: {torch.cuda.get_device_name(0)} "
                  f"(torch {torch.__version__}, cuda {torch.version.cuda})")
        elif os.environ.get("ALLOW_CPU") == "1":
            print("WARNING: no GPU, but ALLOW_CPU=1 — continuing on CPU (will be slow)")
        else:
            print("FATAL: no CUDA device. This kernel's accelerator is not enabled "
                  "(Kaggle shows 'Accelerator: None').\n"
                  "  Fix: Notebook -> Settings -> Accelerator -> GPU T4 x2, or push with\n"
                  "  kernel-metadata.json enable_gpu = \"true\" (a STRING, not a boolean).\n"
                  "  Set ALLOW_CPU=1 to train on CPU on purpose.")
            return 1
    except ImportError:
        print("FATAL: torch not importable in this kernel")
        return 1

    # Fail fast and loudly on a bad path, BEFORE spending GPU minutes.
    if sh([sys.executable, "ml/config.py"]) != 0:
        return 1

    if sh([sys.executable, "ml/dermnet_adapter.py"]) != 0:
        print("FATAL: manifest build failed")
        return 1

    train_cmd = [sys.executable, "ml/train.py",
                 "--model", os.environ.get("MODEL", "resnet50"),
                 "--epochs", os.environ.get("EPOCHS", "15"),
                 "--batch-size", os.environ.get("BATCH_SIZE", "64"),
                 "--img-size", os.environ.get("IMG_SIZE", "224")]
    rc = sh(train_cmd)

    print("\n" + "=" * 70)
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(OUT)}  ({p.stat().st_size / 1e6:.1f} MB)")
    print("=" * 70)
    print("DONE" if rc == 0 else f"TRAINING FAILED rc={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
