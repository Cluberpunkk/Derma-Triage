"""ml/kaggle/launch.py
=========================================================================
Drives the whole Kaggle GPU run headlessly from this machine:

  1. uploads ml/*.py as a PRIVATE Kaggle dataset (the code bundle, ~30 KB)
  2. resolves a DermNet dataset slug to attach (or takes --dermnet <slug>)
  3. pushes a PRIVATE GPU kernel that runs ml/kaggle/run_dermnet.py
  4. polls until it finishes, streaming status
  5. downloads best.pt + report.json into data/ml/kaggle_out/

Credentials: read by the kaggle CLI from ~/.kaggle/access_token (current KGAT_ format)
or ~/.kaggle/kaggle.json (legacy). This script never reads, prints, or copies the
token, and nothing it creates is public — datasets and kernels are both private.

    ml/.venv/bin/python ml/kaggle/launch.py --check          # creds + dataset search only
    ml/.venv/bin/python ml/kaggle/launch.py --dry-run        # build payload, push nothing
    ml/.venv/bin/python ml/kaggle/launch.py                  # full run
    ml/.venv/bin/python ml/kaggle/launch.py --fetch-only     # re-download a finished run
=========================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent.parent
REPO = ML_DIR.parent
KAGGLE = str(ML_DIR / ".venv" / "bin" / "kaggle")
if not Path(KAGGLE).exists():
    KAGGLE = "kaggle"

CODE_DATASET = "derma-ml-code"
KERNEL_SLUG = "derma-dermnet-train"
FETCH_DIR = REPO / "data" / "ml" / "kaggle_out"

# Known DermNet mirrors on Kaggle, preferred order. --dermnet overrides.
DERMNET_CANDIDATES = [
    "shubhamgoel27/dermnet",
    "himanshusharma03/dermnet",
]


def run(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, **kw)


def require_creds(explicit_username: str | None = None) -> str:
    """Verify Kaggle auth and resolve the account username (needed for slugs).

    Supports BOTH credential formats, newest first:
      ~/.kaggle/access_token   new KGAT_… token (current Kaggle flow)
      KAGGLE_API_TOKEN         same token via env
      ~/.kaggle/kaggle.json    legacy {username, key} pair
    The token itself is never read or printed by this code — only its presence is
    checked, and the kaggle CLI reads it directly.
    """
    home = Path.home() / ".kaggle"
    new_token, legacy = home / "access_token", home / "kaggle.json"

    found = []
    for p in (new_token, legacy):
        if p.exists():
            mode = p.stat().st_mode & 0o777
            if mode & 0o077:
                print(f"  note: tightening permissions on {p} (was {oct(mode)})")
                p.chmod(0o600)
            found.append(str(p))
    if os.getenv("KAGGLE_API_TOKEN"):
        found.append("$KAGGLE_API_TOKEN")

    if not found:
        sys.exit(
            "No Kaggle credentials found.\n\n"
            "  1. kaggle.com/settings/api -> Create New Token  (copy the KGAT_… value)\n"
            "  2. mkdir -p ~/.kaggle && echo KGAT_yourtoken > ~/.kaggle/access_token \\\n"
            "       && chmod 600 ~/.kaggle/access_token\n\n"
            "Legacy ~/.kaggle/kaggle.json also works if you have one.\n"
            "Do NOT paste or screenshot the token into a chat — a token that has been "
            "shown anywhere should be revoked and regenerated."
        )
    print(f"  credentials: {', '.join(found)}")

    # Prove auth against the API and, where possible, learn the username from it.
    r = run([KAGGLE, "kernels", "list", "--mine", "--page-size", "1", "--csv"])
    blob = r.stdout + r.stderr
    if r.returncode != 0:
        sys.exit("kaggle CLI cannot authenticate. If you just revoked a token, generate a "
                 f"new one and rewrite ~/.kaggle/access_token.\n{blob.strip()}")

    username = explicit_username or os.getenv("KAGGLE_USERNAME")
    if not username and legacy.exists():
        try:
            username = json.loads(legacy.read_text()).get("username")
        except (json.JSONDecodeError, OSError):
            username = None
    if not username:
        # `kaggle config view` reports the username for OAUTH-style tokens too
        for line in run([KAGGLE, "config", "view"]).stdout.splitlines():
            if line.strip().startswith("- username:"):
                cand = line.split(":", 1)[1].strip()
                if cand and cand.lower() != "none":
                    username = cand
                break
    if not username:
        for line in blob.splitlines():
            ref = line.strip().split(",")[0]
            if "/" in ref and not ref.startswith("ref"):
                username = ref.split("/")[0]
                break
    if not username:
        sys.exit("Authenticated, but could not determine your Kaggle username (no existing "
                 "notebooks to infer it from).\nRe-run with --username <your-kaggle-username> "
                 "— it's the name in your profile URL, kaggle.com/<username>.")
    print(f"  account: {username}")
    return username


def resolve_dermnet(explicit: str | None) -> str:
    if explicit:
        return explicit
    r = run([KAGGLE, "datasets", "list", "-s", "dermnet", "--csv"])
    listed = r.stdout if r.returncode == 0 else ""
    for slug in DERMNET_CANDIDATES:
        if slug in listed:
            print(f"  dermnet dataset: {slug}")
            return slug
    if listed:
        print("  DermNet candidates from search:")
        for line in listed.splitlines()[1:8]:
            print(f"    {line.split(',')[0]}")
    sys.exit("Could not auto-pick a DermNet dataset. Re-run with --dermnet <owner/slug>.")


def push_code_dataset(username: str, dry: bool) -> str:
    slug = f"{username}/{CODE_DATASET}"
    tmp = Path(tempfile.mkdtemp(prefix="derma-code-"))
    for f in sorted(ML_DIR.glob("*.py")):
        shutil.copy(f, tmp / f.name)
    (tmp / "dataset-metadata.json").write_text(json.dumps(
        {"title": "derma ml code", "id": slug, "licenses": [{"name": "unknown"}]}, indent=2))
    files = sorted(p.name for p in tmp.glob("*.py"))
    print(f"  code bundle: {len(files)} files -> {slug} (PRIVATE)")
    if dry:
        print(f"    would push: {files}")
        return slug

    exists = run([KAGGLE, "datasets", "status", slug]).returncode == 0
    if exists:
        # no -d/--delete-old-versions: keeping history is cheap and non-destructive
        cmd = [KAGGLE, "datasets", "version", "-p", str(tmp), "-m", "update code"]
    else:
        # `datasets create` is PRIVATE by default; -u/--public would opt in. Never pass it.
        cmd = [KAGGLE, "datasets", "create", "-p", str(tmp)]
    r = run(cmd)
    out = (r.stdout + r.stderr).strip()
    print("   ", out.replace("\n", "\n    ") or "(no output)")
    if r.returncode != 0 and "already exists" not in out:
        sys.exit("code dataset push failed")
    return slug


def push_kernel(username: str, code_slug: str, dermnet_slug: str, dry: bool,
                model: str, epochs: int, batch: int, img: int) -> str:
    slug = f"{username}/{KERNEL_SLUG}"
    tmp = Path(tempfile.mkdtemp(prefix="derma-kernel-"))
    script = (ML_DIR / "kaggle" / "run_dermnet.py").read_text()
    header = ("import os\n"
              f"os.environ['MODEL'] = {model!r}\n"
              f"os.environ['EPOCHS'] = {str(epochs)!r}\n"
              f"os.environ['BATCH_SIZE'] = {str(batch)!r}\n"
              f"os.environ['IMG_SIZE'] = {str(img)!r}\n\n")
    (tmp / "run_dermnet.py").write_text(header + script)
    # Flags written as STRINGS to match `kaggle kernels init` exactly (booleans are also
    # accepted — verified by pulling the metadata back).
    #
    # IMPORTANT: Kaggle can store enable_gpu=true and STILL run the kernel with
    # "Accelerator: None". The usual cause is account-level: an unverified phone number,
    # or an exhausted weekly GPU quota — both silently downgrade the session to CPU
    # rather than failing the push. Metadata verification therefore is NOT sufficient;
    # run_dermnet.py asserts torch.cuda.is_available() at runtime, which is the only
    # check that catches this.
    (tmp / "kernel-metadata.json").write_text(json.dumps({
        "id": slug,
        "title": "derma dermnet train",
        "code_file": "run_dermnet.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": "",
        "dataset_sources": [code_slug, dermnet_slug],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }, indent=2))
    print(f"  kernel: {slug} (PRIVATE, GPU, internet on)")
    print(f"    inputs: {code_slug}, {dermnet_slug}")
    print(f"    train:  {model} epochs={epochs} batch={batch} img={img}")
    if dry:
        print("    --dry-run: nothing pushed")
        return slug
    r = run([KAGGLE, "kernels", "push", "-p", str(tmp)])
    out = (r.stdout + r.stderr).strip()
    print("   ", out.replace("\n", "\n    "))
    if r.returncode != 0:
        sys.exit("kernel push failed")
    return slug


def verify_kernel_gpu(slug: str) -> bool:
    """Pull the kernel's stored metadata back and confirm the accelerator stuck.

    Kaggle accepts a push whose enable_gpu it then ignores, and the only symptom is a
    slow run. Checking costs one API call; not checking cost 2h10m of CPU training.
    """
    tmp = Path(tempfile.mkdtemp(prefix="derma-verify-"))
    r = run([KAGGLE, "kernels", "pull", slug, "-p", str(tmp), "-m"])
    meta = tmp / "kernel-metadata.json"
    if r.returncode != 0 or not meta.exists():
        print("  WARNING could not verify accelerator setting; watch the first log lines")
        return True
    d = json.loads(meta.read_text())
    gpu = str(d.get("enable_gpu", "")).lower() in ("true", "1")
    net = str(d.get("enable_internet", "")).lower() in ("true", "1")
    print(f"  verified: enable_gpu={d.get('enable_gpu')!r}  enable_internet={d.get('enable_internet')!r}")
    if not gpu:
        print("  ERROR the kernel did NOT keep the GPU setting — it would train on CPU.\n"
              "  Open the notebook -> Settings -> Accelerator -> GPU T4 x2 and re-run.")
    if not net:
        print("  WARNING internet is off; pretrained weights will not download")
    return gpu


def poll(slug: str, timeout_min: int) -> str:
    print(f"\nPolling {slug} (up to {timeout_min} min). Kaggle queues before running.")
    deadline = time.time() + timeout_min * 60
    last = ""
    while time.time() < deadline:
        r = run([KAGGLE, "kernels", "status", slug])
        status = (r.stdout + r.stderr).strip()
        if status != last:
            print(f"  [{time.strftime('%H:%M:%S')}] {status}")
            last = status
        low = status.lower()
        if "complete" in low:
            return "complete"
        if "error" in low or "cancel" in low:
            return "error"
        time.sleep(30)
    return "timeout"


def fetch(slug: str) -> None:
    FETCH_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading outputs -> {FETCH_DIR}")
    r = run([KAGGLE, "kernels", "output", slug, "-p", str(FETCH_DIR)])
    print("  " + (r.stdout + r.stderr).strip().replace("\n", "\n  "))
    got = sorted(p for p in FETCH_DIR.rglob("*") if p.is_file())
    for p in got:
        print(f"  {p.relative_to(FETCH_DIR)}  ({p.stat().st_size / 1e6:.1f} MB)")
    report = next((p for p in got if p.name == "report.json"), None)
    if report:
        d = json.loads(report.read_text())
        print(f"\n  model {d.get('model')} · {d.get('num_classes')} classes · "
              f"device {d.get('device')}")
        for h in d.get("history", []):
            v = h.get("val", {})
            print(f"    e{h['epoch']:<3} train_loss {h['train_loss']:.4f}  "
                  f"val top1 {v.get('top1', 0):.3f} top3 {v.get('top3', 0):.3f} "
                  f"macroF1 {v.get('macro_f1', 0):.3f}")
        t = d.get("test")
        if t:
            print(f"  TEST top1 {t['top1']:.3f}  top3 {t['top3']:.3f}  "
                  f"macroF1 {t['macro_f1']:.3f}  balAcc {t['balanced_acc']:.3f}  n={t['n']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dermnet", help="owner/slug of the DermNet dataset to attach")
    ap.add_argument("--username", help="your Kaggle username (kaggle.com/<username>)")
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--timeout-min", type=int, default=180)
    ap.add_argument("--check", action="store_true", help="verify creds + dataset, then stop")
    ap.add_argument("--dry-run", action="store_true", help="build payloads, push nothing")
    ap.add_argument("--fetch-only", action="store_true", help="download a finished run")
    a = ap.parse_args()

    print("Kaggle launcher — everything created is PRIVATE\n")
    username = require_creds(a.username)

    if a.fetch_only:
        fetch(f"{username}/{KERNEL_SLUG}")
        return 0

    dermnet = resolve_dermnet(a.dermnet)
    if a.check:
        print("\n--check passed: credentials work and a DermNet dataset is resolvable.")
        return 0

    code_slug = push_code_dataset(username, a.dry_run)
    kernel = push_kernel(username, code_slug, dermnet, a.dry_run,
                         a.model, a.epochs, a.batch_size, a.img_size)
    if a.dry_run:
        print("\n--dry-run complete: nothing was pushed to Kaggle.")
        return 0

    if not verify_kernel_gpu(kernel):
        print("\nAborting before the long wait — fix the accelerator, then re-run.")
        return 1

    # A freshly-created dataset needs a moment before a kernel can attach it.
    print("\n  waiting 60s for the code dataset to become attachable…")
    time.sleep(60)

    state = poll(kernel, a.timeout_min)
    print(f"\nkernel state: {state}")
    if state in ("complete", "error"):
        fetch(kernel)
    if state != "complete":
        print(f"\nInspect the run at: https://www.kaggle.com/code/{kernel}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
