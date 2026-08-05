"""ml/runmeta.py
=========================================================================
Freezes the evaluation split and stamps every eval run with a versioned manifest,
so numbers produced weeks apart are actually comparable.

Two failure modes this prevents:

  silent split drift   rebuilding the manifest with a different dataset copy, seed
                       or dedup outcome changes which images are in `test`. Metrics
                       then move for reasons unrelated to the model. `split_hash`
                       makes that impossible to miss.
  silent config drift   a metric recorded without the checkpoint, group file and
                       thresholds that produced it cannot be reproduced or compared.

Every eval report carries the same header block, and every report keeps the
`fitzpatrick` field so it is visible in the output that skin tone is unknown and no
fairness claim is measurable from this corpus.
=========================================================================
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()


def split_hash(case_ids) -> str:
    """Order-independent fingerprint of exactly which cases are in a split."""
    joined = "\n".join(sorted(case_ids)).encode()
    return hashlib.sha256(joined).hexdigest()


def freeze(manifest_dir, split: str, rows) -> dict:
    """Record the frozen split, and verify it against a stored freeze if present.

    The freeze file lives beside the manifest so it travels with the dataset, not
    with a particular run.
    """
    ids = [r["case_id"] for r in rows]
    h = split_hash(ids)
    frozen_path = Path(manifest_dir) / f"frozen_{split}.json"

    record = {"split": split, "n": len(ids), "split_hash": h,
              "schema_version": SCHEMA_VERSION}

    if frozen_path.exists():
        prev = json.loads(frozen_path.read_text())
        record["previously_frozen"] = True
        record["matches_frozen"] = (prev.get("split_hash") == h)
        if not record["matches_frozen"]:
            record["WARNING"] = (
                f"{split} split CHANGED since it was frozen "
                f"(was n={prev.get('n')} {str(prev.get('split_hash'))[:12]}, "
                f"now n={len(ids)} {h[:12]}). Metrics are NOT comparable to earlier runs."
            )
    else:
        frozen_path.write_text(json.dumps(record, indent=2))
        record["previously_frozen"] = False
        record["matches_frozen"] = True
    return record


def manifest(*, tool: str, checkpoint, extra: dict | None = None,
             files: dict | None = None) -> dict:
    """Header block common to every eval report."""
    ck = Path(checkpoint)
    out = {
        "schema_version": SCHEMA_VERSION,
        "tool": tool,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": str(ck),
        "checkpoint_sha256": sha256_file(ck)[:16] if ck.exists() else None,
        # Kept in EVERY eval output on purpose: this corpus has no skin-tone labels,
        # so no per-tone metric exists and none can be inferred.
        "fitzpatrick": "unknown",
        "monk": "unknown",
        "fairness_measurable": False,
        "fairness_note": ("No skin-tone labels in this corpus. No fairness claim across "
                          "skin tones is measurable until Fitzpatrick-labelled data "
                          "(e.g. Fitzpatrick17k for training, DDI held out for testing) "
                          "is added."),
    }
    for name, p in (files or {}).items():
        p = Path(p)
        out[f"{name}_sha256"] = sha256_file(p)[:16] if p.exists() else None
    out.update(extra or {})
    return out


def print_header(m: dict, frozen: dict | None = None) -> None:
    print(f"run  {m['tool']}  ·  {m['generated_utc']}")
    print(f"  checkpoint {Path(m['checkpoint']).name} [{m['checkpoint_sha256']}]")
    if frozen:
        status = "frozen-match" if frozen.get("matches_frozen") else "!! SPLIT CHANGED !!"
        print(f"  {frozen['split']} split n={frozen['n']} "
              f"[{frozen['split_hash'][:12]}] {status}")
        if frozen.get("WARNING"):
            print(f"  WARNING: {frozen['WARNING']}")
    print(f"  fitzpatrick=unknown · fairness_measurable=False")
