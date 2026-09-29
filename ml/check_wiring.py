"""ml/check_wiring.py
=========================================================================
Fails when something PILOT-CRITICAL exists but nothing calls it.

This is the fourth instance of one pattern, and the first three each cost real work:

  1. surface #7 was ASSERTED not-shipping; it was live in App.jsx
  2. the codegen check compared the generator to its own output, so a shared error
     passed — the emergency tier silently returned tier "none" in the browser
  3. /api/capture existed with nothing calling it. The pilot would have run to
     completion and produced an EMPTY labels.csv
  4. SymptomFlow, saveShortCircuit and the withdrawal path: built, tested, unmounted

The common shape: a thing exists, a document says it works, and nobody checked that
anything reaches it. `ml/check_surfaces.py` catches the inverse (declared off but
reachable). This catches declared-built but unreachable.

    ml/.venv/bin/python ml/check_wiring.py          # report
    ml/.venv/bin/python ml/check_wiring.py --check  # exit 1 if a critical gap exists
=========================================================================
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = [p for p in (REPO / "src").rglob("*") if p.suffix in (".js", ".jsx")]
SERVER = [p for p in (REPO / "server").rglob("*") if p.suffix == ".js"]

# Things the PILOT depends on. Each must be referenced from somewhere other than the
# file that defines it, otherwise it is unreachable and the pilot silently loses that
# measurement.
PILOT_CRITICAL = {
    "SymptomFlow": "the questionnaire — without it there is no per-rule firing rate, "
                   "which is the pilot's PRIMARY measurable",
    "CaptureReview": "review-before-submit; without it bad frames enter the dataset",
    "saveHoldoutCapture": "writes the holdout CSV that ml/eval_holdout.py reads",
    "saveShortCircuit": "records emergency encounters, which take no photo and would "
                        "otherwise be the only rule whose behaviour is invisible",
    "newWithdrawalToken": "the only handle a participant has for DPDP withdrawal",
    "captureNative": "native-resolution reticle crop — the whole capture-path argument",
    # Added after a tester hit the consequence directly: MetadataForm was imported but
    # never rendered while the legacy form it replaced still rendered unconditionally
    # below the scan branch — and canRun REQUIRED the legacy fields, so every tester
    # was pulled past the new questionnaire into the old path.
    "MetadataForm": "the five clinical metadata fields, incl. self-reported tone and "
                    "prior_steroid_use — the steroid-modified-tinea field",
}

# Referenced only from a non-src caller, by design. Listed so the check does not
# nag about them, WITH the reason — an unexplained allowlist is how real gaps hide.
ALLOWED_UNCALLED = {
    "/api/capture/stats": "operations endpoint; read by curl and the console, not the app",
    "/api/capture/withdraw": "server endpoint; the participant-facing withdrawal UI is "
                             "NOT BUILT YET — this is a real gap, tracked below",
    "/api/capture/rejected": "nothing rejects while captureQuality.MODE is 'shadow'; "
                             "becomes live only when the gate starts enforcing",
    "questionList": "helper exported for a future flow; SymptomFlow renders from "
                    "SYMPTOM_RULES directly",
    "countBlobs": "module-internal helper called by measure() in the same file",
}

# Gaps we know about and have chosen not to close yet. Reported every run so they
# stay visible rather than becoming folklore.
KNOWN_GAPS = [
    ("participant withdrawal UI", "/api/capture/withdraw works; nothing in the app "
     "shows the token or offers a withdraw button"),
]


def defining_file(symbol: str) -> Path | None:
    # `async` and `default` may both appear before `function`; missing that gave a
    # false "NOT DEFINED" for saveHoldoutCapture on the first run of this check.
    pat = re.compile(
        rf"export\s+(default\s+)?(async\s+)?(function|const|class)\s+{re.escape(symbol)}\b"
        rf"|^\s*(async\s+)?function\s+{re.escape(symbol)}\b", re.M)
    for f in SRC + SERVER:
        if pat.search(f.read_text()):
            return f
    return None


def referrers(symbol: str, exclude: Path | None) -> list[str]:
    word = re.compile(rf"\b{re.escape(symbol)}\b")
    out = []
    for f in SRC + SERVER:
        if exclude and f == exclude:
            continue
        if word.search(f.read_text()):
            out.append(str(f.relative_to(REPO)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    print("PILOT-CRITICAL WIRING\n")
    print(f"  {'symbol':<22} {'refs':>4}  referenced by")
    violations, report = [], []
    for sym, why in PILOT_CRITICAL.items():
        d = defining_file(sym)
        refs = referrers(sym, d)
        report.append({"symbol": sym, "why": why,
                       "defined_in": str(d.relative_to(REPO)) if d else None,
                       "referenced_by": refs})
        if not d:
            violations.append(f"{sym}: NOT DEFINED anywhere")
            print(f"  {sym:<22} {'—':>4}  NOT DEFINED")
        elif not refs:
            violations.append(f"{sym}: defined in {d.relative_to(REPO)} but NOTHING "
                              f"references it\n      why it matters: {why}")
            print(f"  {sym:<22} {0:>4}  *** UNREACHABLE ***")
        else:
            print(f"  {sym:<22} {len(refs):>4}  {', '.join(refs)}")

    print("\nENDPOINTS defined in server/proxy.js\n")
    proxy = (REPO / "server" / "proxy.js").read_text()
    src_all = "\n".join(f.read_text() for f in SRC)
    eps = sorted(set(re.findall(r'app\.(?:get|post)\("([^"]+)"', proxy)))
    for ep in eps:
        called = ep in src_all
        note = "" if called else f"  <- not called from src ({ALLOWED_UNCALLED.get(ep, 'UNEXPLAINED')})"
        print(f"  {ep:<30} {'called' if called else 'DEAD  '}{note}")
        if not called and ep not in ALLOWED_UNCALLED:
            violations.append(f"{ep}: defined but never called, and not in the "
                              f"allowlist. Either wire it or explain it there.")

    print("\nKNOWN GAPS (chosen, not accidental)\n")
    for name, detail in KNOWN_GAPS:
        print(f"  - {name}: {detail}")

    if violations:
        print("\n=== VIOLATIONS ===")
        for v in violations:
            print(f"  {v}")
    else:
        print("\n  all pilot-critical modules are reachable")

    (REPO / "data" / "ml").mkdir(parents=True, exist_ok=True)
    (REPO / "data" / "ml" / "wiring_check.json").write_text(json.dumps(
        {"pilot_critical": report, "violations": violations,
         "known_gaps": [{"name": n, "detail": d} for n, d in KNOWN_GAPS]}, indent=2))

    if a.check and violations:
        print("\n--check FAILED: something pilot-critical is unreachable.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
