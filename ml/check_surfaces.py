"""ml/check_surfaces.py
=========================================================================
Verifies that every user-facing surface's declared status matches the code.

WHY THIS EXISTS
docs/ARCHITECTURE_decision.md enumerates seven surfaces and derives which blockers
bind from which are shipping. It ASSERTED that surface #7 (a user-facing urgency
verdict) was not shipping. It was, in src/App.jsx — and that assertion had been used
as the premise for the finding that Blocker 1 does not bind.

That is the third time in this project an assumption in a document was used as a
premise. So surface status is now:

  DECLARED   in src/lib/surfaces.js  (a flag the build actually reads)
  RENDERED   by named components/expressions in the app
  VERIFIED   here — a surface declared off must not be reachable

A surface marked "not shipping" whose rendering path is reachable in the build is a
FAILURE, not a note.

    ml/.venv/bin/python ml/check_surfaces.py          # report
    ml/.venv/bin/python ml/check_surfaces.py --check  # exit 1 on any violation
=========================================================================
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# For each surface: the flag that declares it, and the code patterns that RENDER it.
# A rendering pattern found OUTSIDE the flag guard means the surface is reachable.
SURFACES = [
    {
        "id": 7,
        "name": "user-facing urgency verdict (LLM path)",
        "flag": "SURFACE_7_URGENCY_VERDICT",
        "renders": [
            (r'src/App\.jsx', r'className=\{"uband " \+ result\.urgency\}'),
            (r'src/App\.jsx', r'result\.possible_conditions\.map'),
            (r'src/App\.jsx', r'result\.red_flags\.map'),
        ],
        "if_on": "Blocker 1 BINDS: 30.2% of benign pigmented lesions flagged at an "
                 "above-floor threshold, measured on a lighter-skinned corpus than "
                 "our users. The LLM path also has no calibration and no per-tone "
                 "characterisation at all.",
    },
    {
        "id": 6,
        "name": "top-3 shortlist from the trained model",
        "flag": "SURFACE_6_SHORTLIST",
        # The trained checkpoint is not wired to the UI at all, so these patterns
        # should find nothing. runTriage() is deliberately NOT listed here: it is the
        # LLM call, i.e. surface #7's data source, not a rendering of model output.
        "renders": [
            (r'src/.*\.jsx', r'shortlist|top_?3|confidence\s*[:=]'),
            (r'src/.*\.jsx', r'best\.pt|/api/predict'),
        ],
        "if_on": "Needs its own operating point and a stated abstention threshold.",
    },
]

FLAGS_FILE = REPO / "src" / "lib" / "surfaces.js"


def declared_flags() -> dict[str, bool]:
    txt = FLAGS_FILE.read_text()
    out = {}
    for m in re.finditer(r"export const (\w+)\s*=\s*(true|false)\s*;", txt):
        out[m.group(1)] = m.group(2) == "true"
    return out


def guarded(path: Path, pattern: str, flag: str) -> tuple[bool, list[int]]:
    """Is every match of `pattern` inside a block guarded by `flag`?

    Deliberately simple: we scan for the flag appearing above the match within the
    same JSX expression region. Crude, but it fails LOUD rather than silently, which
    is the property that matters — a false alarm costs a look, a false pass cost us a
    live urgency surface.
    """
    lines = path.read_text().splitlines()
    hits, unguarded = [], []
    for i, line in enumerate(lines):
        if re.search(pattern, line):
            hits.append(i + 1)
            window = "\n".join(lines[max(0, i - 40):i + 1])
            if flag not in window:
                unguarded.append(i + 1)
    return (len(unguarded) == 0, unguarded or hits)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    flags = declared_flags()
    print(f"declared flags ({FLAGS_FILE.relative_to(REPO)}):")
    for k, v in flags.items():
        print(f"  {k} = {v}")

    violations, report = [], []
    print("\nsurface verification:")
    for s in SURFACES:
        on = flags.get(s["flag"])
        if on is None:
            violations.append(f"surface #{s['id']}: flag {s['flag']} not declared")
            continue
        found_unguarded = []
        for glob_pat, pattern in s["renders"]:
            for f in sorted(REPO.glob("src/**/*.jsx")) + sorted(REPO.glob("src/**/*.js")):
                rel = str(f.relative_to(REPO))
                if not re.fullmatch(glob_pat, rel):
                    continue
                ok, lines_ = guarded(f, pattern, s["flag"])
                if not ok:
                    found_unguarded.append(f"{rel}:{','.join(map(str, lines_))} ({pattern})")
        status = ("ON" if on else "off")
        if not on and found_unguarded:
            violations.append(
                f"surface #{s['id']} ({s['name']}) is declared OFF but is REACHABLE:\n"
                + "\n".join(f"      {u}" for u in found_unguarded)
                + f"\n      if enabled: {s['if_on']}")
            verdict = "VIOLATION — reachable while declared off"
        elif on:
            verdict = f"ON — {s['if_on']}"
        else:
            verdict = "off and not reachable"
        report.append({"id": s["id"], "name": s["name"], "flag": s["flag"],
                       "declared": on, "unguarded_renders": found_unguarded,
                       "verdict": verdict})
        print(f"  #{s['id']:<2} {s['name'][:44]:<44} [{status:>3}] {verdict.splitlines()[0]}")

    if violations:
        print("\n=== VIOLATIONS ===")
        for v in violations:
            print(f"  {v}")
    else:
        print("\n  all surfaces match their declared status")

    (REPO / "data" / "ml").mkdir(parents=True, exist_ok=True)
    (REPO / "data" / "ml" / "surface_check.json").write_text(
        json.dumps({"flags": flags, "surfaces": report,
                    "violations": violations}, indent=2))

    if a.check and violations:
        print("\n--check FAILED: a surface declared 'not shipping' is reachable.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
