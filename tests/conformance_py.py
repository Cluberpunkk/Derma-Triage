"""tests/conformance_py.py
=========================================================================
Runs the PYTHON symptom gate against the hand-written conformance fixture.

Paired with tests/conformance.mjs, which runs the JS gate against the SAME file.
Both must pass. Because the fixture is hand-written rather than generated, an error
shared by the Python source and the generated JS — the failure mode that shipped the
emergency tier broken — fails here instead of passing silently.

    ml/.venv/bin/python tests/conformance_py.py
=========================================================================
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

import symptom_gate as sg  # noqa: E402

fixture = json.loads((ROOT / "tests" / "symptom_conformance.json").read_text())
failed = 0
seen = set()

for c in fixture["cases"]:
    # score_row consumes the CSV/form representation, so booleans become "1"/"0"
    row = {k: ("1" if v else "0") for k, v in c["answers"].items()}
    got = sg.score_row(row)
    seen.add(c["tier"])
    fired = sorted(r["rule"] for r in got["rules_fired"])
    want = sorted(c["rules_fired"])
    problems = []
    if got["tier"] != c["tier"]:
        problems.append(f"tier {got['tier']} != {c['tier']}")
    if got["message"] != c["message"]:
        problems.append(f"message {got['message']!r} != {c['message']!r}")
    if bool(got.get("short_circuit")) != c["short_circuit"]:
        problems.append(f"short_circuit {got.get('short_circuit')} != {c['short_circuit']}")
    if fired != want:
        problems.append(f"rules {fired} != {want}")
    if problems:
        failed += 1
        print(f"FAIL  {c['name']}\n      " + "\n      ".join(problems))

missing = [t for t in fixture["required_tier_coverage"] if t not in seen]
if missing:
    failed += 1
    print(f"FAIL  no fixture case covers tier(s): {', '.join(missing)}")

print(f"\nPython conformance: {len(fixture['cases']) - failed}/{len(fixture['cases'])} "
      f"cases, tiers covered: {', '.join(sorted(seen))}")
sys.exit(1 if failed else 0)
