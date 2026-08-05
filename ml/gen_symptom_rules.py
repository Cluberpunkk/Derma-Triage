"""ml/gen_symptom_rules.py
=========================================================================
Generates src/lib/symptomRules.js from ml/symptom_gate.py.

The in-app questionnaire and the server-side gate must not diverge — a rule edited
in one place and not the other silently changes who gets escalated. So the browser
copy is GENERATED, never transcribed, and `--check` fails if it is stale (wire it
into CI alongside the tests).

    npm run gen:symptoms
    ml/.venv/bin/python ml/gen_symptom_rules.py --check
=========================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import symptom_gate as sg  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "src" / "lib" / "symptomRules.js"

TEMPLATE = '''/* src/lib/symptomRules.js — GENERATED. Do not edit by hand.
 *
 * Source of truth: ml/symptom_gate.py
 * Regenerate:      npm run gen:symptoms
 * Verify in CI:    ml/.venv/bin/python ml/gen_symptom_rules.py --check
 *
 * The in-app questionnaire and the server-side gate must never diverge, so the
 * rules are generated from the Python rather than transcribed. Escalate-only:
 * this can raise urgency and can never lower it, and it never touches the model.
 */

export const SYMPTOM_RULES = {rules};

export const ALL_SYMPTOM_FIELDS = {fields};

/** Derived from ml/symptom_gate.py TIER_ORDER — never hardcode a tier here. */
const TIER_ORDER = {tier_order};

/** Derived from ml/symptom_gate.py — never hardcode a message here. */
const TIER_MESSAGE = {tier_messages};

/** Tiers that short-circuit the whole flow: no photo, no model, no shortlist. */
export const SHORT_CIRCUIT_TIERS = {short_circuit};

/** Apply the gate to answers. Mirrors symptom_gate.score_row exactly. */
export function scoreSymptoms(answers = {{}}) {{
  const fired = [];
  let tier = "none";
  for (const r of SYMPTOM_RULES) {{
    const hits = r.fields.filter((f) => answers[f] === true);
    if (hits.length >= r.min_hits) {{
      fired.push({{ rule: r.id, tier: r.tier, hits }});
      if (TIER_ORDER[r.tier] > TIER_ORDER[tier]) tier = r.tier;
    }}
  }}
  return {{
    tier,
    rulesFired: fired,
    escalates: tier !== "none",
    shortCircuit: SHORT_CIRCUIT_TIERS.includes(tier),
    message: TIER_MESSAGE[tier] ?? "",
  }};
}}

/** Questions in ask-order, deduplicated across rules. */
export function questionList() {{
  const seen = new Set();
  const out = [];
  for (const r of SYMPTOM_RULES) {{
    for (const f of r.fields) {{
      if (seen.has(f)) continue;
      seen.add(f);
      out.push({{ field: f, text: r.questions[f] || f, group: r.id, tier: r.tier }});
    }}
  }}
  return out;
}}
'''


def render() -> str:
    rules = [{"id": r["id"], "tier": r["tier"], "min_hits": r["min_hits"],
              "fields": r["fields"], "questions": r["question_text"],
              "rationale": r["rationale"], "covers": r["covers"]} for r in sg.RULES]
    messages = {"emergency": "Call emergency services now.",
                "today": "Please seek care today.",
                "soon": "Please see a doctor soon.", "none": ""}
    return TEMPLATE.format(
        rules=json.dumps(rules, indent=2),
        fields=json.dumps(sorted(sg.ALL_FIELDS)),
        tier_order=json.dumps(sg.TIER_ORDER),
        tier_messages=json.dumps({k: v for k, v in messages.items() if k in sg.TIER_ORDER}),
        short_circuit=json.dumps(["emergency"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if the generated file is stale")
    a = ap.parse_args()
    new = render()
    if a.check:
        cur = OUT.read_text() if OUT.exists() else ""
        if cur != new:
            print(f"STALE: {OUT} does not match ml/symptom_gate.py. Run: npm run gen:symptoms")
            return 1
        print(f"OK: {OUT.name} matches ml/symptom_gate.py "
              f"({len(sg.RULES)} rule groups, {len(sg.ALL_FIELDS)} fields)")
        return 0
    OUT.write_text(new)
    print(f"generated {OUT} — {len(sg.RULES)} rule groups, {len(sg.ALL_FIELDS)} fields")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
