/* tests/conformance.mjs
 * ---------------------------------------------------------------------------
 * Runs the JS symptom gate against the HAND-WRITTEN conformance fixture.
 *
 * Why not just diff the JS against the Python: gen_symptom_rules.py --check does
 * that, and it PASSED while both were wrong — the generator template hardcoded a
 * tier list with no "emergency", so a fired emergency rule returned tier "none" in
 * the browser while Python said "emergency". A check that compares two artefacts
 * from the same generator cannot detect a shared error.
 *
 * The fixture is written by hand from intended behaviour, so it can.
 *
 *     node tests/conformance.mjs
 * ---------------------------------------------------------------------------
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const fixture = JSON.parse(readFileSync(path.join(here, "symptom_conformance.json"), "utf8"));
const { scoreSymptoms } = await import("../src/lib/symptomRules.js");

let failed = 0;
const seenTiers = new Set();

for (const c of fixture.cases) {
  const got = scoreSymptoms(c.answers);
  seenTiers.add(c.tier);
  const fired = got.rulesFired.map((r) => r.rule).sort();
  const want = [...c.rules_fired].sort();
  const problems = [];
  if (got.tier !== c.tier) problems.push(`tier ${got.tier} != ${c.tier}`);
  if (got.message !== c.message) problems.push(`message ${JSON.stringify(got.message)} != ${JSON.stringify(c.message)}`);
  if (Boolean(got.shortCircuit) !== c.short_circuit) problems.push(`shortCircuit ${got.shortCircuit} != ${c.short_circuit}`);
  if (JSON.stringify(fired) !== JSON.stringify(want)) problems.push(`rules ${JSON.stringify(fired)} != ${JSON.stringify(want)}`);
  if (problems.length) {
    failed++;
    console.log(`FAIL  ${c.name}\n      ${problems.join("\n      ")}`);
  }
}

// A fixture that never exercises a tier cannot catch a bug in it — which is exactly
// how the emergency tier shipped broken.
const missing = fixture.required_tier_coverage.filter((t) => !seenTiers.has(t));
if (missing.length) {
  failed++;
  console.log(`FAIL  no fixture case covers tier(s): ${missing.join(", ")}`);
}

console.log(`\nJS conformance: ${fixture.cases.length - failed}/${fixture.cases.length} cases, ` +
  `tiers covered: ${[...seenTiers].sort().join(", ")}`);
process.exit(failed ? 1 : 0);
