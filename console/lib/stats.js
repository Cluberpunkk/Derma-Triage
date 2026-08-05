// console/lib/stats.js
// =========================================================================
// Data-quality dashboard aggregation, computed over GOLD (confirmed) labels so
// it reflects training-ready ground truth, not app guesses. Surfaces coverage
// gaps: too few dark-skin (V–VI) cases of a condition, "none"-confirmed fungal
// cases, low gradable %, etc.
// =========================================================================

import { db } from "../../server/lib/db.js";
import { goldRows } from "./export.js";
import { DIAGNOSES, FITZ_BAND, FITZ_BANDS, CONFIRMATION_METHODS } from "../shared/vocab.js";

export function dashboard() {
  const gold = goldRows();

  const totals = db().prepare(`
    SELECT
      (SELECT COUNT(*) FROM cases WHERE consent=1 AND withdrawn=0) AS cases,
      (SELECT COUNT(DISTINCT case_id) FROM labels) AS labelled,
      (SELECT COUNT(*) FROM labels WHERE needs_confirmation=1) AS needs_confirmation
  `).get();

  const byDiagnosis = {};
  const byBand = { "I–II": 0, "III–IV": 0, "V–VI": 0 };
  const byMethod = {};
  const matrix = {};            // diagnosis → { band → count }  (coverage grid)
  let gradableYes = 0;
  const gaps = [];

  for (const d of DIAGNOSES) { byDiagnosis[d.key] = 0; matrix[d.key] = { "I–II": 0, "III–IV": 0, "V–VI": 0 }; }
  for (const m of CONFIRMATION_METHODS) byMethod[m.key] = 0;

  for (const r of gold) {
    byDiagnosis[r.confirmed_diagnosis] = (byDiagnosis[r.confirmed_diagnosis] || 0) + 1;
    const band = FITZ_BAND[r.fitzpatrick] || "?";
    if (byBand[band] != null) byBand[band]++;
    byMethod[r.confirmation_method] = (byMethod[r.confirmation_method] || 0) + 1;
    if (matrix[r.confirmed_diagnosis]) matrix[r.confirmed_diagnosis][band]++;
    if (r.gradable === "yes") gradableYes++;
  }

  // Gap heuristics — actionable prompts, not hard rules.
  for (const d of DIAGNOSES) {
    if (d.key === "uncertain" || d.key === "other") continue;
    const total = byDiagnosis[d.key];
    if (total === 0) continue;
    if (matrix[d.key]["V–VI"] < 5)
      gaps.push(`Few dark-skin (V–VI) cases of ${d.label}: ${matrix[d.key]["V–VI"]}`);
  }
  const fungal = ["tinea", "tinea_steroid_modified", "pityriasis_versicolor"];
  const fungalNone = db().prepare(`
    SELECT COUNT(*) n FROM labels
    WHERE confirmation_method IN ('none','clinical') AND confirmed_diagnosis IN (${fungal.map(() => "?").join(",")})
  `).get(...fungal).n;
  if (fungalNone > 0) gaps.push(`${fungalNone} fungal case(s) confirmed only clinically / not KOH-confirmed`);

  return {
    totals: {
      ...totals,
      gold: gold.length,
      unlabelled: totals.cases - totals.labelled,
      gradable_pct: gold.length ? Math.round((gradableYes / gold.length) * 100) : 0,
    },
    by_diagnosis: DIAGNOSES.map((d) => ({ key: d.key, label: d.label, count: byDiagnosis[d.key] })),
    by_band: FITZ_BANDS.map((b) => ({ band: b, count: byBand[b] })),
    by_method: CONFIRMATION_METHODS.map((m) => ({ key: m.key, label: m.label, count: byMethod[m.key] })),
    coverage_matrix: DIAGNOSES.map((d) => ({ key: d.key, label: d.label, bands: matrix[d.key] })),
    gaps,
  };
}
