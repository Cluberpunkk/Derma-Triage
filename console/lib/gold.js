// console/lib/gold.js
// =========================================================================
// Determines the single GROUND-TRUTH ("gold") label for a case from its
// (possibly multiple) labels. This is where "no case is exported without a
// dermatologist-confirmed label" is enforced:
//
//   - An adjudicated label always wins (an expert explicitly blessed it).
//   - Otherwise, exactly one confirmed label → that label.
//   - Multiple confirmed labels that AGREE on the diagnosis → most recent.
//   - Disagreement with no adjudication → NO gold label (excluded until resolved).
// =========================================================================

import { isConfirmed } from "../shared/vocab.js";
import { getLabels, getAdjudication } from "../../server/lib/db.js";

export function goldLabel(caseId) {
  const labels = getLabels(caseId);
  const adj = getAdjudication(caseId);
  if (adj) {
    const chosen = labels.find((l) => l.label_id === adj.label_id);
    if (chosen && isConfirmed(chosen)) return { label: chosen, resolution: "adjudicated" };
    return null;
  }
  const confirmed = labels.filter(isConfirmed);
  if (confirmed.length === 0) return null;
  if (confirmed.length === 1) return { label: confirmed[0], resolution: "single" };
  const distinctDx = new Set(confirmed.map((l) => l.confirmed_diagnosis));
  if (distinctDx.size === 1) return { label: confirmed[0], resolution: "consensus" }; // labels are updated_at DESC
  return null; // conflict → needs adjudication
}
