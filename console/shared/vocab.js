// console/shared/vocab.js
// =========================================================================
// CONTROLLED VOCABULARY for expert labelling. Shared by the console server
// (validation) and the console web app (UI). This is the ground-truth schema:
// only values defined here become a confirmed label.
//
// Weighted toward the Indian outpatient mix, with a can't-miss / refer bucket.
// The diagnosis list is intended to be EDITED — add/rename entries here and both
// the API validation and the UI pick it up. Keep `key` stable once data exists
// (the export/dashboard group on it); `label` is display-only and safe to reword.
// =========================================================================

export const DIAGNOSES = [
  { key: "tinea", label: "Dermatophytosis (tinea)" },
  { key: "tinea_steroid_modified", label: "Dermatophytosis — steroid-modified" },
  { key: "eczema", label: "Eczema / atopic dermatitis" },
  { key: "acne", label: "Acne" },
  { key: "melasma", label: "Melasma" },
  { key: "pih", label: "Post-inflammatory hyperpigmentation" },
  { key: "pityriasis_versicolor", label: "Pityriasis versicolor" },
  { key: "scabies", label: "Scabies" },
  { key: "psoriasis", label: "Psoriasis" },
  { key: "urticaria", label: "Urticaria" },
  { key: "vitiligo", label: "Vitiligo" },
  { key: "impetigo", label: "Impetigo" },
  { key: "warts_molluscum", label: "Viral warts / molluscum" },
  { key: "suspected_malignancy", label: "Suspected malignancy — refer (can't-miss)" },
  { key: "other", label: "Other (describe in notes)" },
  { key: "uncertain", label: "Uncertain / needs confirmation" },
];

// Diagnoses that are NOT a settled ground truth: excluded from the training
// export even when every other mandatory field is present.
export const NON_TERMINAL_DIAGNOSES = new Set(["uncertain"]);

export const CONFIRMATION_METHODS = [
  { key: "clinical", label: "Clinical only" },
  { key: "koh", label: "KOH" },
  { key: "culture", label: "Culture" },
  { key: "dermoscopy", label: "Dermoscopy" },
  { key: "biopsy", label: "Biopsy" },
  { key: "none", label: "None" },
];

export const FITZPATRICK = ["I", "II", "III", "IV", "V", "VI"];

// Monk Skin Tone scale (10 tones). Kept as numbers 1–10.
export const MONK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

export const BODY_SITES = [
  "Face", "Scalp", "Neck", "Chest", "Back", "Abdomen",
  "Groin", "Buttocks", "Arm", "Hand", "Leg", "Foot", "Nails", "Trunk", "Other",
];

export const PRIOR_STEROID = ["yes", "no", "unknown"];

export const GRADABLE = ["yes", "no"];

// Fitzpatrick → coarse skin-tone band, used to stratify splits and to surface
// dark-skin coverage gaps in the dashboard.
export const FITZ_BAND = { I: "I–II", II: "I–II", III: "III–IV", IV: "III–IV", V: "V–VI", VI: "V–VI" };
export const FITZ_BANDS = ["I–II", "III–IV", "V–VI"];

// Fields that MUST be present for a label to count as a confirmed, exportable
// ground-truth label. Enforced server-side on save and again at export.
export const MANDATORY_LABEL_FIELDS = [
  "confirmed_diagnosis",
  "confirmation_method",
  "fitzpatrick",
  "monk",
  "image_gradable",
];

// Validate a submitted label against the vocabulary. Returns { ok, errors }.
export function validateLabel(l) {
  const errors = [];
  const has = (v) => v !== undefined && v !== null && v !== "";

  for (const f of MANDATORY_LABEL_FIELDS) {
    if (!has(l[f])) errors.push(`${f} is required`);
  }
  if (has(l.confirmed_diagnosis) && !DIAGNOSES.some((d) => d.key === l.confirmed_diagnosis))
    errors.push("confirmed_diagnosis not in vocabulary");
  if (has(l.confirmation_method) && !CONFIRMATION_METHODS.some((c) => c.key === l.confirmation_method))
    errors.push("confirmation_method not in vocabulary");
  if (has(l.fitzpatrick) && !FITZPATRICK.includes(l.fitzpatrick))
    errors.push("fitzpatrick not in I–VI");
  if (has(l.monk) && !MONK.includes(Number(l.monk)))
    errors.push("monk not in 1–10");
  if (has(l.body_site) && !BODY_SITES.includes(l.body_site))
    errors.push("body_site not in vocabulary");
  if (has(l.prior_steroid_use) && !PRIOR_STEROID.includes(l.prior_steroid_use))
    errors.push("prior_steroid_use must be yes/no/unknown");
  if (has(l.image_gradable) && !GRADABLE.includes(l.image_gradable))
    errors.push("image_gradable must be yes/no");
  if (l.image_gradable === "no" && !has(l.gradable_reason))
    errors.push("gradable_reason is required when image is not gradable");

  return { ok: errors.length === 0, errors };
}

// A confirmed label eligible to seed the TRAINING export: every mandatory field
// present AND a terminal (settled) diagnosis. Gradability is exported as a column;
// only gradable=yes rows are placed into train/val/test splits (see export.js).
export function isConfirmed(l) {
  if (!l) return false;
  const { ok } = validateLabel(l);
  return ok && !NON_TERMINAL_DIAGNOSES.has(l.confirmed_diagnosis);
}
