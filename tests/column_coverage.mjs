/* tests/column_coverage.mjs
 * ---------------------------------------------------------------------------
 * Submits a synthetic capture through the WRITE path and asserts that every
 * analysis-critical column comes back non-empty.
 *
 * A NEW FAILURE SHAPE
 * check_wiring.py catches "module exists, nothing calls it". This catches something
 * it structurally cannot see: the module exists, IS called, runs without error — and
 * a field is never populated. `fitzpatrick` was empty on every row while
 * eval_holdout.py silently reported fairness_measurable: false. Nothing failed. We
 * would have discovered it after collecting 300 images, not before.
 *
 * The test drives the same payload shape the client sends, so a client that stops
 * sending a field fails here rather than at analysis time.
 *
 *     node tests/column_coverage.mjs
 * ---------------------------------------------------------------------------
 */
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

const dir = mkdtempSync(path.join(tmpdir(), "coverage-"));
process.env.HOLDOUT_DIR = dir;
const H = await import("../server/lib/holdout.js");

// A 1x1 JPEG. Content is irrelevant; only the write path is under test.
const PX = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////" +
  "////////////////////////////////////////////////////////wAALCAABAAEBAREA/8QA" +
  "FAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AKp//2Q==";

/* Columns are grouped by what BREAKS when they are blank. A column being optional
 * is a decision, so each one says why. */
// Sourced from ml/symptom_gate.py ALL_FIELDS via the generated JS, so adding a rule
// automatically adds its fields here rather than silently escaping the check.
const SYMPTOM_FIELDS = ["bleeding_no_trauma", "blisters_large_area", "breathing_difficulty", "changing_lesion", "face_swelling", "fatigue", "fever", "immunosuppressed", "joint_pain", "lesion_bleeds", "lesion_changing_months", "lesion_itches", "lesion_new_after_40", "lesion_non_healing", "lesion_ugly_duckling", "mucosal_involvement", "neonate", "non_healing_4w", "photosensitivity", "pregnant_with_rash", "skin_peeling", "spreading_hours", "tender", "throat_tightness", "tongue_swelling", "warm_to_touch", "weight_loss"];

const CRITICAL = {
  image_path: "no image, no analysis at all",
  capture_id: "rows cannot be joined or withdrawn",
  captured_utc: "no ordering, no session reconstruction",
  source_resolution: "the entire native-resolution argument is unverifiable",
  downsample_factor: "cannot tell a tight crop from a centre-crop fallback",
  crop_method: "cannot separate reticle captures from fallbacks",
  quality_sharpness: "shadow-mode calibration has nothing to calibrate on",
  quality_brightness: "same",
  symptom_tier: "per-rule firing rate is the pilot's PRIMARY measurable",
  self_reported_tone: "the only tone signal the app can collect; without it the " +
    "continuous fairness substitute (powered at n=349) has no input",
  consent_training: "a row without recorded consent must not exist",
  consent_version: "cannot tell which consent text was agreed to",
  withdrawal_token: "DPDP withdrawal becomes impossible for that row",
  preview_matches_stream: "if the preview crops, the reticle does not map onto the " +
    "captured pixels and every framing decision is made against the wrong image",
  stream_w: "what the sensor actually delivered, vs what was requested",
  stream_h: "same",
  camera_facing: "front cameras are materially lower resolution on most phones, so " +
    "without this, front-camera captures silently confound the native-resolution question",
  capture_path: "scan vs upload vs short-circuit — without it, an upload row's expected " +
    "blanks are indistinguishable from a scan row's defects",
  // All 27 symptom fields. SymptomFlow writes every one explicitly as true/false on
  // completion, precisely so "not asked" stays distinguishable from "answered no".
  // A blank here means the client stopped sending them — which is how per-rule firing
  // rate, the pilot's primary measurable, would quietly become uncomputable.
  ...Object.fromEntries(SYMPTOM_FIELDS.map((f) => [f,
    "symptom field — blank means the client stopped sending the questionnaire"])),
};

const OPTIONAL = {
  fitzpatrick: "reader-assigned in the labelling console; ALWAYS blank at capture",
  true_class: "dermatologist-assigned only; blank at capture by design",
  urgency: "derived from true_class at labelling time",
  original_path: "absent when the upload path is used instead of the camera",
  body_site: "user may skip the metadata form",
  duration: "user may skip",
  prior_cream: "user may skip",
  prior_steroid_use: "only asked when a cream was reported",
  recent_change: "user may skip",
  localiser_agreed: "n/a in wide mode, where no localisation runs",
  localiser_iou: "same",
  capture_distance_cm: "not collected in the pilot build",
  notes: "free text",
  quality_fill: "removed metric, retained for schema stability",
  quality_passed: "shadow mode records hints instead",
  capture_mode: "defaults when the toggle is untouched",
  capture_language: "English-only pilot",
  questionnaire_version: "set by the client",
  self_reported_tone_scale: "set only when a tone is chosen",
  symptom_rules_fired: "empty when no rule fires — a real and common outcome",
  device_model: "user-agent string; useful but not analysis-critical",
  camera_facing_requested: "diagnostic only — differs from camera_facing when the " +
    "browser ignores the facingMode hint",
  preview_w: "diagnostic; blank on the upload path",
  preview_h: "diagnostic; blank on the upload path",
  constraint_mode: "which constraint A/B produced this row; blank on upload",
  crop_confidence: "n/a in wide mode",
  crop_native_px: "n/a in wide mode",
};

// Exactly what src/lib/collection.js saveHoldoutCapture() sends.
H.writeCapture({
  modelInputB64: PX,
  originalB64: PX,
  crop: { method: "reticle+refined", confidence: 0.71, w: 940,
          localiserAgreed: true, localiserIou: 0.62 },
  meta: {
    sourceResolution: "4032x3024", downsampleFactor: 4.2, captureMode: "close",
    deviceModel: "coverage-test", language: "en", questionnaireVersion: "v1",
    capturePath: "scan", cameraFacing: "environment",
    cameraFacingRequested: "environment", constraintMode: "browser_default",
    streamW: 4032, streamH: 3024, previewW: 390, previewH: 292,
    previewMatchesStream: "yes",
    selfReportedTone: "7",
    bodySite: "Groin", duration: "months", priorCream: "yes",
    priorSteroidUse: "yes", recentChange: "spreading",
  },
  quality: { sharpness: 180, brightness: 120, fill: 0.55, passed: true },
  // SymptomFlow sets every field explicitly on completion; mirror that here so a
  // client that stops doing so fails this test.
  symptoms: Object.fromEntries(SYMPTOM_FIELDS.map((f) =>
    [f, f === "lesion_ugly_duckling" || f === "lesion_changing_months"])),
  symptomResult: { tier: "soon", rulesFired: [{ rule: "pigmented_lesion" }] },
  consent: { training: true, version: "2026-07-01.v1", withdrawalToken: "ABC12345" },
});

const [head, row] = readFileSync(path.join(dir, "labels.csv"), "utf8").trim().split("\n");
const cols = head.split(",");
const vals = row.split(",");
const get = (c) => (vals[cols.indexOf(c)] ?? "").trim();

let failed = 0;
const unclassified = cols.filter((c) => !(c in CRITICAL) && !(c in OPTIONAL));
for (const [c, why] of Object.entries(CRITICAL)) {
  if (!cols.includes(c)) { failed++; console.log(`FAIL  ${c}: column missing entirely`); }
  else if (!get(c)) { failed++; console.log(`FAIL  ${c} is BLANK — ${why}`); }
}
// fitzpatrick must stay blank: self-report must never land in the reader's column.
if (get("fitzpatrick")) {
  failed++;
  console.log("FAIL  fitzpatrick is populated at capture time — it is reader-assigned " +
    "only, and self-report must not be able to flip fairness_measurable");
}
if (unclassified.length) {
  failed++;
  console.log(`FAIL  ${unclassified.length} column(s) classified neither critical nor ` +
    `optional: ${unclassified.join(", ")}\n      Add each to CRITICAL or OPTIONAL with a ` +
    `reason — an unclassified column is one nobody has decided about.`);
}

rmSync(dir, { recursive: true, force: true });
console.log(`\ncolumn coverage: ${Object.keys(CRITICAL).length} critical checked, ` +
  `${Object.keys(OPTIONAL).length} optional, ${cols.length} total, ${failed} failure(s)`);
process.exit(failed ? 1 : 0);
