/* server/lib/holdout.js
 * ---------------------------------------------------------------------------
 * Writes each in-app capture directly into the holdout schema that
 * ml/eval_holdout.py reads — no manual export step.
 *
 * WHY THIS SHAPE
 * The holdout was originally going to be a separate collection kit. It isn't:
 * captures come through the SAME path production uses, so we stop measuring on a
 * distribution we don't ship. That is the whole point — every earlier number in this
 * project is from curated atlas photography, and the gap to real phone photos is the
 * binding uncertainty.
 *
 * Two images are written per capture:
 *   original/    full-resolution frame, never downsampled — so a better model later
 *                can be re-run on the same capture without re-collecting
 *   model_input/ the native-resolution lesion crop, resized to the model size
 *
 * labels.csv carries everything eval_holdout.py needs plus the 25 symptom fields
 * (symptom_gate.py --eval reads the same file), source resolution, device, capture
 * distance and Fitzpatrick tone.
 *
 * REJECTIONS ARE RECORDED TOO. If the quality gate rejects most real-world attempts
 * that is a product problem, and it is invisible unless the rejections are counted.
 * ---------------------------------------------------------------------------
 */
import fs from "node:fs";
import path from "node:path";
import { DATA_DIR } from "./storage.js";
import { stripExif, b64ToBuffer } from "./images.js";
import { ALL_SYMPTOM_FIELDS } from "../../src/lib/symptomRules.js";

export const HOLDOUT_DIR = process.env.HOLDOUT_DIR || path.join(DATA_DIR, "holdout");

// Column order must stay in sync with ml/eval_holdout.py's reader and with
// ml/symptom_gate.py ALL_FIELDS. `true_class` is left blank at capture time — it is
// filled by a dermatologist in the labelling console, never by the app or the user.
// DERIVED, not duplicated. This list was hardcoded and drifted: throat_tightness and
// tongue_swelling were added with the emergency tier in ml/symptom_gate.py but never
// reached here, so the emergency rule's own answers would never have been stored on
// any row. tests/column_coverage.mjs caught it. Importing from the generated module
// makes that drift impossible — symptomRules.js is itself generated from the Python.
export const SYMPTOM_FIELDS = [...ALL_SYMPTOM_FIELDS];

export const COLUMNS = [
  "image_path", "true_class", "urgency",
  // fitzpatrick is RESERVED FOR A TRAINED READER. It is never written by the app.
  // Self-report goes to self_reported_tone (Monk 1-10 via a swatch strip), because:
  //   - Fitzpatrick was built around burn/tan response in light skin and South Asian
  //     users classify themselves on it inconsistently
  //   - a self-reported value must not be able to flip fairness_measurable, which
  //     should mean "a trained reader assigned this"
  // Keeping them in separate columns also gives three tone signals on one image -
  // self-report, ITA-from-image, and later the reader's - and comparing them is the
  // first real validation of the ITA proxy that OPEN-1 rests on.
  "fitzpatrick", "self_reported_tone", "self_reported_tone_scale",
  "capture_id", "captured_utc", "original_path",
  "source_resolution", "device_model", "capture_distance_cm",
  "crop_method", "crop_confidence", "crop_native_px", "downsample_factor",
  "capture_mode", "localiser_agreed", "localiser_iou",
  // Which path produced this row. An upload has no reticle, no native crop and no
  // live quality metrics, so its blanks are EXPECTED; a scan row with the same blanks
  // is a defect. Without this column the two are indistinguishable after the fact.
  "capture_path", "camera_facing", "camera_facing_requested",
  // Preview geometry, so a cropped-preview regression is visible in the DATA.
  // The stage previously forced a 3/4 box on a 4:3 stream and cropped with
  // object-fit:cover, which looked like the camera opening zoomed in.
  "preview_w", "preview_h", "stream_w", "stream_h", "preview_matches_stream",
  "constraint_mode",
  "quality_sharpness", "quality_brightness", "quality_fill", "quality_passed",
  "symptom_tier", "symptom_rules_fired",
  // Language the QUESTIONNAIRE was answered in. The rules are the load-bearing
  // safety mechanism, so a mistranslation is a safety defect, not a UX one. A
  // per-language firing-rate difference is a TRANSLATION defect until proven
  // epidemiological — and it is undetectable unless recorded here.
  "capture_language", "questionnaire_version",
  // Clinical metadata — the ml/metadata.py fusion inputs. Asked AFTER capture: the
  // photo is the scarce artifact, so bank it before spending the user's patience.
  "body_site", "duration", "prior_cream", "prior_steroid_use", "recent_change",
  ...SYMPTOM_FIELDS,
  "consent_training", "consent_version", "notes",
  // The ONLY handle for consent withdrawal. We store no name, phone or email,
  // so a lost token is unrecoverable — the correct trade against holding an
  // identifier we promised not to collect.
  "withdrawal_token",
];

function ensureDirs() {
  for (const d of ["original", "model_input"]) {
    fs.mkdirSync(path.join(HOLDOUT_DIR, d), { recursive: true });
  }
  const csv = path.join(HOLDOUT_DIR, "labels.csv");
  if (!fs.existsSync(csv)) fs.writeFileSync(csv, COLUMNS.join(",") + "\n");
  return csv;
}

function csvCell(v) {
  if (v === undefined || v === null) return "";
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/**
 * Persist one capture into the holdout schema.
 * Returns { capture_id, image_path, original_path }.
 */
export function writeCapture({
  modelInputB64, originalB64, meta = {}, symptoms = {}, symptomResult = {},
  quality = {}, crop = {}, consent = {},
}) {
  if (!consent.training) {
    throw new Error("refusing to write a capture without training consent");
  }
  const csv = ensureDirs();
  const id = `cap_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
  const rel = path.join("model_input", `${id}.jpg`);
  const relOrig = originalB64 ? path.join("original", `${id}.jpg`) : "";

  // EXIF/GPS stripped server-side as a guarantee, on top of the canvas re-encode
  fs.writeFileSync(path.join(HOLDOUT_DIR, rel), stripExif(b64ToBuffer(modelInputB64)));
  if (originalB64) {
    fs.writeFileSync(path.join(HOLDOUT_DIR, relOrig), stripExif(b64ToBuffer(originalB64)));
  }

  const row = {
    image_path: rel,
    true_class: "",          // dermatologist-assigned only; never the app or the user
    urgency: "",             // derived from true_class at labelling time
    fitzpatrick: "",                       // reader-assigned only; never the app
    self_reported_tone: meta.selfReportedTone ?? "",
    self_reported_tone_scale: meta.selfReportedTone ? "monk_1_10" : "",
    capture_id: id,
    captured_utc: new Date().toISOString(),
    original_path: relOrig,
    source_resolution: meta.sourceResolution || "",
    device_model: meta.deviceModel || "",
    capture_distance_cm: meta.captureDistanceCm ?? "",
    crop_method: crop.method || "",
    crop_confidence: crop.confidence != null ? Number(crop.confidence).toFixed(3) : "",
    crop_native_px: crop.w ?? "",
    downsample_factor: meta.downsampleFactor ?? "",
    capture_mode: meta.captureMode || "",
    capture_path: meta.capturePath || (crop.method ? "scan" : "upload"),
    camera_facing: meta.cameraFacing || "",
    camera_facing_requested: meta.cameraFacingRequested || "",
    preview_w: meta.previewW ?? "",
    preview_h: meta.previewH ?? "",
    stream_w: meta.streamW ?? "",
    stream_h: meta.streamH ?? "",
    preview_matches_stream: meta.previewMatchesStream || "",
    constraint_mode: meta.constraintMode || "",
    localiser_agreed: crop.localiserAgreed === undefined ? "" : (crop.localiserAgreed ? "1" : "0"),
    localiser_iou: crop.localiserIou ?? "",
    quality_sharpness: quality.sharpness != null ? Math.round(quality.sharpness) : "",
    quality_brightness: quality.brightness != null ? Math.round(quality.brightness) : "",
    quality_fill: quality.fill != null ? Number(quality.fill).toFixed(3) : "",
    // Tri-state. The old form was `passed === false ? "no" : "yes"`, so a row with NO
    // quality data recorded "yes" — a wrong value rather than a missing one, and the
    // reason a metadata-less upload row looked like it had passed the gate.
    quality_passed: quality.passed === true ? "yes"
                  : quality.passed === false ? "no" : "",
    capture_language: meta.language || "",
    questionnaire_version: meta.questionnaireVersion || "",
    body_site: meta.bodySite || "",
    duration: meta.duration || "",
    prior_cream: meta.priorCream || "",
    prior_steroid_use: meta.priorSteroidUse || "",
    recent_change: meta.recentChange || "",
    symptom_tier: symptomResult.tier || "none",
    symptom_rules_fired: (symptomResult.rulesFired || []).map((r) => r.rule).join("|"),
    consent_training: "yes",
    consent_version: consent.version || "",
    notes: meta.notes || "",
    withdrawal_token: consent.withdrawalToken || "",
  };
  // ONE normalisation point for all 27 fields. Each screen used to normalise its own,
  // which meant the airway screen could ship un-normalised (it did) and every other
  // form could repeat it independently. `answered` says whether the questionnaire ran
  // at all, so a skipped questionnaire stays distinguishable from all-negative answers.
  const answered = symptoms && Object.keys(symptoms).length > 0;
  for (const f of SYMPTOM_FIELDS) {
    row[f] = answered ? (symptoms[f] === true ? "1" : "0") : "";
  }

  fs.appendFileSync(csv, COLUMNS.map((c) => csvCell(row[c])).join(",") + "\n");
  return { capture_id: id, image_path: rel, original_path: relOrig };
}

/**
 * Record a capture the quality gate REJECTED. No image is stored — only the reason.
 * The rejection rate is a product metric: a gate that rejects most real attempts is
 * a usability failure, and it is invisible unless counted.
 */
export function recordRejection({ reasons = [], quality = {}, meta = {} }) {
  fs.mkdirSync(HOLDOUT_DIR, { recursive: true });
  const f = path.join(HOLDOUT_DIR, "rejections.csv");
  if (!fs.existsSync(f)) {
    fs.writeFileSync(f, "rejected_utc,reasons,sharpness,brightness,fill,source_resolution,device_model\n");
  }
  fs.appendFileSync(f, [
    new Date().toISOString(), reasons.join("|"),
    quality.sharpness != null ? Math.round(quality.sharpness) : "",
    quality.brightness != null ? Math.round(quality.brightness) : "",
    quality.fill != null ? Number(quality.fill).toFixed(3) : "",
    meta.sourceResolution || "", meta.deviceModel || "",
  ].map(csvCell).join(",") + "\n");
}

/**
 * A SHORT-CIRCUIT encounter: the symptom gate reached `emergency`, so the flow
 * stopped before any photo was taken.
 *
 * These must still be recorded. The emergency rule has the weakest prevalence anchor
 * on the whole form (SJS/TEN incidence is ~1-2 per million, so our 0.2% assumption is
 * likely wrong by orders of magnitude), and its firing rate is one of the things the
 * pilot exists to measure. If short-circuit encounters left no row, the rule we
 * scrutinise hardest would be the only one whose behaviour is invisible.
 *
 * No image is written — there is none. Every other column behaves as normal.
 */
export function writeShortCircuit({
  meta = {}, symptoms = {}, symptomResult = {}, consent = {},
}) {
  if (!consent.training) {
    throw new Error("refusing to write a short-circuit row without training consent");
  }
  const csv = ensureDirs();
  const id = `sc_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
  const row = {
    image_path: "",             // no photo was taken; the flow stopped first
    true_class: "",
    urgency: "",
    fitzpatrick: "",                       // reader-assigned only; never the app
    self_reported_tone: meta.selfReportedTone ?? "",
    self_reported_tone_scale: meta.selfReportedTone ? "monk_1_10" : "",
    capture_id: id,
    captured_utc: new Date().toISOString(),
    original_path: "",
    source_resolution: "",
    device_model: meta.deviceModel || "",
    capture_distance_cm: "",
    crop_method: "short-circuit-no-capture",
    crop_confidence: "", crop_native_px: "", downsample_factor: "",
    capture_mode: "", localiser_agreed: "", localiser_iou: "",
    capture_path: "short-circuit",
    camera_facing: "",
    camera_facing_requested: "", preview_w: "", preview_h: "",
    stream_w: "", stream_h: "", preview_matches_stream: "", constraint_mode: "",
    quality_sharpness: "", quality_brightness: "", quality_fill: "",
    quality_passed: "",
    capture_language: meta.language || "",
    questionnaire_version: meta.questionnaireVersion || "",
    body_site: "", duration: "", prior_cream: "", prior_steroid_use: "",
    recent_change: "",                     // no capture happened, so none were asked
    symptom_tier: symptomResult.tier || "",
    symptom_rules_fired: (symptomResult.rulesFired || []).map((r) => r.rule).join("|"),
    consent_training: "yes",
    consent_version: consent.version || "",
    notes: meta.notes || "",
    withdrawal_token: consent.withdrawalToken || "",
  };
  for (const f of SYMPTOM_FIELDS) {
    row[f] = symptoms[f] === true ? "1" : symptoms[f] === false ? "0" : "";
  }
  fs.appendFileSync(csv, COLUMNS.map((c) => csvCell(row[c])).join(",") + "\n");
  return { capture_id: id, short_circuit: true };
}

/**
 * Funnel drop-off. The emergency questions are asked FIRST, and four airway questions
 * put to someone photographing acne is an alarming opening. If it costs completions
 * that is a product problem, and it is invisible unless the abandonment point is
 * recorded.
 */
export function recordDropoff({ step = "", meta = {} }) {
  fs.mkdirSync(HOLDOUT_DIR, { recursive: true });
  const f = path.join(HOLDOUT_DIR, "dropoff.csv");
  if (!fs.existsSync(f)) {
    fs.writeFileSync(f, "utc,step,language,device_model\n");
  }
  fs.appendFileSync(f, [new Date().toISOString(), step,
    meta.language || "", meta.deviceModel || ""].map(csvCell).join(",") + "\n");
}

/**
 * CONSENT WITHDRAWAL — delete a participant's images and rows.
 *
 * A DPDP requirement, and promised in docs/CLINIC_STUDY_spec.md ("right to withdraw
 * at any time, with deletion of that participant's images and records"). Collecting
 * before this existed would have meant making a promise we could not keep.
 *
 * Deletion is by `withdrawal_token` — a short code shown to the participant after
 * capture. We deliberately store no name, phone or email, so the token is the ONLY
 * handle on a row. That means a lost token is unrecoverable, which is the correct
 * trade: the alternative is holding an identifier we promised not to collect.
 *
 * Images are unlinked from disk and the CSV row is rewritten out. `--dry-run` reports
 * what would go without touching anything.
 */
export function withdraw(token, { dryRun = false } = {}) {
  const csv = path.join(HOLDOUT_DIR, "labels.csv");
  if (!fs.existsSync(csv) || !token) return { matched: 0, deleted: [], dry_run: dryRun };

  const lines = fs.readFileSync(csv, "utf8").trim().split("\n");
  const header = lines[0].split(",");
  const iTok = header.indexOf("withdrawal_token");
  const iImg = header.indexOf("image_path");
  const iOrig = header.indexOf("original_path");
  if (iTok < 0) return { matched: 0, deleted: [], error: "no withdrawal_token column" };

  const keep = [lines[0]];
  const files = [];
  let matched = 0;
  for (const line of lines.slice(1)) {
    const cells = line.split(",");
    if (cells[iTok] === token) {
      matched++;
      for (const i of [iImg, iOrig]) {
        const rel = cells[i];
        if (rel) files.push(rel);
      }
    } else {
      keep.push(line);
    }
  }

  if (!dryRun) {
    for (const rel of files) {
      const abs = path.join(HOLDOUT_DIR, rel);
      try { if (fs.existsSync(abs)) fs.unlinkSync(abs); } catch { /* already gone */ }
    }
    fs.writeFileSync(csv, keep.join("\n") + "\n");
    // Audit trail: WHAT was deleted and WHEN, with no content and no token.
    const log = path.join(HOLDOUT_DIR, "withdrawals.log");
    fs.appendFileSync(log,
      `${new Date().toISOString()} withdrew ${matched} row(s), ${files.length} file(s)\n`);
  }
  return { matched, deleted: files, dry_run: dryRun };
}

/** Counts for the console dashboard, including the gate's rejection rate. */
export function stats() {
  const csv = path.join(HOLDOUT_DIR, "labels.csv");
  const rej = path.join(HOLDOUT_DIR, "rejections.csv");
  const count = (f) => (fs.existsSync(f)
    ? Math.max(0, fs.readFileSync(f, "utf8").trim().split("\n").length - 1) : 0);
  const accepted = count(csv);
  const rejected = count(rej);
  const dropped = count(path.join(HOLDOUT_DIR, "dropoff.csv"));
  const total = accepted + rejected;
  let shortCircuit = 0;
  if (fs.existsSync(csv)) {
    const lines = fs.readFileSync(csv, "utf8").trim().split("\n").slice(1);
    const i = COLUMNS.indexOf("symptom_tier");
    shortCircuit = lines.filter((l) => l.split(",")[i] === "emergency").length;
  }
  return {
    accepted, rejected, total, dropped, short_circuit: shortCircuit,
    rejection_rate: total ? +(rejected / total).toFixed(4) : null,
    dir: HOLDOUT_DIR,
  };
}
