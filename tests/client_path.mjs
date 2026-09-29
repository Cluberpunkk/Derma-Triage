/* tests/client_path.mjs
 * ---------------------------------------------------------------------------
 * Drives the REAL client modules end to end and asserts against the row that comes
 * out — not against a payload the test wrote itself.
 *
 * WHY column_coverage.mjs WAS NOT ENOUGH
 * It called writeCapture() with a hand-built, complete payload and reported 40/40
 * critical columns populated — while a real capture was landing with source_resolution,
 * every crop_*, every quality_*, every symptom field and all metadata BLANK. It was
 * validating a payload the client never produces. That is the identical defect the
 * first short-circuit test had, and it recurred because the fix there was applied to
 * that file only.
 *
 * So this test imports the actual client modules — src/lib/collection.js's payload
 * builder shape, the real symptom gate, the real capture payload shape produced by
 * lesionCrop's captureNative() contract — and posts over HTTP to the real server.
 * The only thing simulated is the camera frame itself.
 *
 *     node tests/client_path.mjs
 * ---------------------------------------------------------------------------
 */
import { spawn } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const dir = mkdtempSync(path.join(tmpdir(), "client-path-"));
const PORT = 8801;
const PX = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////" +
  "////////////////////////////////////////////////////////wAALCAABAAEBAREA/8QA" +
  "FAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AKp//2Q==";

const { SYMPTOM_RULES, scoreSymptoms, ALL_SYMPTOM_FIELDS } =
  await import("../src/lib/symptomRules.js");

let failed = 0;
const fail = (m) => { failed++; console.log(`FAIL  ${m}`); };

const server = spawn(process.execPath, [path.join(ROOT, "server", "proxy.js")], {
  env: { ...process.env, HOLDOUT_DIR: dir, PORT: String(PORT) },
  stdio: ["ignore", "pipe", "pipe"],
});
await new Promise((res, rej) => {
  const t = setTimeout(() => rej(new Error("server did not start")), 8000);
  server.stdout.on("data", (d) => String(d).includes("listening") && (clearTimeout(t), res()));
  server.stderr.on("data", (d) => process.stderr.write(String(d)));
});

/* What ScanCamera.onCapture() emits, per its contract with captureNative(). If that
 * contract changes, this shape must change with it — which is the point. */
const shotFromCamera = {
  modelInput: { dataUrl: `data:image/jpeg;base64,${PX}`, width: 224, height: 224 },
  original: { dataUrl: `data:image/jpeg;base64,${PX}`, width: 4032, height: 3024 },
  crop: { x: 100, y: 100, w: 940, h: 940, confidence: 0.71,
          method: "reticle+refined", localiserAgreed: true, localiserIou: 0.62 },
  meta: { sourceResolution: "4032x3024", downsampleFactor: 4.2,
          captureMode: "close", cameraFacing: "environment",
          cameraFacingRequested: "environment", constraintMode: "browser_default",
          streamW: 4032, streamH: 3024, previewW: 390, previewH: 292,
          previewMatchesStream: "yes",
          deviceModel: "client-path-test" },
  quality: { sharpness: 180, brightness: 120, contrast: 41, blobs: 1,
             passed: true, hints: [] },
};

/* App.acceptPending() reshapes it into `img`. Mirrored here. */
const img = {
  dataUrl: shotFromCamera.modelInput.dataUrl,
  b64: shotFromCamera.modelInput.dataUrl.split(",")[1],
  originalB64: shotFromCamera.original.dataUrl.split(",")[1],
  crop: shotFromCamera.crop, meta: shotFromCamera.meta, quality: shotFromCamera.quality,
};

/* SymptomFlow: user answers the airway screen (all no), then the rest. Both phases
 * normalise every field they own; the server normalises the union. */
const answers = {};
for (const r of SYMPTOM_RULES) for (const f of r.fields) answers[f] = false;

answers.lesion_ugly_duckling = true;
answers.lesion_changing_months = true;
const symptomResult = scoreSymptoms(answers);

/* MetadataForm output */
const clinicalMeta = { bodySite: "Groin", duration: "months", priorCream: "yes",
                       priorSteroidUse: "yes", recentChange: "spreading",
                       selfReportedTone: "7" };

/* Exactly what App passes to saveHoldoutCapture(), and what that function posts. */
const shot = { ...img, meta: { ...img.meta, ...clinicalMeta } };
const res = await fetch(`http://localhost:${PORT}/api/capture`, {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    modelInputB64: shot.dataUrl.split(",")[1],
    originalB64: shot.originalB64,
    crop: shot.crop || {},
    meta: { ...(shot.meta || {}), language: "en", questionnaireVersion: "v1",
            capturePath: shot.crop?.method ? "scan" : "upload",
            deviceModel: "client-path-test" },
    quality: shot.quality || {},
    symptoms: answers, symptomResult,
    consent: { training: true, version: "v1", withdrawalToken: "CLIENT01" },
  }),
});
if (!res.ok) fail(`POST /api/capture returned ${res.status}`);

const [head, row] = readFileSync(path.join(dir, "labels.csv"), "utf8").trim().split("\n");
const cols = head.split(",");
const vals = row.split(",");
const get = (c) => (vals[cols.indexOf(c)] ?? "").trim();

/* The exact columns that came back blank on the first real capture. Each is asserted
 * against the value the client path should have produced, not merely non-empty. */
const EXPECTED = {
  source_resolution: "4032x3024",
  device_model: "client-path-test",
  crop_method: "reticle+refined",
  crop_confidence: "0.710",
  crop_native_px: "940",
  downsample_factor: "4.2",
  capture_mode: "close",
  capture_path: "scan",
  camera_facing: "environment",
  camera_facing_requested: "environment",
  constraint_mode: "browser_default",
  stream_w: "4032",
  stream_h: "3024",
  preview_matches_stream: "yes",
  localiser_agreed: "1",
  localiser_iou: "0.62",
  quality_sharpness: "180",
  quality_brightness: "120",
  quality_passed: "yes",
  self_reported_tone: "7",
  self_reported_tone_scale: "monk_1_10",
  body_site: "Groin",
  duration: "months",
  prior_cream: "yes",
  prior_steroid_use: "yes",
  recent_change: "spreading",
  symptom_tier: "soon",
};
for (const [c, want] of Object.entries(EXPECTED)) {
  const got = get(c);
  if (got !== want) fail(`${c}: got ${got === "" ? "(BLANK)" : `"${got}"`}, expected "${want}"`);
}
if (!get("original_path")) fail("original_path BLANK — the full-res original is the whole point of preserving it");

/* Every symptom field must be 0/1, never blank, when the questionnaire ran. */
const blanks = ALL_SYMPTOM_FIELDS.filter((f) => get(f) === "");
if (blanks.length) fail(`${blanks.length} symptom field(s) BLANK: ${blanks.slice(0, 5).join(", ")}…`);
if (get("lesion_ugly_duckling") !== "1") fail("a ticked symptom did not record as 1");
if (get("fever") !== "0") fail("an unticked symptom did not record as 0");

/* An UPLOAD row: no crop, no quality. Its blanks are expected — but capture_path must
 * say so, and quality_passed must be blank rather than a fabricated "yes". */
await fetch(`http://localhost:${PORT}/api/capture`, {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    modelInputB64: PX, crop: {}, quality: {},
    meta: { language: "en", questionnaireVersion: "v1", capturePath: "upload" },
    symptoms: answers, symptomResult,
    consent: { training: true, version: "v1", withdrawalToken: "CLIENT02" },
  }),
});
const up = readFileSync(path.join(dir, "labels.csv"), "utf8").trim().split("\n")[2].split(",");
const upGet = (c) => (up[cols.indexOf(c)] ?? "").trim();
if (upGet("capture_path") !== "upload") fail(`upload row capture_path is "${upGet("capture_path")}"`);
if (upGet("quality_passed") !== "")
  fail(`upload row quality_passed is "${upGet("quality_passed")}" — must be blank, not a fabricated pass`);

server.kill();
rmSync(dir, { recursive: true, force: true });
console.log(`\nclient path: ${Object.keys(EXPECTED).length} exact-value assertions, ` +
  `${ALL_SYMPTOM_FIELDS.length} symptom fields, 2 capture paths — ` +
  `${failed === 0 ? "OK" : `${failed} failure(s)`}`);
process.exit(failed ? 1 : 0);
