/* tests/shortcircuit_path.mjs
 * ---------------------------------------------------------------------------
 * End-to-end test of the EMERGENCY path: client payload -> HTTP endpoint -> CSV row.
 *
 * WHY THIS EXISTS SEPARATELY
 * The emergency tier is the only path that skips capture entirely, so it writes
 * through saveShortCircuit()/writeShortCircuit() rather than the normal capture
 * route. Nothing exercised it:
 *
 *   tests/conformance.*      cover the RULE LOGIC (does throat tightness -> emergency)
 *   tests/column_coverage    covers the SCHEMA on the normal capture write
 *   this file                covers the WRITE PATH that only emergencies take
 *
 * That path has already failed twice for unrelated reasons — saveShortCircuit was
 * unreferenced (caught by check_wiring), and throat_tightness/tongue_swelling were
 * missing from the schema (caught by column_coverage). Both were found by checks
 * built for other purposes. This one is aimed at it directly.
 *
 * It boots the real server on a scratch port and posts over HTTP, so a broken route,
 * a bad guard, or a serialisation bug fails here rather than in the pilot — where the
 * symptom is silence, since an emergency encounter produces no image to notice.
 *
 *     node tests/shortcircuit_path.mjs
 * ---------------------------------------------------------------------------
 */
import { spawn } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const dir = mkdtempSync(path.join(tmpdir(), "sc-path-"));
const PORT = 8799;

const { scoreSymptoms, SYMPTOM_RULES } = await import("../src/lib/symptomRules.js");
const AIRWAY = SYMPTOM_RULES.find((r) => r.tier === "emergency").fields;

let failed = 0;
const fail = (m) => { failed++; console.log(`FAIL  ${m}`); };

const server = spawn(process.execPath, [path.join(ROOT, "server", "proxy.js")], {
  env: { ...process.env, HOLDOUT_DIR: dir, PORT: String(PORT) },
  stdio: ["ignore", "pipe", "pipe"],
});
const ready = new Promise((res, rej) => {
  const t = setTimeout(() => rej(new Error("server did not start in 8s")), 8000);
  server.stdout.on("data", (d) => {
    if (String(d).includes("listening")) { clearTimeout(t); res(); }
  });
  server.stderr.on("data", (d) => process.stderr.write(String(d)));
});

try {
  await ready;

  // --- reproduce the CLIENT'S OWN construction, not a hand-built payload.
  //
  // The first version of this test built the payload itself and passed while the
  // component was broken: an untouched checkbox never fires onChange, so SymptomFlow
  // was sending only the ticked field and three airway answers landed blank. A test
  // that constructs the ideal input cannot catch a client that doesn't produce it.
  //
  // `touched` is what React state holds after one tick; the normalisation below is
  // the component's, and must stay in step with it.
  const touched = { throat_tightness: true };
  const answers = { ...touched };
  for (const f of AIRWAY) answers[f] = touched[f] === true;
  const result = scoreSymptoms(answers);

  if (result.tier !== "emergency") fail(`gate returned ${result.tier}, expected emergency`);
  if (!result.shortCircuit) fail("gate did not set shortCircuit");

  // --- exactly what src/lib/collection.js saveShortCircuit() posts
  const res = await fetch(`http://localhost:${PORT}/api/capture/short-circuit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      meta: { language: "en", questionnaireVersion: "v1", deviceModel: "sc-path-test" },
      symptoms: answers,
      symptomResult: result,
      consent: { training: true, version: "2026-07-01.v1", withdrawalToken: "SCTEST01" },
    }),
  });
  if (!res.ok) fail(`endpoint returned HTTP ${res.status}`);
  const body = await res.json().catch(() => ({}));
  if (!body.short_circuit) fail("response did not confirm short_circuit");

  // --- the row that actually landed
  const csvPath = path.join(dir, "labels.csv");
  if (!existsSync(csvPath)) {
    fail("no labels.csv was written at all");
  } else {
    const [head, ...rows] = readFileSync(csvPath, "utf8").trim().split("\n");
    const cols = head.split(",");
    if (rows.length !== 1) fail(`expected exactly 1 row, got ${rows.length}`);
    const vals = rows[0].split(",");
    const get = (c) => (vals[cols.indexOf(c)] ?? "").trim();

    if (get("symptom_tier") !== "emergency")
      fail(`symptom_tier is ${get("symptom_tier") || "(blank)"}, expected emergency`);
    if (get("symptom_rules_fired") !== "airway_emergency")
      fail(`symptom_rules_fired is ${get("symptom_rules_fired") || "(blank)"}`);
    if (get("crop_method") !== "short-circuit-no-capture")
      fail(`crop_method is ${get("crop_method") || "(blank)"}`);

    // no photo was taken, so both image columns must be empty — a non-empty one would
    // mean an image was written for an encounter that never captured
    for (const c of ["image_path", "original_path"])
      if (get(c)) fail(`${c} should be empty on a short-circuit row, got "${get(c)}"`);

    // the airway answers are the whole evidentiary value of this row
    for (const f of AIRWAY) {
      const v = get(f);
      if (v === "") fail(`airway field ${f} is BLANK — cannot tell "not asked" from "no"`);
      else if (f === "throat_tightness" && v !== "1") fail(`${f} should be 1, got ${v}`);
      else if (f !== "throat_tightness" && v !== "0") fail(`${f} should be 0, got ${v}`);
    }

    for (const c of ["capture_id", "captured_utc", "consent_training", "withdrawal_token"])
      if (!get(c)) fail(`${c} is blank on the short-circuit row`);
    if (get("fitzpatrick")) fail("fitzpatrick must never be written by the app");
  }

  // --- withdrawal must reach a short-circuit row too: it is the only handle, and a
  // row with no image is easy to forget when implementing deletion
  const w = await fetch(`http://localhost:${PORT}/api/capture/withdraw`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token: "SCTEST01" }),
  });
  const wb = await w.json().catch(() => ({}));
  if (wb.matched !== 1) fail(`withdrawal matched ${wb.matched}, expected 1`);
  const left = readFileSync(path.join(dir, "labels.csv"), "utf8").trim().split("\n").length - 1;
  if (left !== 0) fail(`${left} row(s) remain after withdrawal`);
} catch (e) {
  fail(`${e.message}`);
} finally {
  server.kill();
  rmSync(dir, { recursive: true, force: true });
}

console.log(`\nshort-circuit path: ${failed === 0 ? "OK" : `${failed} failure(s)`}`);
process.exit(failed ? 1 : 0);
