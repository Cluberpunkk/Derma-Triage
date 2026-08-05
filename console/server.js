// console/server.js
// =========================================================================
// LABELLING CONSOLE BACKEND — a SEPARATE Express app on its own port (:8788),
// fully decoupled from the public triage proxy (server/proxy.js, :8787). The
// public app has no route that reaches here; deploying the public app does not
// require this process to run.
//
// Every route except /console-api/login is behind requireAuth. Images are
// streamed through an auth-gated route — they are never placed on a public
// static path and never get a public URL.
//
//   Run:  CONSOLE_PASSWORD=... npm run console:server
// =========================================================================

import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import { requireAuth, passwordOk, setSession, clearSession } from "./lib/auth.js";
import {
  queue, getCase, getLabels, upsertLabel, setAdjudication,
  getAdjudication, withdrawCase,
} from "../server/lib/db.js";
import { readImage, DATA_DIR } from "../server/lib/storage.js";
import { validateLabel, DIAGNOSES, CONFIRMATION_METHODS, FITZPATRICK, MONK, BODY_SITES, PRIOR_STEROID } from "./shared/vocab.js";
import { dashboard } from "./lib/stats.js";
import { writeExport } from "./lib/export.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
app.use(express.json({ limit: "2mb" }));

if (!process.env.CONSOLE_PASSWORD) {
  console.error("Refusing to start: set CONSOLE_PASSWORD (the console admin credential).");
  process.exit(1);
}

// ---- auth ---------------------------------------------------------------
app.post("/console-api/login", (req, res) => {
  const { password, labeller } = req.body || {};
  const name = String(labeller || "").trim();
  if (!passwordOk(password)) return res.status(401).json({ error: "bad credentials" });
  if (!name) return res.status(400).json({ error: "labeller name required" });
  // labeller_id is a slug of the chosen identity; every label is attributed to it.
  const labellerId = name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "labeller";
  setSession(res, labellerId);
  res.json({ ok: true, labellerId });
});

app.post("/console-api/logout", (req, res) => { clearSession(res); res.json({ ok: true }); });

app.get("/console-api/me", requireAuth, (req, res) => {
  res.json({ labellerId: req.labellerId });
});

// The controlled vocabulary the UI renders.
app.get("/console-api/vocab", requireAuth, (_req, res) => {
  res.json({ DIAGNOSES, CONFIRMATION_METHODS, FITZPATRICK, MONK, BODY_SITES, PRIOR_STEROID });
});

// ---- queue + case detail ------------------------------------------------
app.get("/console-api/queue", requireAuth, (req, res) => {
  const filter = String(req.query.filter || "unlabelled");
  res.json({ cases: queue(req.labellerId, filter) });
});

app.get("/console-api/case/:id", requireAuth, (req, res) => {
  const c = getCase(req.params.id);
  if (!c || c.withdrawn) return res.status(404).json({ error: "not found" });
  const labels = getLabels(c.case_id);
  res.json({
    case: { ...c, symptoms: safeJson(c.symptoms) },
    labels,
    mine: labels.find((l) => l.labeller_id === req.labellerId) || null,
    adjudication: getAdjudication(c.case_id) || null,
  });
});

// Auth-gated image stream — the only way to view an image; no public URL.
app.get("/console-api/image/:id", requireAuth, (req, res) => {
  const c = getCase(req.params.id);
  if (!c || c.withdrawn) return res.status(404).end();
  try {
    res.setHeader("Content-Type", "image/jpeg");
    res.setHeader("Cache-Control", "private, no-store");
    res.end(readImage(c.image_path));
  } catch { res.status(404).end(); }
});

// ---- labelling ----------------------------------------------------------
app.post("/console-api/case/:id/label", requireAuth, (req, res) => {
  const c = getCase(req.params.id);
  if (!c || c.withdrawn) return res.status(404).json({ error: "not found" });
  const l = req.body || {};
  // Autosave-friendly: partial saves are allowed, but we ALWAYS validate against
  // the vocabulary and report which mandatory fields are still missing so the UI
  // can gate "confirmed". A save never invents ground truth.
  const { ok, errors } = validateLabel(l);
  const saved = upsertLabel(c.case_id, req.labellerId, l);
  res.json({ ok: true, label: saved, confirmed: ok, errors });
});

app.post("/console-api/case/:id/adjudicate", requireAuth, (req, res) => {
  const { label_id } = req.body || {};
  const labels = getLabels(req.params.id);
  if (!labels.some((l) => l.label_id === label_id)) return res.status(400).json({ error: "unknown label" });
  setAdjudication(req.params.id, label_id, req.labellerId);
  res.json({ ok: true });
});

// Withdrawal (DPDP right-to-withdraw) — hides the case from queue and export.
app.post("/console-api/case/:id/withdraw", requireAuth, (req, res) => {
  withdrawCase(req.params.id);
  res.json({ ok: true });
});

// ---- dashboard + export -------------------------------------------------
app.get("/console-api/dashboard", requireAuth, (_req, res) => res.json(dashboard()));

app.post("/console-api/export", requireAuth, (req, res) => {
  // Exports live alongside the data (under DATA_DIR), so they honour a mounted
  // volume and are covered by the data/ gitignore rule.
  const outDir = path.join(DATA_DIR, "exports", `export_${new Date().toISOString().replace(/[:.]/g, "-")}`);
  try { res.json({ ok: true, ...writeExport(outDir) }); }
  catch (e) { console.error(e); res.status(500).json({ error: "export_failed" }); }
});

// ---- static console web (built) ----------------------------------------
// Serves the console SPA from its own build. This is the ONLY app served here.
const webDist = path.resolve(__dirname, "web/dist");
app.use(express.static(webDist));
app.get(/^\/(?!console-api\/).*/, (_req, res) => res.sendFile(path.join(webDist, "index.html")));

function safeJson(s) { try { return JSON.parse(s); } catch { return s; } }

const PORT = process.env.CONSOLE_PORT || 8788;
app.listen(PORT, () => console.log(`labelling console listening on :${PORT} (separate from public proxy :8787)`));
