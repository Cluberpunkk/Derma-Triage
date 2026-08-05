// server/proxy.js
// Minimal backend that holds your API key and forwards image + details to the
// vision model. Run with:  npm run proxy
//
// THIS is the file you change when you move from the placeholder engine to a
// real (fine-tuned) dermatology model: point the fetch at your model instead.

import express from "express";
import { CLINICAL_SYSTEM_PROMPT } from "../src/lib/clinicalPrompt.js";
import { insertCase, newId } from "./lib/db.js";
import { putImage } from "./lib/storage.js";
import { b64ToBuffer, stripExif } from "./lib/images.js";
import { writeCapture, writeShortCircuit, recordRejection, recordDropoff,
         withdraw, stats as holdoutStats } from "./lib/holdout.js";

const app = express();
app.use(express.json({ limit: "12mb" }));

// -------------------------------------------------------------------------
// Training-data ingestion. Called by the PUBLIC app ONLY when the user has
// consented to contribute. EXIF is stripped before the image touches disk, and
// the image is written outside any web-served path (see storage.js). This lives
// on the public backend so the public app can collect data with the labelling
// console absent; nothing here reads or exposes the console.
// -------------------------------------------------------------------------
app.post("/api/case", (req, res) => {
  const b = req.body || {};
  if (!b.imageB64) return res.status(400).json({ error: "missing image" });
  if (!b.consent) return res.status(400).json({ error: "consent required to store" });
  try {
    const caseId = newId("case");
    const clean = stripExif(b64ToBuffer(b.imageB64)); // strip GPS/EXIF guarantee
    const imagePath = putImage(caseId, clean);
    insertCase({
      case_id: caseId,
      created_at: new Date().toISOString(),
      image_path: imagePath,
      body_location: b.body_location ?? null,
      duration: b.duration ?? null,
      symptoms: JSON.stringify(Array.isArray(b.symptoms) ? b.symptoms : []),
      recent_change: b.recent_change ?? null,
      prior_cream: b.prior_cream ?? null,
      steroid: b.steroid ?? null,
      skin_tone_self: b.skin_tone_self ?? null,
      consent: 1,
      consent_ts: b.consent_ts || new Date().toISOString(),
      consent_version: b.consent_version ?? null,
      exif_stripped: 1,
      face_blurred: b.face_blurred ? 1 : 0,
    });
    res.json({ ok: true, case_id: caseId });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "store_failed" });
  }
});

// ---------------------------------------------------------------------------
// In-app capture -> holdout schema. The collection instrument IS the production
// capture path, so we stop measuring on a distribution we do not ship.
// ---------------------------------------------------------------------------
app.post("/api/capture", (req, res) => {
  const { modelInputB64, originalB64, meta, symptoms, symptomResult, quality, crop, consent } = req.body || {};
  if (!consent?.training) {
    return res.status(400).json({ error: "training consent required to store a capture" });
  }
  if (!modelInputB64) return res.status(400).json({ error: "modelInputB64 required" });
  try {
    const out = writeCapture({ modelInputB64, originalB64, meta, symptoms, symptomResult, quality, crop, consent });
    res.json({ ok: true, ...out });
  } catch (e) {
    console.error("capture write failed:", e.message);
    res.status(500).json({ error: "capture write failed" });
  }
});

// Rejections are recorded WITHOUT an image. A gate that rejects most real attempts
// is a usability failure, and it is invisible unless counted.
app.post("/api/capture/rejected", (req, res) => {
  try {
    recordRejection(req.body || {});
    res.json({ ok: true });
  } catch (e) {
    res.status(500).json({ error: "rejection log failed" });
  }
});

// Emergency short-circuit: no photo was taken, but the encounter is still recorded.
app.post("/api/capture/short-circuit", (req, res) => {
  if (!req.body?.consent?.training) {
    return res.status(400).json({ error: "training consent required" });
  }
  try {
    res.json({ ok: true, ...writeShortCircuit(req.body) });
  } catch (e) {
    console.error("short-circuit write failed:", e.message);
    res.status(500).json({ error: "short-circuit write failed" });
  }
});

app.post("/api/capture/dropoff", (req, res) => {
  try { recordDropoff(req.body || {}); res.json({ ok: true }); }
  catch { res.status(500).json({ error: "dropoff log failed" }); }
});

// Consent withdrawal (DPDP). Token-only: we hold no other identifier by design.
app.post("/api/capture/withdraw", (req, res) => {
  const { token, dryRun } = req.body || {};
  if (!token) return res.status(400).json({ error: "withdrawal token required" });
  try {
    const r = withdraw(String(token), { dryRun: Boolean(dryRun) });
    res.json({ ok: true, ...r });
  } catch (e) {
    console.error("withdrawal failed:", e.message);
    res.status(500).json({ error: "withdrawal failed" });
  }
});

app.get("/api/capture/stats", (_req, res) => res.json(holdoutStats()));

app.post("/api/triage", async (req, res) => {
  const { imageB64, details } = req.body || {};
  if (!imageB64) return res.status(400).json({ error: "missing image" });
  if (!process.env.ANTHROPIC_API_KEY) {
    return res.status(500).json({ error: "ANTHROPIC_API_KEY is not set" });
  }

  try {
    const r = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": process.env.ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        // Placeholder engine. Swap for a stronger vision model or your own
        // fine-tuned model once the data pipeline (see README) produces one.
        model: "claude-sonnet-4-6",
        max_tokens: 1200,
        system: CLINICAL_SYSTEM_PROMPT,
        messages: [{
          role: "user",
          content: [
            { type: "image", source: { type: "base64", media_type: "image/jpeg", data: imageB64 } },
            { type: "text", text: `Patient-reported details:\n${details}\n\nReturn ONLY the JSON.` },
          ],
        }],
      }),
    });

    const data = await r.json();
    const text = (data.content || [])
      .filter((b) => b.type === "text")
      .map((b) => b.text)
      .join("");
    res.json({ text });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "engine_error" });
  }
});

const PORT = process.env.PORT || 8787;
app.listen(PORT, () => console.log(`triage proxy listening on :${PORT}`));
