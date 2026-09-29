// src/lib/collection.js
// =========================================================================
// PUBLIC-app data-collection helpers, kept separate from the triage engine.
//  - Training-consent copy + version (bump the version if the wording changes;
//    it is stored per case so you know which consent each subject agreed to).
//  - saveCase(): posts one consented still + metadata to the ingestion endpoint.
//  - tryBlurFaces(): optional, on-device face blur BEFORE upload, so a raw face
//    never leaves the device. No-op (and reported unsupported) where the browser
//    lacks FaceDetector.
// This module never shows the user a diagnosis or treatment.
// =========================================================================

export const CONSENT_VERSION = "2026-07-01.v1";

export const CONSENT_COPY = {
  title: "Help build a fairer skin-AI dataset",
  points: [
    "If you agree, the single photo you capture plus the details you enter will be stored securely to build and train a medical-AI dataset focused on Indian skin.",
    "Location data (EXIF/GPS) is removed from the photo. Images are never shown publicly. You can optionally blur faces before sending.",
    "This is used only for building and evaluating the model — not shown to other users, not sold.",
    "You can withdraw and ask for your case to be deleted at any time (contact the study team). Withdrawal removes it from the dataset.",
  ],
  agree: "I agree — contribute my case",
  decline: "Use triage without contributing",
};

// Fire-and-forget: storage never blocks the user's triage result. Returns the
// server response or null on failure.
export async function saveCase(payload) {
  try {
    const res = await fetch("/api/case", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...payload, consent: true, consent_version: CONSENT_VERSION, consent_ts: new Date().toISOString() }),
    });
    if (!res.ok) return null;
    return await res.json();
  } catch { return null; }
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = reject;
    im.src = src;
  });
}

// Optional on-device face blur. Returns { dataUrl, b64, blurred, supported }.
export async function tryBlurFaces(dataUrl) {
  const passthrough = { dataUrl, b64: dataUrl.split(",")[1], blurred: false };
  if (!("FaceDetector" in window)) return { ...passthrough, supported: false };
  try {
    const img = await loadImage(dataUrl);
    const faces = await new window.FaceDetector({ fastMode: true }).detect(img);
    if (!faces.length) return { ...passthrough, supported: true };
    const c = document.createElement("canvas");
    c.width = img.width; c.height = img.height;
    const ctx = c.getContext("2d");
    ctx.drawImage(img, 0, 0);
    ctx.filter = "blur(18px)";
    for (const f of faces) {
      const { x, y, width, height } = f.boundingBox;
      const pad = width * 0.15;
      ctx.drawImage(img, x - pad, y - pad, width + pad * 2, height + pad * 2, x - pad, y - pad, width + pad * 2, height + pad * 2);
    }
    ctx.filter = "none";
    const out = c.toDataURL("image/jpeg", 0.9);
    return { dataUrl: out, b64: out.split(",")[1], blurred: true, supported: true };
  } catch {
    return { ...passthrough, supported: true };
  }
}


/**
 * Post a reticle-path capture into the HOLDOUT schema (data/holdout/labels.csv),
 * which is what ml/eval_holdout.py reads. Distinct from saveCase(), which writes the
 * older SQLite console record — the two serve different consumers and both are kept.
 *
 * Fire-and-forget: storage never blocks or changes what the user sees.
 */
export async function saveHoldoutCapture({ shot, symptoms, symptomResult, withdrawalToken }) {
  try {
    const res = await fetch("/api/capture", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        modelInputB64: shot.dataUrl?.split(",")[1] ?? shot.b64,
        originalB64: shot.originalB64,
        crop: shot.crop || {},
        meta: {
          ...(shot.meta || {}),
          language: "en",
          questionnaireVersion: CONSENT_VERSION,
          // Recorded explicitly rather than inferred from which fields happen to be
          // present, so an upload row's blanks are legible as expected-by-path.
          capturePath: shot.capturePath || (shot.crop?.method ? "scan" : "upload"),
          deviceModel: navigator.userAgent.slice(0, 80),
        },
        quality: shot.quality || {},
        symptoms: symptoms || {},
        symptomResult: symptomResult || { tier: "none", rulesFired: [] },
        consent: { training: true, version: CONSENT_VERSION, withdrawalToken },
      }),
    });
    return res.ok ? await res.json() : null;
  } catch { return null; }
}

/** Emergency short-circuit: no photo taken, but the encounter is still recorded. */
export async function saveShortCircuit({ symptoms, symptomResult, withdrawalToken }) {
  try {
    const res = await fetch("/api/capture/short-circuit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        meta: { language: "en", questionnaireVersion: CONSENT_VERSION,
                deviceModel: navigator.userAgent.slice(0, 80) },
        symptoms, symptomResult,
        consent: { training: true, version: CONSENT_VERSION, withdrawalToken },
      }),
    });
    return res.ok ? await res.json() : null;
  } catch { return null; }
}

/** Short, human-readable code. The ONLY handle a participant has for withdrawal. */
export function newWithdrawalToken() {
  const A = "ABCDEFGHJKMNPQRSTUVWXYZ23456789";   // no confusable characters
  return Array.from({ length: 8 }, () => A[Math.floor(Math.random() * A.length)]).join("");
}
