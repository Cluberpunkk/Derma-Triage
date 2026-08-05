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
