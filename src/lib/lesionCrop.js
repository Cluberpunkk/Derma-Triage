/* src/lib/lesionCrop.js
 * ---------------------------------------------------------------------------
 * Native-resolution, lesion-centred cropping.
 *
 * THE DEFECT THIS FIXES
 * The training/eval path resizes the short side to 255 and centre-crops to 224.
 * On a 12MP phone frame that keeps 0.41% of the pixels and throws away a 13.5x
 * linear downsample before the model sees anything — and the SK/BCC distinction we
 * fail on lives in fine texture (milia-like cysts, arborising vessels).
 *
 * So: locate the lesion in the FULL-RESOLUTION frame, crop a square around it at
 * native pixels, and only then downsample that crop to the model's input size. The
 * full-resolution original is preserved alongside, so a better model later can be
 * re-run on the same capture without re-collecting.
 *
 * LESION LOCALISATION — deliberately simple, and stated as a heuristic
 * No segmentation model runs on-device. Skin dominates the frame, so the modal LAB
 * colour is skin and the lesion is the region least like it. Two channels are
 * combined to find that region: colour distance from modal skin, and colour-
 * INDEPENDENT edge energy. The second channel is not optional — see the measured
 * fairness defect documented at the combination step.
 *
 * This is a heuristic; it fails on whole-limb rashes and on lesions that match skin
 * tone closely. `confidence` reports how peaked the result is, and callers fall back
 * to a centre crop when it is low rather than trusting a bad box.
 *
 * All processing is on-device. Nothing here uploads.
 * ---------------------------------------------------------------------------
 */

// Analysis runs on a downscaled copy — locating the lesion needs no detail, only
// the box. The CROP is then applied at native resolution.
const ANALYSIS_W = 256;
const MIN_CROP_FRAC = 0.25;   // never crop tighter than this fraction of the short side
const MAX_CROP_FRAC = 0.95;
const PAD = 1.6;              // lesion box is padded by this factor for context

function toLab(r, g, b) {
  const f = (v) => {
    v /= 255;
    return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  };
  const [R, G, B] = [f(r), f(g), f(b)];
  const x = (0.4124 * R + 0.3576 * G + 0.1805 * B) / 0.95047;
  const y = 0.2126 * R + 0.7152 * G + 0.0722 * B;
  const z = (0.0193 * R + 0.1192 * G + 0.9505 * B) / 1.08883;
  const k = (v) => (v > 0.008856 ? Math.cbrt(v) : 7.787 * v + 16 / 116);
  const [fx, fy, fz] = [k(x), k(y), k(z)];
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

/**
 * Locate the lesion in a full-resolution source.
 * @returns {{x,y,w,h,confidence,method}} box in SOURCE pixel coordinates.
 */
export function findLesion(source, srcW, srcH) {
  const scale = ANALYSIS_W / Math.max(srcW, srcH);
  const aw = Math.max(32, Math.round(srcW * scale));
  const ah = Math.max(32, Math.round(srcH * scale));

  const c = document.createElement("canvas");
  c.width = aw;
  c.height = ah;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(source, 0, 0, aw, ah);
  const { data } = ctx.getImageData(0, 0, aw, ah);

  // modal skin colour, from a coarse LAB histogram (8x8x8 over a,b and L)
  const bins = new Map();
  const lab = new Float32Array(aw * ah * 3);
  for (let i = 0, p = 0; i < data.length; i += 4, p += 3) {
    const [L, A, B] = toLab(data[i], data[i + 1], data[i + 2]);
    lab[p] = L; lab[p + 1] = A; lab[p + 2] = B;
    const key = `${Math.round(L / 12)}_${Math.round(A / 6)}_${Math.round(B / 6)}`;
    bins.set(key, (bins.get(key) || 0) + 1);
  }
  let best = null; let bestN = -1;
  for (const [k, n] of bins) if (n > bestN) { bestN = n; best = k; }
  const [bl, ba, bb] = best.split("_").map(Number);
  const skin = [bl * 12, ba * 6, bb * 6];

  // Two channels, combined 50/50:
  //   colour distance from modal skin  — works when the lesion contrasts in hue/lightness
  //   local edge energy                — colour-INDEPENDENT, so it does not weaken as
  //                                      skin darkens and LAB headroom shrinks
  //
  // Colour distance alone was measured (ml/ita_audit.py) to fall back to a blind
  // centre crop 69% of the time on the darkest ITA bin vs 56% on the lightest — a
  // capture-path fairness defect, on top of falling back most of the time for
  // everyone. Adding the edge channel cut overall fallback 3-4x (to 0.12-0.23) and
  // roughly halved the tone gap (+0.134 -> +0.079). A residual gap remains.
  const colour = new Float32Array(aw * ah);
  const edge = new Float32Array(aw * ah);
  let maxC = 0, maxE = 0;
  for (let i = 0, p = 0; i < colour.length; i++, p += 3) {
    const d = Math.hypot(lab[p] - skin[0], lab[p + 1] - skin[1], lab[p + 2] - skin[2]);
    colour[i] = d;
    if (d > maxC) maxC = d;
  }
  for (let y = 0; y < ah; y++) {
    for (let x = 0; x < aw; x++) {
      const i = y * aw + x;
      const L = lab[i * 3];
      const gx = Math.abs(L - lab[(y * aw + Math.max(0, x - 1)) * 3]);
      const gy = Math.abs(L - lab[(Math.max(0, y - 1) * aw + x) * 3]);
      edge[i] = gx + gy;
      if (edge[i] > maxE) maxE = edge[i];
    }
  }
  const dist = new Float32Array(aw * ah);
  let maxD = 0;
  for (let i = 0; i < dist.length; i++) {
    dist[i] = 0.5 * (colour[i] / (maxC + 1e-8)) + 0.5 * (edge[i] / (maxE + 1e-8));
    if (dist[i] > maxD) maxD = dist[i];
  }
  if (maxD <= 1e-6) {
    return { ...centreBox(srcW, srcH), confidence: 0, method: "centre-fallback" };
  }

  // threshold at 55% of max distance, take the weighted centroid + spread
  const thr = maxD * 0.55;
  let sx = 0, sy = 0, n = 0;
  for (let y = 0; y < ah; y++) {
    for (let x = 0; x < aw; x++) {
      const d = dist[y * aw + x];
      if (d >= thr) { sx += x; sy += y; n++; }
    }
  }
  if (n < 12) return { ...centreBox(srcW, srcH), confidence: 0, method: "centre-fallback" };

  const cx = sx / n, cy = sy / n;
  let vx = 0, vy = 0;
  for (let y = 0; y < ah; y++) {
    for (let x = 0; x < aw; x++) {
      if (dist[y * aw + x] >= thr) { vx += (x - cx) ** 2; vy += (y - cy) ** 2; }
    }
  }
  const spread = Math.sqrt(Math.max(vx, vy) / n);

  // confidence: how concentrated the off-skin mass is. A whole-frame rash spreads
  // it out and scores low, which is correct — a tight crop would be wrong there.
  const frac = n / (aw * ah);
  const confidence = Math.max(0, Math.min(1, (1 - frac / 0.45) * Math.min(1, 24 / (spread + 1e-6))));

  const shortSide = Math.min(srcW, srcH);
  let side = Math.round((spread * PAD * 2) / scale);
  side = Math.max(Math.round(shortSide * MIN_CROP_FRAC),
                  Math.min(Math.round(shortSide * MAX_CROP_FRAC), side));
  const ox = Math.round(cx / scale), oy = Math.round(cy / scale);
  return {
    x: Math.max(0, Math.min(srcW - side, Math.round(ox - side / 2))),
    y: Math.max(0, Math.min(srcH - side, Math.round(oy - side / 2))),
    w: side, h: side, confidence, method: "colour+edge",
  };
}

/** Reticle given as fractions of the frame -> source-pixel square box. */
function reticleBox({ cx = 0.5, cy = 0.5, size = 0.6 }, w, h) {
  const side = Math.round(Math.min(w, h) * size);
  return {
    x: Math.max(0, Math.min(w - side, Math.round(cx * w - side / 2))),
    y: Math.max(0, Math.min(h - side, Math.round(cy * h - side / 2))),
    w: side, h: side,
  };
}

/** Intersection-over-union, for logging localiser-vs-reticle agreement. */
function iou(a, b) {
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const uni = a.w * a.h + b.w * b.h - inter;
  return uni > 0 ? inter / uni : 0;
}

function centreBox(w, h) {
  const s = Math.min(w, h);
  return { x: Math.round((w - s) / 2), y: Math.round((h - s) / 2), w: s, h: s };
}

/**
 * Capture at native resolution and produce BOTH artifacts.
 *
 * @returns {{
 *   original: {dataUrl, width, height},   full-resolution frame, never downsampled
 *   modelInput: {dataUrl, width, height}, lesion crop resized to modelSize
 *   crop: {x,y,w,h,confidence,method},
 *   sourceResolution: string
 * }}
 */
export function captureNative(video, { modelSize = 224, quality = 0.95,
                                       minConfidence = 0.25,
                                       reticle = null, mode = "close" } = {}) {
  const srcW = video.videoWidth;
  const srcH = video.videoHeight;
  if (!srcW || !srcH) throw new Error("video has no frame yet");

  // 1. full-resolution original — this is what gets preserved
  const full = document.createElement("canvas");
  full.width = srcW;
  full.height = srcH;
  full.getContext("2d").drawImage(video, 0, 0, srcW, srcH);

  // 2. Choose the crop box.
  //
  // RETICLE IS PRIMARY. A user-aligned reticle is tone-independent BY CONSTRUCTION —
  // it cannot degrade on darker skin, because no colour statistic enters it. The
  // colour+edge localiser was measured to fall back 3-4x more often before the edge
  // channel, with a residual tone gradient, so it is demoted to a REFINEMENT that
  // applies only when it is confident AND broadly agrees with where the user aimed.
  //
  // `mode` is declared by the user, not inferred. A whole-limb rash is a "wide"
  // capture; inferring that from low localiser confidence conflated "the user is
  // photographing something diffuse" with "the heuristic failed".
  const auto = findLesion(full, srcW, srcH);
  let box;
  if (mode === "wide") {
    // diffuse/whole-limb: the frame IS the region of interest. No localisation.
    box = { ...centreBox(srcW, srcH), confidence: 1, method: "wide-declared" };
  } else if (reticle) {
    const r = reticleBox(reticle, srcW, srcH);
    const agrees = auto.confidence >= minConfidence && iou(r, auto) >= 0.35;
    box = agrees
      ? { ...auto, method: "reticle+refined" }
      : { ...r, confidence: 1, method: "reticle" };
    box.localiserAgreed = agrees;
    box.localiserIou = +iou(r, auto).toFixed(3);
    box.localiserConfidence = +auto.confidence.toFixed(3);
  } else {
    // no reticle (upload path): localiser with centre fallback, as before
    box = auto.confidence >= minConfidence
      ? auto
      : { ...centreBox(srcW, srcH), confidence: auto.confidence, method: "centre-fallback" };
  }

  // 3. crop at NATIVE pixels, then downsample only that crop
  const out = document.createElement("canvas");
  out.width = modelSize;
  out.height = modelSize;
  const octx = out.getContext("2d");
  octx.imageSmoothingQuality = "high";
  octx.drawImage(full, box.x, box.y, box.w, box.h, 0, 0, modelSize, modelSize);

  return {
    original: { dataUrl: full.toDataURL("image/jpeg", quality), width: srcW, height: srcH },
    modelInput: { dataUrl: out.toDataURL("image/jpeg", quality),
                  width: modelSize, height: modelSize },
    crop: box,
    sourceResolution: `${srcW}x${srcH}`,
    // Downsample factor applied to the crop: source pixels per model pixel.
    // LOWER IS BETTER. A tight lesion crop on a 12MP frame gives ~1.8 (fine detail
    // survives); a centre-crop fallback on the same frame gives ~13.5 (it does not).
    // Named `downsampleFactor` because `effectiveScale` read as if higher were better.
    downsampleFactor: +(box.w / modelSize).toFixed(2),
  };
}

/**
 * Camera constraints.
 *
 * `forceHighRes` asks for 4096x3072. That is NOT obviously the right thing: asking
 * for a specific size can make a browser select a different sensor mode — often a
 * cropped or upscaled one — rather than the native full-frame output it would pick
 * on its own. Which behaviour a device shows is an empirical question per device, so
 * both are exposed and getSettings() is recorded for each.
 *
 * Default is OFF: let the browser choose, then measure what we got.
 */
export function cameraConstraints(facing = "environment", forceHighRes = false) {
  const video = { facingMode: { ideal: facing } };
  if (forceHighRes) {
    video.width = { ideal: 4096 };
    video.height = { ideal: 3072 };
  }
  return { video, audio: false };
}

/** @deprecated use cameraConstraints(facing, true) */
export function highResConstraints(facing = "environment") {
  return cameraConstraints(facing, true);
}
