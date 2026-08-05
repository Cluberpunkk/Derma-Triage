// cameraQuality.js
// =========================================================================
// CLIENT-SIDE CAPTURE-QUALITY CHECKS (heuristics, on-device only)
//
// These measure THE PHOTO — sharpness, brightness, framing — never the skin.
// They exist so the live scan captures a clean still; they are NOT a clinical
// signal and say nothing about how serious anything is. Do not surface them as
// a "skin score." The engine contract is unchanged: we still hand a single
// still frame to runTriage() exactly as the upload path does.
//
// Everything runs on a small downscaled canvas so the checks are cheap enough
// to run inside a throttled rAF loop without blocking the UI thread.
// =========================================================================

// Width we downscale each sampled frame to before analysis. Small = fast.
const WORK_W = 200;

// Heuristic thresholds. Tuned conservatively for a prototype; expect to adjust
// per device. They gate capture, so err toward asking for a slightly better
// frame rather than accepting a poor one.
export const THRESHOLDS = {
  sharpness: 55,      // variance of Laplacian over the centre region
  brightnessLow: 42,  // mean luma (0–255) below this = too dark
  brightnessHigh: 222,// above this = washed out / glare
  fill: 0.40,         // share of frame edge-energy inside the centre box
};

// Draw the current video frame onto a small work canvas and pull its pixels.
// Returns null if the video has no dimensions yet (not ready).
function sampleFrame(video, canvas) {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) return null;
  const w = WORK_W;
  const h = Math.max(1, Math.round((vh / vw) * w));
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(video, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

// Rec. 601 luma. Cheap and good enough for these heuristics.
function toGray(data) {
  const { width, height } = data;
  const px = data.data;
  const gray = new Float32Array(width * height);
  for (let i = 0, j = 0; i < px.length; i += 4, j++) {
    gray[j] = 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2];
  }
  return gray;
}

// Analyse one frame. Returns { sharpness, brightness, fill } or null if the
// frame isn't ready. `fill` is a rough proxy for "does the subject sit centred
// and fill the frame" — the fraction of total edge-energy inside the centre
// box. It is a heuristic, not a segmentation.
export function analyzeFrame(video, canvas) {
  const img = sampleFrame(video, canvas);
  if (!img) return null;
  const { width: w, height: h } = img;
  const gray = toGray(img);

  // Mean brightness over the whole frame.
  let sum = 0;
  for (let i = 0; i < gray.length; i++) sum += gray[i];
  const brightness = sum / gray.length;

  // Centre box ≈ central 60% — matches the on-screen reticle.
  const cx0 = Math.floor(w * 0.2), cx1 = Math.ceil(w * 0.8);
  const cy0 = Math.floor(h * 0.2), cy1 = Math.ceil(h * 0.8);

  // 4-neighbour Laplacian: sharpness = variance inside the centre box;
  // edge-energy tallies inside vs. outside the box drive the fill ratio.
  let lapSum = 0, lapSqSum = 0, lapN = 0;
  let energyCentre = 0, energyTotal = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const lap = 4 * gray[i] - gray[i - 1] - gray[i + 1] - gray[i - w] - gray[i + w];
      const mag = Math.abs(lap);
      energyTotal += mag;
      const inCentre = x >= cx0 && x < cx1 && y >= cy0 && y < cy1;
      if (inCentre) {
        energyCentre += mag;
        lapSum += lap;
        lapSqSum += lap * lap;
        lapN++;
      }
    }
  }

  const mean = lapN ? lapSum / lapN : 0;
  const sharpness = lapN ? lapSqSum / lapN - mean * mean : 0;
  const fill = energyTotal > 0 ? energyCentre / energyTotal : 0;

  return { sharpness, brightness, fill };
}

// Turn metrics into a pass/fail + one short coaching line. Priority order is
// deliberate: fix lighting first, then framing, then focus. `null` metrics
// (frame not ready) read as "hold on".
export function coach(m) {
  if (!m) return { pass: false, message: "Starting camera…" };
  const { sharpness, brightness, fill } = m;

  if (brightness < THRESHOLDS.brightnessLow)
    return { pass: false, message: "Too dark — find more light" };
  if (brightness > THRESHOLDS.brightnessHigh)
    return { pass: false, message: "Too much glare — reduce direct light" };
  if (fill < THRESHOLDS.fill)
    return { pass: false, message: "Move closer to the skin area" };
  if (sharpness < THRESHOLDS.sharpness)
    return { pass: false, message: "Hold steady — image is blurry" };

  return { pass: true, message: "Looks good — hold steady" };
}

// 0–100 capture-quality score for the single chosen still. Presentation only:
// it describes the photo, NOT the skin or any condition. Blends the three
// normalised sub-scores so a frame has to be lit, framed AND sharp to score high.
export function qualityScore(m) {
  if (!m) return 0;
  const { sharpness, brightness, fill } = m;
  const clamp = (v) => Math.max(0, Math.min(1, v));
  // Brightness: best in the mid range, penalise dark and washed-out.
  const bright = clamp(1 - Math.abs(brightness - 135) / 135);
  const sharp = clamp(sharpness / (THRESHOLDS.sharpness * 2.5));
  const frame = clamp(fill / (THRESHOLDS.fill * 1.6));
  return Math.round((0.4 * sharp + 0.3 * frame + 0.3 * bright) * 100);
}

// Human label for a score, for the "scan quality" chip.
export function qualityLabel(score) {
  if (score >= 80) return "Excellent";
  if (score >= 60) return "Good";
  if (score >= 40) return "Fair";
  return "Low";
}
