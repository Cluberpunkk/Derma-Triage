/* src/lib/captureQuality.js
 * ---------------------------------------------------------------------------
 * Quality assessment for the reticle-based capture flow.
 *
 * SHADOW MODE IS THE DEFAULT, AND THAT IS THE POINT
 * You cannot calibrate a gate without data you cannot collect through the gate.
 * The previous thresholds were measured on DermNet and found to be either inert
 * (variance-of-Laplacian 55 -> 0.000 rejections in every ITA bin) or catastrophic
 * (fill 0.40 -> rejects 94-99.6% of images). Neither could have been fixed by
 * retuning against the same corpus.
 *
 * So: MEASURE EVERYTHING, REJECT NOTHING. Every metric is computed and logged on
 * every capture; the UI shows soft guidance ("looks blurry - retake?") and keeps the
 * shot either way. Thresholds get set from the observed distribution of real
 * captures, after the pilot. `MODE` flips to "enforcing" only once that distribution
 * exists.
 *
 * FILL IS NOT RETUNED, IT IS REPLACED
 * The old `fill` metric inferred framing from centre-vs-total edge energy. With a
 * user-placed reticle, framing is DECLARED - the user has said where the lesion is.
 * A heuristic that infers it is solving a problem that no longer exists, and it was
 * the single worst-behaved constant in the system. It is replaced by reticle-relative
 * measures: is the reticle a sensible fraction of the frame, and is there enough
 * surrounding skin left for the ITA annulus.
 *
 * THRESHOLD PROVENANCE — see docs/CONSTANTS_audit.md. Values marked `assumed` have
 * no evidence behind them and exist only to drive UI hints while in shadow mode.
 * ---------------------------------------------------------------------------
 */

/** "shadow" = measure + hint, never block. "enforcing" = block on hard failures. */
export const MODE = "shadow";

export const THRESHOLDS = {
  // MEASURED by ml/degradation_calibration.py: classification accuracy first drops
  // 5% at Gaussian sigma=1.0, where median VoL = 164. The inherited constant was 55 —
  // roughly 3x too permissive, past the point where ~20% of accuracy is already lost.
  // That is why it never fired on DermNet.
  sharpnessAbs: 164,

  // NEITHER SHARPNESS FORM IS FAIR. Measured, at identical synthetic blur:
  //   absolute VoL      1.30x spread across ITA bins; at threshold 164, 61.8% of the
  //                     darkest bin falls below vs 38.2% of the lightest (spread 0.236)
  //   VoL / contrast^2  equalises the light-vs-dark pair almost exactly (1.01x) but
  //                     pushes MID-tones 1.8x above both extremes — a different
  //                     disparity, not a fix. Overall spread gets worse (1.89x).
  // corr(local_contrast, VoL) = -0.018 on undegraded images, so contrast does not
  // predict VoL across natural images and cannot be a general normaliser.
  //
  // Both are logged; NEITHER is enforced. Which (if either) to use must be decided
  // from the pilot's real capture distribution. This is precisely why MODE defaults
  // to "shadow".
  sharpnessRelative: 0.09,
  // MEASURED: accuracy drops 5% below luma ~31 and above luma ~195. The inherited
  // band (42..222) was too tight at the dark end and too loose at the bright end —
  // over-exposure costs far more accuracy than under-exposure (at gain 2.2, accuracy
  // 0.164 vs 0.427 at gain 0.25).
  brightnessLow: 31,
  brightnessHigh: 195,
  // Reticle-relative framing, replacing the removed `fill` heuristic.
  reticleMin: 0.20,           // assumed: below this the lesion is too small to resolve
  reticleMax: 0.75,           // assumed: above this no skin is left for the ITA annulus
  annulusMin: 0.15,           // assumed: fraction of frame outside the reticle
};

/**
 * Connected off-skin regions inside the reticle, via a coarse flood fill on a
 * downscaled mask. Deliberately crude: we only need "one blob or several", not a
 * segmentation. Returns the count of blobs occupying >2% of the reticle area.
 */
function countBlobs(g, size, reticle) {
  const S = 48;                        // work on a 48x48 mask; enough for a count
  const step = size / S;
  const cx = (reticle?.cx ?? 0.5) * S;
  const cy = (reticle?.cy ?? 0.5) * S;
  const rad = ((reticle?.size ?? 0.6) * S) / 2;

  // mean luma inside the reticle, then mark pixels far from it
  let sum = 0, n = 0;
  const inside = new Uint8Array(S * S);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      if (Math.hypot(x - cx, y - cy) > rad) continue;
      inside[y * S + x] = 1;
      sum += g[Math.floor(y * step) * size + Math.floor(x * step)];
      n++;
    }
  }
  if (n < 20) return 0;
  const mu = sum / n;
  let sd = 0;
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      if (!inside[y * S + x]) continue;
      sd += (g[Math.floor(y * step) * size + Math.floor(x * step)] - mu) ** 2;
    }
  }
  sd = Math.sqrt(sd / n);
  const mask = new Uint8Array(S * S);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      if (!inside[y * S + x]) continue;
      const v = g[Math.floor(y * step) * size + Math.floor(x * step)];
      if (Math.abs(v - mu) > 1.1 * sd) mask[y * S + x] = 1;
    }
  }

  const seen = new Uint8Array(S * S);
  const minArea = Math.max(4, Math.round(0.02 * n));
  let blobs = 0;
  const stack = [];
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i] || seen[i]) continue;
    let area = 0;
    stack.length = 0;
    stack.push(i);
    seen[i] = 1;
    while (stack.length) {
      const j = stack.pop();
      area++;
      const jx = j % S, jy = (j / S) | 0;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const nx = jx + dx, ny = jy + dy;
        if (nx < 0 || ny < 0 || nx >= S || ny >= S) continue;
        const k = ny * S + nx;
        if (mask[k] && !seen[k]) { seen[k] = 1; stack.push(k); }
      }
    }
    if (area >= minArea) blobs++;
  }
  return blobs;
}

function grey(video, canvas, size = 200) {
  const c = canvas || document.createElement("canvas");
  c.width = size;
  c.height = size;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(video, 0, 0, size, size);
  const { data } = ctx.getImageData(0, 0, size, size);
  const g = new Float32Array(size * size);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    g[p] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }
  return { g, size };
}

/**
 * Every metric, always. Nothing here decides anything.
 * @param reticle {cx, cy, size} as fractions of the frame, or null.
 */
export function measure(video, canvas, reticle = null) {
  const { g, size } = grey(video, canvas);

  let mean = 0;
  for (let i = 0; i < g.length; i++) mean += g[i];
  mean /= g.length;
  let varSum = 0;
  for (let i = 0; i < g.length; i++) varSum += (g[i] - mean) ** 2;
  const contrast = Math.sqrt(varSum / g.length);

  // variance of Laplacian
  let lMean = 0;
  const lap = new Float32Array((size - 2) * (size - 2));
  for (let y = 1, k = 0; y < size - 1; y++) {
    for (let x = 1; x < size - 1; x++, k++) {
      lap[k] = -4 * g[y * size + x] + g[(y - 1) * size + x] + g[(y + 1) * size + x]
        + g[y * size + x - 1] + g[y * size + x + 1];
      lMean += lap[k];
    }
  }
  lMean /= lap.length;
  let lVar = 0;
  for (let i = 0; i < lap.length; i++) lVar += (lap[i] - lMean) ** 2;
  const sharpness = lVar / lap.length;

  // Contrast-normalised sharpness. NOT a fairness fix — measured to equalise the
  // light-vs-dark pair (1.01x) while creating a mid-tone disparity (1.8x). Logged
  // for the pilot so the choice can be made on real captures. See THRESHOLDS.
  const sharpnessRelative = sharpness / (contrast * contrast + 1e-6);

  // SINGLE-LESION CHECK. Counts distinct off-skin blobs inside the reticle. Two
  // lesions in one frame means the label, the crop and the symptom answers all refer
  // to something ambiguous — and at labelling time a reader cannot tell which lesion
  // the answers describe. Advisory in shadow mode like everything else.
  const blobs = countBlobs(g, size, reticle);

  const reticleFrac = reticle ? reticle.size : null;
  const annulusFrac = reticle ? Math.max(0, 1 - reticle.size * reticle.size) : null;

  return {
    sharpness, sharpnessRelative, brightness: mean, contrast,
    reticleFrac, annulusFrac, blobs,
    // clipping — a proxy for blown highlights / crushed shadows
    clipped: (() => {
      let n = 0;
      for (let i = 0; i < g.length; i++) if (g[i] <= 2 || g[i] >= 253) n++;
      return n / g.length;
    })(),
  };
}

/**
 * Soft guidance for the UI. In shadow mode `blocking` is always false — the shot is
 * kept regardless, and the hint is advisory.
 */
export function assess(m, mode = MODE) {
  const hints = [];
  // Absolute form used for the hint because the contrast-relative form was measured
  // to introduce a mid-tone disparity of its own. Both metrics are logged either way.
  if (m.sharpness < THRESHOLDS.sharpnessAbs) {
    hints.push({ code: "blur", text: "Looks blurry — hold steady and try again?" });
  }
  if (m.brightness < THRESHOLDS.brightnessLow) {
    hints.push({ code: "dark", text: "Quite dark — more light would help." });
  }
  if (m.brightness > THRESHOLDS.brightnessHigh || m.clipped > 0.10) {
    hints.push({ code: "bright", text: "Very bright — try avoiding direct light." });
  }
  if (m.reticleFrac != null) {
    if (m.reticleFrac < THRESHOLDS.reticleMin) {
      hints.push({ code: "reticle-small", text: "Move closer, or enlarge the circle." });
    } else if (m.reticleFrac > THRESHOLDS.reticleMax) {
      hints.push({ code: "reticle-large", text: "Leave some normal skin visible around it." });
    }
  }
  if (m.blobs > 1) {
    hints.push({ code: "multi-lesion",
                 text: "Looks like more than one spot — photograph one at a time." });
  }
  return {
    metrics: m,
    hints,
    mode,
    // SHADOW MODE: never blocks. The capture is always kept and always logged, so the
    // pilot produces the distribution these thresholds will eventually be set from.
    blocking: mode === "enforcing" && hints.some((h) => h.code === "blur"),
    passed: hints.length === 0,
  };
}
