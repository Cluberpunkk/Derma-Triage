/* src/components/CaptureReview.jsx
 * ---------------------------------------------------------------------------
 * "Does this look right?" between capture and submit.
 *
 * Cheap, and it is the only point where a user can catch a bad capture before it
 * enters the dataset. Without it every mis-framed shot becomes a row a dermatologist
 * later has to mark ungradable — reader time is the scarcest resource in the clinic
 * study, so a retake here is far cheaper than a rejection there.
 *
 * It shows WHAT WAS CAPTURED, not what the model thinks. No class name, no
 * confidence, no urgency — quality hints only, and those describe the photo, never
 * the skin.
 *
 * The diagnostics block is shown to PILOT testers (they are debugging the capture
 * path) and hidden from ordinary users, who have no use for a downsample factor.
 * ---------------------------------------------------------------------------
 */
import { MODE } from "../lib/captureQuality.js";

export default function CaptureReview({ shot, showDiagnostics = false, onRetake, onAccept }) {
  if (!shot) return null;
  const { crop = {}, meta = {}, quality = {} } = shot;
  const hints = quality.hints || [];

  return (
    <section className="review">
      <h2>Does this look right?</h2>

      <div className="review-shot">
        <img src={shot.modelInput?.dataUrl || shot.dataUrl} alt="the area you captured" />
      </div>

      <ul className="review-check">
        <li>Is the affected area clearly in the picture?</li>
        <li>Is it in focus, and lit well enough to see?</li>
        {meta.captureMode !== "wide" && <li>Is there some normal skin visible around it?</li>}
      </ul>

      {hints.length > 0 && (
        <div className="scan-hints">
          {hints.map((h) => <span key={h} className="hint">{labelFor(h)}</span>)}
          {MODE === "shadow" && (
            <span className="hint ok">You can submit anyway — we keep it either way.</span>
          )}
        </div>
      )}

      {showDiagnostics && (
        <details className="review-diag">
          <summary>Capture diagnostics (pilot)</summary>
          <dl>
            <dt>Source resolution</dt><dd>{meta.sourceResolution || "—"}</dd>
            <dt>Downsample factor</dt>
            <dd>{meta.downsampleFactor ?? "—"} <em>(lower is better)</em></dd>
            <dt>Crop method</dt><dd>{crop.method || "—"}</dd>
            <dt>Crop size (native px)</dt><dd>{crop.w ?? "—"}</dd>
            <dt>Localiser agreed</dt>
            <dd>{crop.localiserAgreed === undefined ? "n/a"
              : crop.localiserAgreed ? `yes (IoU ${crop.localiserIou})`
              : `no (IoU ${crop.localiserIou}) — your framing was used`}</dd>
            <dt>Sharpness / brightness</dt>
            <dd>{Math.round(quality.sharpness ?? 0)} / {Math.round(quality.brightness ?? 0)}</dd>
            <dt>Distinct spots detected</dt><dd>{quality.blobs ?? "—"}</dd>
          </dl>
        </details>
      )}

      <div className="sf-actions">
        <button onClick={onRetake}>Retake</button>
        <button className="primary" onClick={onAccept}>Use this photo</button>
      </div>
    </section>
  );
}

function labelFor(code) {
  return {
    blur: "Looks a bit blurry",
    dark: "Quite dark",
    bright: "Very bright",
    "reticle-small": "The circle was small",
    "reticle-large": "Little normal skin visible",
    "multi-lesion": "More than one spot in frame",
  }[code] || code;
}
