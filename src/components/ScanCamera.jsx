// ScanCamera.jsx
// =========================================================================
// Live capture on the NATIVE-RESOLUTION RETICLE path.
//
// getUserMedia is only ever called after the consent step in App is accepted
// (this component mounts only then). Frames are analysed on-device and thrown
// away; no video is persisted or uploaded.
//
// WHAT CHANGED FROM THE OLD PATH
// It used to resize the whole frame to <=1100px and hand that to the model. On a
// 12MP phone frame that discarded 99.6% of the pixels before inference - and the
// SK-vs-BCC distinction we fail on lives in fine texture. Now: the user places a
// RETICLE, we crop at NATIVE pixels around it, and only that crop is downsampled.
// The full-resolution original is kept alongside, so a better model later can be
// re-run on the same capture without re-collecting.
//
// WHY THE RETICLE LEADS, NOT THE LOCALISER
// Measured (ml/ita_audit.py): colour-based localisation fell back to a blind centre
// crop 69% of the time on the darkest ITA bin vs 56% on the lightest. An added edge
// channel cut that to 20%/12%, but a residual tone gradient remains (+0.079, CI
// excludes zero; OPEN-1 in the architecture record). A user-placed reticle is
// tone-independent BY CONSTRUCTION, so it leads; the localiser only refines when it
// is confident AND agrees with where the user aimed.
//
// QUALITY IS MEASURED, NEVER ENFORCED
// captureQuality.MODE is "shadow": every metric is logged, hints are advisory, and
// the shot is kept regardless. Thresholds get set from the pilot's real
// distribution - a gate cannot be calibrated on data it prevents you collecting.
//
// NOTHING FROM THE MODEL APPEARS HERE. No class name, no confidence, no urgency.
// This component never calls the model at all. Quality hints describe THE PHOTO,
// never the skin.
// =========================================================================

import React, { useCallback, useEffect, useRef, useState } from "react";
import { captureNative, highResConstraints } from "../lib/lesionCrop.js";
import { MODE, assess, measure } from "../lib/captureQuality.js";

const ANALYZE_EVERY_MS = 150;   // ~7 analyses/sec, off the render path

export default function ScanCamera({
  initialFacing = "environment", captureMode = "close", onModeChange,
  onCapture, onUseUpload, onCancel,
}) {
  const videoRef = useRef(null);
  const workRef = useRef(null);
  const streamRef = useRef(null);
  const rafRef = useRef(0);
  const lastRunRef = useRef(0);
  const reticleRef = useRef({ cx: 0.5, cy: 0.5, size: 0.55 });

  const [facing, setFacing] = useState(initialFacing);
  const [status, setStatus] = useState("starting");
  const [error, setError] = useState("");
  const [reticle, setReticle] = useState(reticleRef.current);
  const [live, setLive] = useState(null);
  const [retry, setRetry] = useState(0);

  reticleRef.current = reticle;   // read inside the rAF loop without re-subscribing

  const stopStream = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const s = await navigator.mediaDevices.getUserMedia(highResConstraints(facing));
        if (cancelled) { s.getTracks().forEach((t) => t.stop()); return; }
        streamRef.current = s;
        const v = videoRef.current;
        v.srcObject = s;
        await v.play();
        setStatus("live");

        const loop = (ts) => {
          rafRef.current = requestAnimationFrame(loop);
          if (ts - lastRunRef.current < ANALYZE_EVERY_MS) return;
          lastRunRef.current = ts;
          if (!videoRef.current?.videoWidth) return;
          try {
            setLive(assess(measure(videoRef.current, workRef.current, reticleRef.current)));
          } catch { /* a dropped frame must not kill the loop */ }
        };
        rafRef.current = requestAnimationFrame(loop);
      } catch (e) {
        if (cancelled) return;
        setStatus("denied");
        setError(e?.name === "NotAllowedError"
          ? "Camera permission was declined."
          : `Could not start the camera (${e?.name || "unknown"}).`);
      }
    })();
    return () => { cancelled = true; stopStream(); };
  }, [facing, retry, stopStream]);

  const take = useCallback(() => {
    const v = videoRef.current;
    if (!v?.videoWidth) return;
    const shot = captureNative(v, {
      modelSize: 224,
      // In wide mode the frame IS the region of interest — the user has declared it,
      // so no reticle is passed and no localisation runs.
      reticle: captureMode === "wide" ? null : reticleRef.current,
      mode: captureMode,
    });
    const q = assess(measure(v, workRef.current, reticleRef.current));
    stopStream();
    onCapture({
      modelInput: shot.modelInput,      // native crop, downsampled to 224
      original: shot.original,          // full-resolution frame, never downsampled
      crop: shot.crop,                  // incl. localiserAgreed / localiserIou
      meta: {
        sourceResolution: shot.sourceResolution,
        downsampleFactor: shot.downsampleFactor,   // LOWER IS BETTER
        captureMode,
        deviceModel: navigator.userAgent.slice(0, 80),
      },
      quality: { ...q.metrics, passed: q.passed, hints: q.hints.map((h) => h.code) },
    });
  }, [captureMode, onCapture, stopStream]);

  if (status === "denied") {
    return (
      <section className="scan">
        <h2>Camera unavailable</h2>
        <p>{error}</p>
        <p className="sf-note">
          You can allow camera access in your browser settings, or upload a photo
          instead. Uploads skip the reticle, so the crop is less precise.
        </p>
        <div className="sf-actions">
          <button onClick={() => {
            setStatus("starting"); setError(""); setRetry((n) => n + 1);
          }}>Try again</button>
          <button className="primary" onClick={onUseUpload}>Upload a photo</button>
          {onCancel && <button onClick={onCancel}>Cancel</button>}
        </div>
      </section>
    );
  }

  // The reticle is PLACEABLE, not fixed at centre. Back, scalp and awkward sites
  // are a large share of real presentations and were uncapturable while it sat in
  // the middle of the frame.
  const place = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const cx = Math.min(0.95, Math.max(0.05, (e.clientX - r.left) / r.width));
    const cy = Math.min(0.95, Math.max(0.05, (e.clientY - r.top) / r.height));
    setReticle((s) => ({ ...s, cx, cy }));
  };

  const pct = (n) => `${n * 100}%`;
  const wide = captureMode === "wide";
  return (
    <section className="scan">
      <div className="scan-stage"
           onPointerDown={wide ? undefined : place}
           onPointerMove={(e) => { if (!wide && e.buttons) place(e); }}>
        <video ref={videoRef} playsInline muted autoPlay
               className={facing === "user" ? "mirrored" : ""} />
        {wide ? (
          <div className="wide-frame"><span>Whole area — no circle</span></div>
        ) : (
          <div className="reticle" style={{
            left: pct(reticle.cx - reticle.size / 2),
            top: pct(reticle.cy - reticle.size / 2),
            width: pct(reticle.size),
            height: pct(reticle.size),
          }} />
        )}
      </div>

      <div className="scan-mode" role="radiogroup" aria-label="What are you photographing?">
        <button className={!wide ? "on" : ""} onClick={() => onModeChange?.("close")}
                aria-pressed={!wide}>One spot</button>
        <button className={wide ? "on" : ""} onClick={() => onModeChange?.("wide")}
                aria-pressed={wide}>A whole area / rash</button>
      </div>

      <div className="scan-hints" aria-live="polite">
        {live?.hints?.length
          ? live.hints.map((h) => <span key={h.code} className="hint">{h.text}</span>)
          : <span className="hint ok">{wide
              ? "Fit the whole affected area in the frame."
              : "Tap where the spot is, then size the circle."}</span>}
      </div>

      {!wide && <label className="scan-slider">
        <span>Circle size</span>
        <input type="range" min="0.2" max="0.8" step="0.01" value={reticle.size}
               onChange={(e) => setReticle((r) => ({ ...r, size: +e.target.value }))} />
      </label>}

      <div className="sf-actions">
        <button onClick={() => setFacing((f) => (f === "user" ? "environment" : "user"))}>
          Flip camera
        </button>
        <button className="primary" onClick={take} disabled={status !== "live"}>
          Take photo
        </button>
        <button onClick={onUseUpload}>Upload instead</button>
      </div>

      {MODE === "shadow" && (
        <p className="sf-note">Hints are advisory — your photo is kept either way.</p>
      )}
      <canvas ref={workRef} hidden />
    </section>
  );
}
