import React, { useState, useRef } from "react";
import { runTriage } from "./lib/triageEngine.js";
import ScanCamera from "./components/ScanCamera.jsx";
import CaptureReview from "./components/CaptureReview.jsx";
import { SURFACE_7_URGENCY_VERDICT } from "./lib/surfaces.js";
import { qualityLabel } from "./lib/cameraQuality.js";
import { CONSENT_COPY, saveCase, tryBlurFaces } from "./lib/collection.js";

const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

.dx *{box-sizing:border-box;}
.dx{
  --bg:#EEF2F1;--paper:#FBFCFC;--ink:#10302F;--ink2:#3D5654;
  --line:#D6DEDC;--teal:#2E7D74;--teal-d:#1F5C55;
  --amber:#B8742A;--amber-bg:#FBF1E4;
  --red:#B23A3A;--red-bg:#FAEDED;
  --green:#3C7A53;--green-bg:#EAF3EC;--mut:#6B807E;
  font-family:'Inter',system-ui,sans-serif;color:var(--ink);
  background:radial-gradient(1200px 480px at 82% -12%,#E1ECEA 0%,transparent 60%),var(--bg);
  min-height:100vh;padding:28px 16px 56px;line-height:1.5;-webkit-font-smoothing:antialiased;
}
.dx .wrap{max-width:680px;margin:0 auto;}
.dx .topline{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:7px;flex-wrap:wrap;}
.dx .wordmark{font-weight:700;font-size:22px;letter-spacing:-0.02em;}
.dx .wordmark b{color:var(--teal);font-weight:700;}
.dx .proto-tag{font-family:'IBM Plex Mono',monospace;font-size:10px;font-weight:600;letter-spacing:0.14em;text-transform:uppercase;color:var(--amber);border:1px solid var(--amber);border-radius:999px;padding:3px 9px;background:var(--amber-bg);}
.dx .sub{color:var(--ink2);font-size:14.5px;margin:0 0 14px;}
.dx .engine{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--mut);border-top:1px dashed var(--line);border-bottom:1px dashed var(--line);padding:8px 0;margin:0 0 20px;display:flex;gap:9px;align-items:center;}
.dx .dot{width:7px;height:7px;border-radius:50%;background:var(--amber);box-shadow:0 0 0 3px var(--amber-bg);flex:none;}

.dx .card{background:var(--paper);border:1px solid var(--line);border-radius:16px;padding:20px;box-shadow:0 1px 0 rgba(16,48,47,.03),0 14px 32px -26px rgba(16,48,47,.3);}
.dx .eyebrow{font-family:'IBM Plex Mono',monospace;font-size:10.5px;font-weight:600;letter-spacing:.16em;text-transform:uppercase;color:var(--mut);margin:0 0 6px;}
.dx .lead{font-size:13.5px;color:var(--ink2);margin:0 0 16px;}

.dx .upload{border:1.5px dashed var(--line);border-radius:12px;padding:24px;text-align:center;cursor:pointer;transition:.15s;background:#fff;}
.dx .upload:hover{border-color:var(--teal);background:#FAFEFD;}
.dx .upload-t{font-weight:600;font-size:15px;}
.dx .upload-s{font-size:12.5px;color:var(--mut);margin-top:3px;}
.dx .preview{position:relative;border-radius:12px;overflow:hidden;border:1px solid var(--line);}
.dx .preview img{display:block;width:100%;max-height:300px;object-fit:cover;}
.dx .preview button{position:absolute;top:10px;right:10px;font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:600;background:rgba(16,48,47,.8);color:#fff;border:none;border-radius:8px;padding:6px 10px;cursor:pointer;}

.dx .field{margin-top:18px;}
.dx .label{font-size:13px;font-weight:600;color:var(--ink2);margin:0 0 8px;}
.dx .label .opt{font-weight:500;color:var(--mut);font-size:12px;margin-left:5px;}
.dx .seg{display:flex;flex-wrap:wrap;gap:7px;}
.dx .ob{font-size:13px;font-weight:500;padding:7px 12px;border-radius:9px;border:1px solid var(--line);background:#fff;color:var(--ink2);cursor:pointer;transition:.12s;}
.dx .ob:hover{border-color:var(--teal);}
.dx .ob.on{background:var(--teal);border-color:var(--teal);color:#fff;}
.dx .ob.chip.on{background:var(--ink);border-color:var(--ink);}

.dx .btn{width:100%;margin-top:22px;font-weight:600;font-size:15.5px;color:#fff;background:var(--ink);border:none;border-radius:12px;padding:15px;cursor:pointer;position:relative;overflow:hidden;transition:.15s;}
.dx .btn:hover:not(:disabled){background:var(--teal-d);}
.dx .btn:disabled{opacity:.42;cursor:not-allowed;}
.dx .btn.loading{background:var(--teal-d);}
.dx .btn.loading::after{content:"";position:absolute;inset:0;background:linear-gradient(110deg,transparent 30%,rgba(255,255,255,.18) 50%,transparent 70%);transform:translateX(-100%);animation:sweep 1.2s infinite;}
@keyframes sweep{to{transform:translateX(100%);}}
.dx .disc{font-size:11.5px;color:var(--mut);text-align:center;margin-top:14px;line-height:1.55;}
.dx .err{background:var(--red-bg);border:1px solid #E6B7B7;color:var(--red);border-radius:12px;padding:13px;font-size:13.5px;margin-top:16px;}

.dx .res{animation:fade .4s ease both;}
@keyframes fade{from{opacity:0;transform:translateY(8px);}to{opacity:1;transform:none;}}
.dx .uband{border-radius:14px;padding:15px 17px;margin-bottom:8px;display:flex;gap:13px;align-items:flex-start;border:1px solid;}
.dx .uband .ic{font-family:'IBM Plex Mono',monospace;font-weight:600;font-size:10.5px;letter-spacing:.07em;flex:none;padding:5px 8px;border-radius:7px;text-transform:uppercase;}
.dx .uband.routine{background:var(--green-bg);border-color:#BCD8C4;color:var(--green);}
.dx .uband.routine .ic{background:var(--green);color:#fff;}
.dx .uband.soon{background:var(--amber-bg);border-color:#E6CBA4;color:var(--amber);}
.dx .uband.soon .ic{background:var(--amber);color:#fff;}
.dx .uband.urgent{background:var(--red-bg);border-color:#E6B7B7;color:var(--red);}
.dx .uband.urgent .ic{background:var(--red);color:#fff;}
.dx .uband .ut{font-weight:700;font-size:15px;margin-bottom:2px;}
.dx .uband .us{font-size:13.5px;line-height:1.5;}

.dx .block{margin-top:20px;}
.dx .block-h{font-family:'IBM Plex Mono',monospace;font-size:10.5px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--mut);margin:0 0 11px;}
.dx .cond{border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:9px;background:#fff;}
.dx .cond-top{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:5px;}
.dx .cond-name{font-weight:600;font-size:15px;}
.dx .pill{font-family:'IBM Plex Mono',monospace;font-size:9.5px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;padding:3px 8px;border-radius:999px;white-space:nowrap;}
.dx .pill.more{background:#E5F0ED;color:var(--teal-d);}
.dx .pill.possible{background:#EDF1F0;color:var(--ink2);}
.dx .pill.less{background:#F1F3F2;color:var(--mut);}
.dx .cond-about{font-size:13.5px;color:var(--ink2);margin-bottom:6px;}
.dx .cond-why{font-size:12.5px;color:var(--mut);font-style:italic;}

.dx .flag{display:flex;gap:9px;font-size:13.5px;color:var(--ink2);padding:9px 12px;background:var(--red-bg);border:1px solid #EAC6C6;border-radius:10px;margin-bottom:7px;}
.dx .flag .fm{color:var(--red);flex:none;font-weight:700;}
.dx .qitem{display:flex;gap:10px;font-size:13.5px;color:var(--ink2);padding:8px 0;border-bottom:1px solid var(--line);}
.dx .qitem:last-child{border-bottom:none;}
.dx .qn{font-family:'IBM Plex Mono',monospace;color:var(--teal);font-weight:600;flex:none;}
.dx .note{font-size:13px;color:var(--ink2);background:#F4F7F6;border:1px solid var(--line);border-left:3px solid var(--teal);border-radius:0 10px 10px 0;padding:12px 14px;margin-top:18px;line-height:1.55;}
.dx .retake{font-size:14px;color:var(--ink2);background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px;line-height:1.55;}

.dx .why{margin-top:14px;border:1px solid var(--line);border-radius:10px;overflow:hidden;}
.dx .why summary{cursor:pointer;list-style:none;padding:11px 14px;font-size:13px;font-weight:600;color:var(--ink2);display:flex;align-items:center;gap:8px;background:#fff;}
.dx .why summary::-webkit-details-marker{display:none;}
.dx .why summary::before{content:"+";font-family:'IBM Plex Mono',monospace;color:var(--teal);font-weight:700;}
.dx .why[open] summary::before{content:"–";}
.dx .why .wb{padding:0 14px 14px;font-size:12.5px;color:var(--mut);line-height:1.6;}

.dx .restart{display:block;margin:20px auto 0;font-family:'IBM Plex Mono',monospace;font-size:12px;font-weight:600;letter-spacing:.04em;background:none;border:1px solid var(--line);color:var(--ink2);border-radius:9px;padding:9px 16px;cursor:pointer;}
.dx .restart:hover{border-color:var(--teal);color:var(--teal-d);}
.dx .footer{margin-top:24px;padding-top:16px;border-top:1px solid var(--line);font-size:11.5px;color:var(--mut);text-align:center;line-height:1.6;}
.dx :focus-visible{outline:2px solid var(--teal);outline-offset:2px;}
@media (prefers-reduced-motion:reduce){.dx *{animation:none!important;}}
@media(max-width:520px){.dx .wordmark{font-size:20px;}}

/* ---- capture mode toggle -------------------------------------------- */
.dx .modes{display:flex;gap:6px;background:#EEF3F1;border:1px solid var(--line);border-radius:11px;padding:4px;margin:0 0 16px;}
.dx .modes button{flex:1;font-size:13px;font-weight:600;color:var(--ink2);background:none;border:none;border-radius:8px;padding:9px;cursor:pointer;transition:.12s;}
.dx .modes button.on{background:#fff;color:var(--ink);box-shadow:0 1px 0 rgba(16,48,47,.04),0 6px 14px -10px rgba(16,48,47,.35);}

/* ---- consent gate --------------------------------------------------- */
.dx .consent{border:1px solid var(--line);border-radius:12px;padding:18px;background:#fff;}
.dx .consent-t{font-weight:600;font-size:15px;margin:0 0 8px;}
.dx .consent-l{font-size:13px;color:var(--ink2);line-height:1.6;margin:0 0 8px;display:flex;gap:9px;}
.dx .consent-l .ck{color:var(--teal);font-weight:700;flex:none;}
.dx .consent-note{font-size:12px;color:var(--mut);margin:10px 0 0;line-height:1.55;}

/* ---- live scan ------------------------------------------------------ */
.dx .scan{}
.dx .scan-stage{position:relative;border-radius:14px;overflow:hidden;background:#0C1F1E;aspect-ratio:3/4;max-height:420px;margin:0 auto;}
.dx .scan-video{width:100%;height:100%;object-fit:cover;display:block;}
.dx .scan-video.mirror{transform:scaleX(-1);}
.dx .scan-wait{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#CFE0DD;font-size:13.5px;font-family:'IBM Plex Mono',monospace;}
.dx .scan-reticle{position:absolute;left:12%;top:14%;right:12%;bottom:22%;pointer-events:none;}
.dx .scan-reticle .c{position:absolute;width:26px;height:26px;border:2.5px solid rgba(255,255,255,.9);}
.dx .scan-reticle .tl{top:0;left:0;border-right:none;border-bottom:none;border-radius:8px 0 0 0;}
.dx .scan-reticle .tr{top:0;right:0;border-left:none;border-bottom:none;border-radius:0 8px 0 0;}
.dx .scan-reticle .bl{bottom:0;left:0;border-right:none;border-top:none;border-radius:0 0 0 8px;}
.dx .scan-reticle .br{bottom:0;right:0;border-left:none;border-top:none;border-radius:0 0 8px 0;}
.dx .scan-sweep{position:absolute;left:0;right:0;height:3px;background:linear-gradient(90deg,transparent,rgba(120,230,214,.95),transparent);box-shadow:0 0 14px 2px rgba(120,230,214,.5);animation:scanline 1.4s ease-in-out infinite;}
@keyframes scanline{0%{top:2%}50%{top:96%}100%{top:2%}}
.dx .scan-stage.scanning .scan-reticle .c{border-color:rgba(120,230,214,.95);}
.dx .scan-coach{position:absolute;left:50%;bottom:12px;transform:translateX(-50%);white-space:nowrap;font-size:13px;font-weight:600;color:#fff;background:rgba(12,31,30,.72);border:1px solid rgba(255,255,255,.14);border-radius:999px;padding:7px 14px;backdrop-filter:blur(3px);}
.dx .scan-coach.ok{background:rgba(46,125,116,.85);border-color:rgba(120,230,214,.5);}
.dx .scan-flip{position:absolute;top:10px;right:10px;font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:600;color:#fff;background:rgba(12,31,30,.72);border:1px solid rgba(255,255,255,.16);border-radius:8px;padding:6px 10px;cursor:pointer;}
.dx .scan-controls{display:flex;align-items:center;gap:10px;margin-top:14px;}
.dx .scan-btn{font-weight:600;font-size:14.5px;border-radius:11px;padding:13px 16px;cursor:pointer;border:1px solid var(--line);transition:.14s;}
.dx .scan-btn.ghost{background:#fff;color:var(--ink2);}
.dx .scan-btn.ghost:hover{border-color:var(--teal);color:var(--teal-d);}
.dx .scan-btn.shutter{flex:1;background:var(--ink);color:#fff;border-color:var(--ink);}
.dx .scan-btn.shutter:hover:not(.disabled){background:var(--teal-d);border-color:var(--teal-d);}
.dx .scan-btn.shutter.disabled{opacity:.45;cursor:not-allowed;}
.dx .scan-auto{display:flex;align-items:center;gap:6px;font-size:12.5px;font-weight:600;color:var(--ink2);cursor:pointer;user-select:none;}
.dx .scan-auto input{width:15px;height:15px;accent-color:var(--teal);}
.dx .scan-privacy{font-size:11.5px;color:var(--mut);line-height:1.55;margin:12px 0 0;text-align:center;}
.dx .scan-fallback{border:1px solid #E6CBA4;background:var(--amber-bg);border-radius:12px;padding:18px;}
.dx .scan-fallback-t{font-weight:600;font-size:15px;color:var(--amber);margin-bottom:6px;}
.dx .scan-fallback-s{font-size:13px;color:var(--ink2);line-height:1.55;margin-bottom:13px;}
.dx .scan-fallback-row{display:flex;gap:9px;flex-wrap:wrap;}

/* ---- scanned frame in result --------------------------------------- */
.dx .shot{position:relative;border-radius:12px;overflow:hidden;border:1px solid var(--line);margin-bottom:14px;}
.dx .shot img{display:block;width:100%;max-height:300px;object-fit:cover;}
.dx .shot-focus{position:absolute;left:16%;top:16%;right:16%;bottom:16%;border:2px solid rgba(120,230,214,.9);border-radius:12px;box-shadow:0 0 0 100vmax rgba(12,31,30,.14);pointer-events:none;}
.dx .shot-label{position:absolute;left:10px;bottom:10px;font-family:'IBM Plex Mono',monospace;font-size:9.5px;font-weight:600;letter-spacing:.04em;color:#fff;background:rgba(12,31,30,.72);border-radius:6px;padding:4px 8px;max-width:80%;line-height:1.35;}
.dx .qchip{position:absolute;top:10px;right:10px;display:flex;align-items:center;gap:6px;font-family:'IBM Plex Mono',monospace;font-size:10px;font-weight:600;color:#fff;background:rgba(12,31,30,.72);border-radius:7px;padding:5px 9px;}
.dx .qchip .qd{width:7px;height:7px;border-radius:50%;flex:none;}
.dx .qchip .qd.excellent,.dx .qchip .qd.good{background:#5FD6C2;}
.dx .qchip .qd.fair{background:#E4B25E;}
.dx .qchip .qd.low{background:#E08585;}

/* ---- training-contribution note + blur toggle ---------------------- */
.dx .contrib-note{font-size:12px;color:var(--ink2);background:#F4F7F6;border:1px solid var(--line);border-radius:9px;padding:9px 12px;margin:0 0 4px;line-height:1.5;}
.dx .contrib-note .linkish{background:none;border:none;color:var(--teal-d);font-weight:600;font-size:12px;cursor:pointer;padding:0;text-decoration:underline;}
.dx .blurtoggle{display:flex;gap:9px;align-items:flex-start;margin-top:18px;font-size:13px;font-weight:600;color:var(--ink2);cursor:pointer;}
.dx .blurtoggle input{width:16px;height:16px;margin-top:2px;accent-color:var(--teal);flex:none;}
`;

const LOCS = ["Face", "Scalp", "Neck", "Chest / Back", "Arm / Hand", "Leg / Foot", "Groin", "Nails", "Other"];
const DURS = ["< 1 week", "1–4 weeks", "1–6 months", "> 6 months"];
const SYMS = ["Itching", "Pain", "Burning", "Bleeding", "Spreading", "Scaling", "Color change"];
const TONES = [
  { l: "Very fair", f: "I" }, { l: "Fair", f: "II" }, { l: "Medium", f: "III" },
  { l: "Olive / Tan", f: "IV" }, { l: "Brown", f: "V" }, { l: "Deep brown", f: "VI" },
];
const CHANGES = ["No real change", "New / just appeared", "Growing or changing", "Not sure"];

const U = {
  routine: { tag: "Routine", title: "Routine — not an emergency" },
  soon: { tag: "Soon", title: "Worth seeing a doctor soon" },
  urgent: { tag: "Urgent", title: "See a doctor promptly" },
};

function resizeImage(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const im = new Image();
      im.onload = () => {
        const max = 1100;
        let w = im.width, h = im.height;
        if (w > max || h > max) { const r = Math.min(max / w, max / h); w = Math.round(w * r); h = Math.round(h * r); }
        const c = document.createElement("canvas");
        c.width = w; c.height = h;
        c.getContext("2d").drawImage(im, 0, 0, w, h);
        const dataUrl = c.toDataURL("image/jpeg", 0.9);
        resolve({ dataUrl, b64: dataUrl.split(",")[1] });
      };
      im.onerror = reject;
      im.src = e.target.result;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

export default function App() {
  const [img, setImg] = useState(null);
  const [loc, setLoc] = useState("");
  const [dur, setDur] = useState("");
  const [sym, setSym] = useState([]);
  const [applied, setApplied] = useState("");
  const [steroid, setSteroid] = useState("");
  const [tone, setTone] = useState("");
  const [change, setChange] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [mode, setMode] = useState("scan");   // "scan" (live camera) | "upload"
  const [scanning, setScanning] = useState(false); // consent accepted, camera live
  // "close" = one spot with a placeable reticle; "wide" = a whole area / rash, where
  // the frame IS the region and no localisation runs. Declared by the user, never
  // inferred from low localiser confidence — those are different situations.
  const [captureMode, setCaptureMode] = useState("close");
  // Capture is held here until the user confirms it. Review-before-submit is the only
  // point a bad frame can be caught before it becomes a row a reader must mark
  // ungradable, and reader time is the scarcest resource in the clinic study.
  const [pending, setPending] = useState(null);
  const [trainingConsent, setTrainingConsent] = useState(null); // null | true | false
  const [blurFaces, setBlurFaces] = useState(true);
  const fileRef = useRef(null);

  const toggleSym = (s) => setSym((cur) => cur.includes(s) ? cur.filter((x) => x !== s) : [...cur, s]);

  // Front camera for face-area scans, rear otherwise; the user can still flip.
  const initialFacing = ["Face", "Scalp", "Neck"].includes(loc) ? "user" : "environment";

  // ScanCamera now returns the native-resolution reticle payload. Adapt it to the
  // { dataUrl, b64 } shape the rest of App uses, and carry the extra artifacts
  // (full-resolution original, crop diagnostics, capture metadata) through to the
  // holdout row. The upload path still produces the old shape and is unchanged.
  const onCapture = (shot) => {
    setScanning(false);
    setPending(shot);          // -> CaptureReview, not straight into the flow
  };

  const acceptPending = () => {
    const { modelInput, original, crop, meta, quality } = pending;
    setImg({
      dataUrl: modelInput.dataUrl,
      b64: modelInput.dataUrl.split(",")[1],
      originalB64: original?.dataUrl?.split(",")[1],
      crop, meta, quality,
    });
    setPending(null);
    setResult(null);
    setError("");
  };

  const onFile = async (e) => {
    const f = e.target.files && e.target.files[0];
    if (!f) return;
    setError("");
    try { const r = await resizeImage(f); setImg(r); setResult(null); }
    catch { setError("Couldn't read that image. Try a different photo."); }
  };

  const canRun = img && loc && dur && !loading;

  const analyze = async () => {
    setLoading(true); setError(""); setResult(null);
    const tObj = TONES.find((t) => t.l === tone);
    const details = [
      `Body location: ${loc}`,
      `Duration: ${dur}`,
      `Symptoms reported: ${sym.length ? sym.join(", ") : "none"}`,
      `Recent change: ${change || "not specified"}`,
      `Already applied a cream/ointment: ${applied || "not specified"}${applied === "yes" && steroid ? ` — steroid or combination cream: ${steroid}` : ""}`,
      `Self-reported skin tone: ${tone ? `${tone} (approx Fitzpatrick ${tObj ? tObj.f : "?"})` : "not specified"}`,
    ].join("\n");

    // Optional on-device face blur — applied to whatever we send AND store, so a
    // raw face never leaves the device when the user asked to blur it.
    let sendB64 = img.b64, faceBlurred = false;
    if (trainingConsent === true && blurFaces) {
      try { const r = await tryBlurFaces(img.dataUrl); sendB64 = r.b64; faceBlurred = r.blurred; } catch { /* keep original */ }
    }

    // Contribute the case to the training dataset only if the user consented.
    // Fire-and-forget: storage never blocks or changes the triage result.
    if (trainingConsent === true) {
      saveCase({
        imageB64: sendB64,
        body_location: loc, duration: dur, symptoms: sym,
        recent_change: change, prior_cream: applied, steroid,
        skin_tone_self: tone, face_blurred: faceBlurred,
      });
    }

    try {
      // Surface #7 is off, so nothing from runTriage is displayed. Do not call it
    // either: sending the image to a third-party API to produce output no one sees
    // is a needless disclosure and a needless cost. Gating the RENDER without
    // gating the CALL would have left the image leaving the device regardless.
    const parsed = SURFACE_7_URGENCY_VERDICT
      ? await runTriage({ imageB64: sendB64, details })
      : { gradable: true, pilot_mode: true };
      setResult(parsed);
    } catch (err) {
      setError("The engine couldn't return a clean reading. Try again, or use a sharper, well-lit photo.");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => { setResult(null); setError(""); };

  return (
    <div className="dx">
      <style>{CSS}</style>
      <div className="wrap">
        <div className="topline">
          <div className="wordmark">Derma<b>Triage</b></div>
          <div className="proto-tag">Prototype</div>
        </div>
        <p className="sub">Take a photo, answer a few questions, and get a read on what to discuss with a dermatologist — and how soon.</p>
        <div className="engine">
          <span className="dot" />
          engine: general vision model (placeholder) · not a validated dermatology model
        </div>

        {!result && trainingConsent === null && (
          <div className="card">
            <div className="eyebrow">Before you start · How your case is used</div>
            <div className="consent-t" style={{ fontSize: 17, marginBottom: 10 }}>{CONSENT_COPY.title}</div>
            {CONSENT_COPY.points.map((p, i) => (
              <div className="consent-l" key={i}><span className="ck">✓</span><span>{p}</span></div>
            ))}
            <button className="btn" onClick={() => setTrainingConsent(true)}>{CONSENT_COPY.agree}</button>
            <button className="restart" style={{ margin: "12px auto 0" }} onClick={() => setTrainingConsent(false)}>{CONSENT_COPY.decline}</button>
            <p className="disc">You can withdraw a contributed case later by contacting the study team. Declining still lets you use the triage tool — nothing is stored.</p>
          </div>
        )}

        {!result && trainingConsent !== null && (
          <div className="card">
            <div className="eyebrow">Step 1 · Photo + a few details</div>
            <p className="lead">Use a sharp, well-lit, close photo of the skin area. Fill in what you can — location and how long are enough to start.</p>
            <div className="contrib-note">
              {trainingConsent
                ? <>Thanks — this case will help build the dataset. <button className="linkish" onClick={() => setTrainingConsent(false)}>Stop contributing</button></>
                : <>Not contributing to the dataset — triage only. <button className="linkish" onClick={() => setTrainingConsent(true)}>Contribute instead</button></>}
            </div>

            {img ? (
              <div className="preview">
                <img src={img.dataUrl} alt="skin area" />
                <button onClick={() => setImg(null)}>Replace</button>
              </div>
            ) : (
              <>
                <div className="modes" role="tablist" aria-label="How to add a photo">
                  <button role="tab" aria-selected={mode === "scan"} className={mode === "scan" ? "on" : ""} onClick={() => { setMode("scan"); setScanning(false); }}>Live scan</button>
                  <button role="tab" aria-selected={mode === "upload"} className={mode === "upload" ? "on" : ""} onClick={() => { setMode("upload"); setScanning(false); }}>Upload a photo</button>
                </div>

                {mode === "scan" && !scanning && (
                  <div className="consent">
                    <div className="consent-t">Before we turn on the camera</div>
                    <div className="consent-l"><span className="ck">✓</span><span>The camera is used only to capture one still photo of the skin area, for information.</span></div>
                    <div className="consent-l"><span className="ck">✓</span><span>Frames are checked on your device. Video is never recorded, saved, or uploaded.</span></div>
                    <div className="consent-l"><span className="ck">✓</span><span>Only the single photo you capture is sent, and only for this session.</span></div>
                    <button className="btn" style={{ marginTop: 16 }} onClick={() => setScanning(true)}>Allow camera &amp; start scan</button>
                    <p className="consent-note">Prefer not to use the camera? Switch to “Upload a photo” above.</p>
                  </div>
                )}

                {mode === "scan" && pending && (
                  <CaptureReview
                    shot={pending}
                    showDiagnostics
                    onRetake={() => { setPending(null); setScanning(true); }}
                    onAccept={acceptPending}
                  />
                )}

                {mode === "scan" && scanning && !pending && (
                  <ScanCamera
                    captureMode={captureMode}
                    onModeChange={setCaptureMode}
                    initialFacing={initialFacing}
                    onCapture={onCapture}
                    onCancel={() => setScanning(false)}
                    onUseUpload={() => { setScanning(false); setMode("upload"); }}
                  />
                )}

                {mode === "upload" && (
                  <div className="upload" onClick={() => fileRef.current && fileRef.current.click()}>
                    <div className="upload-t">Add a photo of the skin area</div>
                    <div className="upload-s">Tap to choose or take a picture</div>
                  </div>
                )}
              </>
            )}
            <input ref={fileRef} type="file" accept="image/*" capture="environment" onChange={onFile} style={{ display: "none" }} />

            <div className="field">
              <div className="label">Where on the body?</div>
              <div className="seg">
                {LOCS.map((o) => (
                  <button key={o} className={"ob" + (loc === o ? " on" : "")} onClick={() => setLoc(o)}>{o}</button>
                ))}
              </div>
            </div>

            <div className="field">
              <div className="label">How long has it been there?</div>
              <div className="seg">
                {DURS.map((o) => (
                  <button key={o} className={"ob" + (dur === o ? " on" : "")} onClick={() => setDur(o)}>{o}</button>
                ))}
              </div>
            </div>

            <div className="field">
              <div className="label">Any symptoms?<span className="opt">select any</span></div>
              <div className="seg">
                {SYMS.map((o) => (
                  <button key={o} className={"ob chip" + (sym.includes(o) ? " on" : "")} onClick={() => toggleSym(o)}>{o}</button>
                ))}
              </div>
            </div>

            <div className="field">
              <div className="label">Has it changed recently?<span className="opt">optional</span></div>
              <div className="seg">
                {CHANGES.map((o) => (
                  <button key={o} className={"ob" + (change === o ? " on" : "")} onClick={() => setChange(o)}>{o}</button>
                ))}
              </div>
            </div>

            <div className="field">
              <div className="label">Already applied any cream or ointment?<span className="opt">optional</span></div>
              <div className="seg">
                {["yes", "no"].map((o) => (
                  <button key={o} className={"ob" + (applied === o ? " on" : "")} onClick={() => { setApplied(o); if (o === "no") setSteroid(""); }}>{o === "yes" ? "Yes" : "No"}</button>
                ))}
              </div>
            </div>

            {applied === "yes" && (
              <div className="field">
                <div className="label">Was it a steroid or combination cream?<span className="opt">if you know</span></div>
                <div className="seg">
                  {["Yes", "No", "Not sure"].map((o) => (
                    <button key={o} className={"ob" + (steroid === o ? " on" : "")} onClick={() => setSteroid(o)}>{o}</button>
                  ))}
                </div>
              </div>
            )}

            <div className="field">
              <div className="label">Your skin tone<span className="opt">helps flag where the read is less reliable</span></div>
              <div className="seg">
                {TONES.map((o) => (
                  <button key={o.l} className={"ob" + (tone === o.l ? " on" : "")} onClick={() => setTone(o.l)}>{o.l}</button>
                ))}
              </div>
            </div>

            {trainingConsent === true && (
              <label className="blurtoggle">
                <input type="checkbox" checked={blurFaces} onChange={(e) => setBlurFaces(e.target.checked)} />
                <span>Blur faces in the stored photo <span className="opt">recommended for face areas · done on your device</span></span>
              </label>
            )}

            <button className={"btn" + (loading ? " loading" : "")} disabled={!canRun} onClick={analyze}>
              {loading ? "Reading the image…" : "Analyze photo"}
            </button>
            {error && <div className="err">{error}</div>}
            <p className="disc">For information only. Not a diagnosis and not a medical device. Always see a qualified dermatologist for a real assessment.</p>
          </div>
        )}

        {result && (
          <div className="card res">
            {result.gradable === false ? (
              <div className="retake">
                <strong>The photo needs a retake.</strong><br />
                {result.image_note || "Try a sharper, closer, well-lit photo of just the skin area."}
              </div>
            ) : (
              <>
                {img && (
                  <div className="shot">
                    <img src={img.dataUrl} alt="captured skin area" />
                    <div className="shot-focus" aria-hidden="true" />
                    {img.quality && (
                      <div className="qchip" title="Quality of the captured photo — not a measure of the skin">
                        <span className={"qd " + qualityLabel(img.quality.score).toLowerCase()} />
                        Scan quality: {qualityLabel(img.quality.score)}
                      </div>
                    )}
                    <div className="shot-label">Illustrative framing only — not a precise medical localization</div>
                  </div>
                )}

                {/* SURFACE #7 — user-facing urgency verdict. OFF for the pilot.
                     See src/lib/surfaces.js and docs/ARCHITECTURE_decision.md. */}
                {SURFACE_7_URGENCY_VERDICT && (<>
                <div className={"uband " + result.urgency}>
                  <span className="ic">{(U[result.urgency] || U.soon).tag}</span>
                  <div>
                    <div className="ut">{(U[result.urgency] || U.soon).title}</div>
                    <div className="us">{result.urgency_reason}</div>
                  </div>
                </div>

                {Array.isArray(result.red_flags) && result.red_flags.length > 0 && (
                  <div className="block">
                    <div className="block-h">Features worth getting checked</div>
                    {result.red_flags.map((f, i) => (
                      <div className="flag" key={i}><span className="fm">!</span><span>{f}</span></div>
                    ))}
                  </div>
                )}

                {Array.isArray(result.possible_conditions) && result.possible_conditions.length > 0 && (
                  <div className="block">
                    <div className="block-h">Possibilities to discuss with a doctor</div>
                    {result.possible_conditions.map((c, i) => (
                      <div className="cond" key={i}>
                        <div className="cond-top">
                          <span className="cond-name">{c.name}</span>
                          <span className={"pill " + (c.likelihood === "more likely" ? "more" : c.likelihood === "less likely" ? "less" : "possible")}>{c.likelihood}</span>
                        </div>
                        {c.about && <div className="cond-about">{c.about}</div>}
                        {c.why && <div className="cond-why">Why considered: {c.why}</div>}
                      </div>
                    ))}
                  </div>
                )}

                {Array.isArray(result.ask_your_doctor) && result.ask_your_doctor.length > 0 && (
                  <div className="block">
                    <div className="block-h">Bring these to your visit</div>
                    {result.ask_your_doctor.map((q, i) => (
                      <div className="qitem" key={i}><span className="qn">{String(i + 1).padStart(2, "0")}</span><span>{q}</span></div>
                    ))}
                  </div>
                )}

                </>)}
                {!SURFACE_7_URGENCY_VERDICT && (
                  <div className="note">
                    Photo received. This pilot build does not show an assessment —
                    it is collecting capture quality and questionnaire answers only.
                  </div>
                )}
                {SURFACE_7_URGENCY_VERDICT && result.limitations
                  && <div className="note">{result.limitations}</div>}

                <details className="why">
                  <summary>Why this tool never suggests a treatment or product</summary>
                  <div className="wb">By design, it never names a cream, medicine, or product. In India, self-applied steroid and combination creams are a leading driver of worsened, hard-to-treat skin infections, and a photo alone can't safely tell those cases apart. Any product suggestion here would risk real harm. Treatment is a decision for your dermatologist after an in-person exam.</div>
                </details>
              </>
            )}

            <button className="restart" onClick={reset}>Check another photo</button>
          </div>
        )}

        <div className="footer">
          Prototype for testing the experience only · Not a medical device · Not a diagnosis · Always consult a qualified dermatologist · In an emergency, contact local emergency services.
        </div>
      </div>
    </div>
  );
}
