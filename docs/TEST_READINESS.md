# Test-readiness note — pilot capture flow

What a tester sees, what is real, what is stubbed, what lands in the holdout row.

---

## ⚠️ The negative claim does NOT currently hold. Read this first.

You asked me to confirm that no class name, confidence score or urgency verdict
reaches the user, and that the only escalation shown comes from the symptom gate.

**Verified for the trained model — true. Verified for the app as a whole — false.**

| | status |
|---|---|
| The trained checkpoint (`runs_final/best.pt`) | **never wired to the UI at all.** No class name, confidence, capture score or urgency from it reaches any screen. `ScanCamera` and `SymptomFlow` do not import the model, the triage engine, or anything downstream of them. |
| The **pre-existing LLM triage path** (`runTriage` → `clinicalPrompt.js`) | **renders an urgency band and named conditions to the user today.** |

`src/App.jsx` currently renders, from the LLM result:

```
line 481  <div className={"uband " + result.urgency}>     <- urgency verdict
line 498  result.possible_conditions.map(...)             <- named conditions
line 489  result.red_flags.map(...)                       <- escalation not from the gate
```

This predates my work in this session — I did not wire it in — but it directly
contradicts `ARCHITECTURE_decision.md`, which lists a **user-facing urgency flag as
surface #7, "not shipping"**, and which is the basis for the finding that Blocker 1
does not bind. If that surface ships, **Blocker 1 binds immediately**: 30.2% of benign
pigmented lesions flagged at an above-floor threshold, and it was measured on a
lighter-skinned corpus than our users.

**This is a decision, not a bug, and it is yours.** Three options:

1. **Remove the urgency band and condition list from the pilot build.** Consistent
   with the architecture record. The pilot is about capture and the symptom gate;
   neither needs the LLM output.
2. **Keep them and reopen Blocker 1** as binding, with the pigmentation false-alarm
   rate as a launch criterion.
3. **Keep them but relabel** — the LLM's urgency is not the CNN's urgency score, and
   is not what any of our measurements describe. Nothing in the record characterises
   its per-tone behaviour at all, which is arguably worse than a measured 30.2%.

**I have not changed it.** Flagging rather than deciding, because it changes what the
product is.

---

## What a tester sees, step by step

| # | screen | real or stubbed | what is recorded |
|---|---|---|---|
| 1 | Training-consent screen | **real** | consent flag + version + timestamp; declining still allows triage, nothing stored |
| 2 | **"Before we start"** — one screen, four checkboxes: breathing/swallowing difficulty, face or lip swelling, throat tightness, tongue swelling | **real** | all four answers |
| 2a | If any ticked → **"Call emergency services now."** Flow stops. No photo. | **real** | `writeShortCircuit()` writes a full row with no image, `crop_method = short-circuit-no-capture` |
| 3 | Camera with a draggable-size **reticle**, live advisory hints | **real** | every quality metric, every frame (hints only, never blocks) |
| 4 | Take photo | **real** | native-res crop + full-res original, `crop.method`, `localiserAgreed`, `localiserIou`, `downsampleFactor` |
| 5 | Remaining symptom screens, one per rule group (6) | **real** | all 27 fields as explicit true/false |
| 6 | Result screen | **⚠️ LLM path — see the warning above** | — |

Abandonment at any step posts to `/api/capture/dropoff`.

---

## What is real vs stubbed

**Real:** reticle capture at native resolution; two-channel localiser refinement;
all quality metrics; single-lesion blob count; the full symptom gate including the
emergency tier; short-circuit rows; drop-off logging; holdout CSV writes with EXIF
stripping.

**Stubbed / absent:** the trained model is not called from the UI at any point.
Burst-capture sharpness and reticle spot-metering are **not built yet** — sharpness
is single-frame and metering is whatever the camera chooses. The labelling console
does not yet read `data/holdout/` (`true_class` stays empty by design; only a
dermatologist may fill it).

**Shadow mode:** `captureQuality.MODE = "shadow"`. Nothing is ever rejected. This is
deliberate — thresholds must come from the pilot's own distribution, because a gate
cannot be calibrated on the data it prevents you collecting.

---

## What lands in a holdout row

53 columns in `data/holdout/labels.csv`. Notable ones:

- `image_path`, `original_path` — the 224 crop and the full-resolution frame
- `source_resolution`, `device_model`, `downsample_factor` (**lower is better**)
- `crop_method` (`reticle` / `reticle+refined` / `wide-declared` / `centre-fallback`),
  `crop_confidence`, `localiser_agreed`, `localiser_iou`
- `quality_sharpness`, `quality_brightness`, `quality_fill`, `quality_passed`
- all 27 symptom fields, `symptom_tier`, `symptom_rules_fired`
- `capture_language`, `questionnaire_version`
- `true_class` — **always empty at capture.** Dermatologist-assigned only.

Plus `rejections.csv` and `dropoff.csv` alongside.

---

## What a first tester will hit that we have not handled

1. **The opening screen is alarming.** Four airway questions before anything else, to
   someone photographing acne. Instrumented via drop-off, but not solved. This is the
   most likely cause of funnel loss and we will only learn its size from the pilot.
2. **English only.** `TRANSLATION_plan.md` is written; nothing is translated. A
   Telugu/Kannada/Tamil tester cannot use it.
3. **The reticle is size-adjustable but not draggable.** It is fixed at centre. A
   lesion the user cannot centre (back, scalp) has no good path.
4. **No "wide" mode in the UI.** `captureNative` supports `mode: "wide"` for
   whole-limb rashes, but nothing exposes it — every capture is `close`.
5. **iOS Safari resolution.** `highResConstraints` requests 4096×3072; Safari may
   silently return far less. `source_resolution` records what we actually got, so
   this is measurable, but it is untested on a real device.
6. **No review-before-submit.** The capture is committed immediately; there is no
   "does this look right?" step.
7. **Consent decline path with the camera open** is untested.
8. **Nothing is deleted.** Consent withdrawal is specified but not built.

---

## Conformance and checks

```
npm run test:conformance     # 16 hand-written cases, JS + Python, all 4 tiers
npm run build                # app builds
ml/.venv/bin/python ml/regenerate_report.py <ckpt> --check
```

The conformance fixture is hand-written, not generated, so a shared codegen error
fails it — verified with a negative control that reintroduced the bug which had
previously shipped the emergency tier broken.
