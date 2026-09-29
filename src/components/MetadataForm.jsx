/* src/components/MetadataForm.jsx
 * ---------------------------------------------------------------------------
 * The five clinical metadata fields, asked AFTER capture.
 *
 * ORDER: the photo is the scarce artifact, so it is banked before we spend the
 * user's patience. Only the airway questions run before the camera, because airway
 * angioedema is minutes-scale and must not require taking a photo first.
 *
 * WHAT WAS REMOVED
 * The legacy form's seven symptom chips (Itching / Pain / Burning / Bleeding /
 * Spreading / Scaling / Color change) are gone. They are morphology descriptors that
 * overlapped `tender` and `lesion_itches` in the symptom gate, fed only the LLM path
 * (now flagged off), and cost completions at the front of the funnel for nothing.
 *
 * WHAT WAS KEPT, AND WHY
 * These five map onto ml/metadata.py's fusion inputs. prior_steroid_use is the field
 * that defines steroid-modified tinea — absent from every public dataset and the one
 * thing a public corpus cannot teach.
 *
 * SKIN TONE IS A SWATCH STRIP, NOT FITZPATRICK
 * Fitzpatrick was built around burn/tan response in light skin, and South Asian users
 * classify themselves on it inconsistently. The Monk Skin Tone scale was designed for
 * self-identification across diverse populations, and people answer "which of these is
 * closest?" far more reliably than a text scale.
 *
 * It is stored as self_reported_tone, NEVER as fitzpatrick — that column is reserved
 * for a trained reader, and self-report must not be able to flip fairness_measurable.
 * The payoff is three tone signals on one image (self-report, ITA-from-image, reader),
 * and comparing them is the first real validation of the ITA proxy that OPEN-1 rests on.
 * ---------------------------------------------------------------------------
 */
import { useState } from "react";

// Monk Skin Tone scale, 10 swatches.
const MONK = [
  "#f6ede4", "#f3e7db", "#f7ead0", "#eadaba", "#d7bd96",
  "#a07e56", "#825c43", "#604134", "#3a312a", "#292420",
];

// Body sites mirror console/shared/vocab.js BODY_SITES.
const SITES = ["Face", "Scalp", "Neck", "Chest", "Back", "Abdomen",
               "Groin", "Buttocks", "Arm", "Hand", "Leg", "Foot", "Nails", "Other"];
const DURATION = [["days", "A few days"], ["weeks", "A few weeks"],
                  ["months", "Months"], ["years", "Years"]];
const CHANGE = [["spreading", "Spreading"], ["shrinking", "Getting smaller"],
                ["same", "About the same"], ["new", "Brand new"]];

export default function MetadataForm({ onComplete, onSkip }) {
  const [v, setV] = useState({ bodySite: "", duration: "", priorCream: "",
                               priorSteroidUse: "", recentChange: "",
                               selfReportedTone: "" });
  const set = (k) => (val) => setV((s) => ({ ...s, [k]: val }));

  return (
    <section className="sf">
      <h2>A few details</h2>
      <p className="sf-note">All optional — they help a doctor read the photo.</p>

      <Chips label="Where on the body?" options={SITES.map((s) => [s, s])}
             value={v.bodySite} onChange={set("bodySite")} />

      <Chips label="How long has it been there?" options={DURATION}
             value={v.duration} onChange={set("duration")} />

      <Chips label="Has it changed recently?" options={CHANGE}
             value={v.recentChange} onChange={set("recentChange")} />

      <Chips label="Have you put any cream or ointment on it?"
             options={[["yes", "Yes"], ["no", "No"]]}
             value={v.priorCream} onChange={set("priorCream")} />

      {/* The steroid question only appears if a cream was used. It is the single
          highest-value field we collect: steroid-modified tinea is invisible in
          every public dataset. */}
      {v.priorCream === "yes" && (
        <Chips label="Was it a steroid or combination cream?"
               options={[["yes", "Yes"], ["no", "No"], ["unknown", "Not sure"]]}
               value={v.priorSteroidUse} onChange={set("priorSteroidUse")} />
      )}

      <div className="mf-block">
        <div className="mf-label">Which of these is closest to your skin?</div>
        <div className="swatches" role="radiogroup" aria-label="Skin tone">
          {MONK.map((hex, i) => (
            <button key={hex} type="button" title={`Tone ${i + 1}`}
                    aria-label={`Tone ${i + 1} of 10`}
                    aria-pressed={v.selfReportedTone === String(i + 1)}
                    className={"swatch" + (v.selfReportedTone === String(i + 1) ? " on" : "")}
                    style={{ background: hex }}
                    onClick={() => set("selfReportedTone")(String(i + 1))} />
          ))}
        </div>
        <div className="sf-note">
          Pick the nearest — it does not need to be exact. This helps us check the app
          works equally well on every skin tone.
        </div>
      </div>

      <div className="sf-actions">
        <button onClick={onSkip}>Skip</button>
        <button className="primary" onClick={() => onComplete(v)}>Done</button>
      </div>
    </section>
  );
}

function Chips({ label, options, value, onChange }) {
  return (
    <div className="mf-block">
      <div className="mf-label">{label}</div>
      <div className="chips">
        {options.map(([val, text]) => (
          <button key={val} type="button" aria-pressed={value === val}
                  className={"chip" + (value === val ? " on" : "")}
                  onClick={() => onChange(value === val ? "" : val)}>{text}</button>
        ))}
      </div>
    </div>
  );
}
