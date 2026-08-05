/* src/components/SymptomFlow.jsx
 * ---------------------------------------------------------------------------
 * The symptom questionnaire — the load-bearing safety mechanism.
 *
 * Per docs/ARCHITECTURE_decision.md §3b: per-tone MODEL performance is not
 * measurable at any sample size we can reach, so the safety case rests on this
 * rule-based, escalate-only gate rather than on the model. It never sees a model
 * output, and it can only raise urgency, never lower it.
 *
 * THE OPENING SCREEN
 * The airway questions are asked first because they are the only minutes-scale tier.
 * But four escalating airway questions put to someone photographing acne is an
 * alarming way to open a consumer app, and if it makes people leave we lose captures
 * at the front of the funnel. So they are presented as ONE neutral screen -
 * "any of these right now?" - rather than four separate escalating prompts.
 *
 * Abandonment is reported at every step (POST /api/capture/dropoff) so the pilot can
 * tell us whether the opening costs completions. That is a real risk of this design
 * and it is instrumented rather than assumed away.
 *
 * WHAT THIS COMPONENT NEVER SHOWS
 * No class name, no confidence, no urgency score, no model output of any kind. The
 * only escalation it can display comes from scoreSymptoms().
 * ---------------------------------------------------------------------------
 */
import { useState } from "react";
import { SYMPTOM_RULES, scoreSymptoms } from "../lib/symptomRules.js";

const EMERGENCY_RULE = SYMPTOM_RULES.find((r) => r.tier === "emergency");
const REST = SYMPTOM_RULES.filter((r) => r.tier !== "emergency");

function report(step) {
  fetch("/api/capture/dropoff", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ step, meta: { deviceModel: navigator.userAgent.slice(0, 80) } }),
  }).catch(() => {});
}

export default function SymptomFlow({ onComplete, onEmergency }) {
  const [screen, setScreen] = useState(0);   // 0 = urgent-now, then one per rule group
  const [answers, setAnswers] = useState({});

  const set = (field, value) => setAnswers((a) => ({ ...a, [field]: value }));

  // ---- screen 0: the single neutral opening ----
  if (screen === 0) {
    const anyChecked = EMERGENCY_RULE.fields.some((f) => answers[f] === true);
    return (
      <section className="sf">
        <h2>Before we start</h2>
        <p className="sf-lead">Are any of these happening <strong>right now</strong>?</p>
        <ul className="sf-list">
          {EMERGENCY_RULE.fields.map((f) => (
            <li key={f}>
              <label>
                <input type="checkbox" checked={answers[f] === true}
                       onChange={(e) => set(f, e.target.checked)} />
                <span>{EMERGENCY_RULE.questions[f]}</span>
              </label>
            </li>
          ))}
        </ul>
        <div className="sf-actions">
          <button className="primary" onClick={() => {
            const r = scoreSymptoms(answers);
            if (r.shortCircuit) { onEmergency(answers, r); return; }
            setScreen(1);
          }}>{anyChecked ? "Continue" : "None of these"}</button>
        </div>
        <p className="sf-note">
          We ask this first because these need immediate help, not a photo.
        </p>
      </section>
    );
  }

  // ---- one screen per remaining rule group ----
  const group = REST[screen - 1];
  if (!group) {
    return null;
  }
  const isLast = screen - 1 === REST.length - 1;

  return (
    <section className="sf">
      <h2>A few questions</h2>
      <p className="sf-progress">Step {screen} of {REST.length}</p>
      <ul className="sf-list">
        {group.fields.map((f) => (
          <li key={f}>
            <label>
              <input type="checkbox" checked={answers[f] === true}
                     onChange={(e) => set(f, e.target.checked)} />
              <span>{group.questions[f]}</span>
            </label>
          </li>
        ))}
      </ul>
      <div className="sf-actions">
        <button onClick={() => { report(`abandon:${group.id}`); setScreen(screen - 1); }}>
          Back
        </button>
        <button className="primary" onClick={() => {
          if (isLast) {
            // Answered questions become explicit false, so "not answered" and
            // "answered no" stay distinguishable in the holdout row.
            const full = {};
            for (const r of SYMPTOM_RULES) {
              for (const f of r.fields) full[f] = answers[f] === true;
            }
            onComplete(full, scoreSymptoms(full));
          } else {
            setScreen(screen + 1);
          }
        }}>{isLast ? "Done" : "Next"}</button>
      </div>
    </section>
  );
}
