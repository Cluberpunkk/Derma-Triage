/* src/lib/surfaces.js
 * ---------------------------------------------------------------------------
 * Which user-facing surfaces are enabled.
 *
 * docs/ARCHITECTURE_decision.md enumerates every surface and which blockers bind
 * under each. That document ASSUMED surface #7 (a user-facing urgency verdict) was
 * not shipping. It was found live in src/App.jsx. The assumption had been used as a
 * premise for the finding that Blocker 1 does not bind.
 *
 * So surfaces are now DECLARED here and VERIFIED in code (ml/check_surfaces.py),
 * rather than asserted in prose.
 *
 * SURFACE_7_URGENCY_VERDICT must stay false for the pilot. Beyond the architecture
 * argument: a tester shown named conditions reacts to the diagnosis rather than to
 * the capture flow, and it may contaminate their symptom answers. The pilot measures
 * capture, questionnaire and firing rates - a verdict on screen degrades all three.
 *
 * Turning it on reopens Blocker 1 as BINDING (30.2% of benign pigmented lesions
 * flagged at an above-floor threshold, measured on a lighter-skinned corpus than our
 * users), and the LLM path it renders has no calibration and no per-tone
 * characterisation at all.
 * ---------------------------------------------------------------------------
 */

/** Surface #7 — urgency band + named conditions from the LLM triage path. */
export const SURFACE_7_URGENCY_VERDICT = false;

/** Surface #6 — top-3 shortlist from the trained model. Never wired. */
export const SURFACE_6_SHORTLIST = false;
