# Every magic number in the capture and quality path

Five constants have now caused a wrong conclusion or a broken gate. This is the whole
list, once, rather than discovering them one at a time.

**Provenance labels**

| label | meaning |
|---|---|
| **measured** | derived from data, with the measurement named |
| **inherited** | copied from an earlier file or a convention; never validated here |
| **assumed** | I chose it. No evidence behind it. |

---

## Capture path — `src/lib/lesionCrop.js`

| constant | value | provenance | basis | if wrong |
|---|---|---|---|---|
| `ANALYSIS_W` | 256 | **assumed** | "enough to find a box, cheap to compute" | Too small: lesion missed on wide frames. Too large: slower live loop. Low risk — only affects localisation, and the reticle is primary. |
| channel weight | 0.50 / 0.50 | **measured (coarsely)** | edge channel cut fallback 3–4× vs colour alone (`ita_audit.py --remedy edge`). The **split itself was never swept** — 0.5 was the first value tried and it worked. | A better ratio likely exists. Low risk while the reticle is primary; would matter if the localiser ever became load-bearing. |
| distance threshold | `0.55 × max` | **inherited** | carried from the original colour-only implementation | Directly sets which pixels count as lesion, so it drives `spread`, `confidence` and the fallback rate — i.e. the tone gradient. **Never swept.** A different value could change OPEN-1 entirely. |
| `PAD` | 1.6 | **assumed** | "some context around the lesion" | Too tight: clinically relevant surrounding skin is cropped away. Too loose: detail is thrown away in the downsample. Untested either way. |
| `MIN_CROP_FRAC` / `MAX_CROP_FRAC` | 0.25 / 0.95 | **assumed** | sanity rails | Rarely binding; low risk. |
| `minConfidence` | 0.25 | **assumed** | the fallback trigger | **This is the constant the entire tone-gradient finding rests on.** Fallback rate = P(confidence < 0.25). A different cut moves every number in `ita_audit.py`, including whether the +0.079 gap exists. Never swept. |
| reticle IoU | 0.35 | **assumed** | "broadly agrees with where the user aimed" | Too high: localiser refinement never applies, reticle box always used (safe). Too low: a bad localiser box overrides good user framing (unsafe). Asymmetric — bias high. |

## Quality gate — `src/lib/cameraQuality.js`

| constant | value | provenance | basis | if wrong |
|---|---|---|---|---|
| `sharpnessAbs` (VoL) | ~~55~~ → **164** | **measured** | `degradation_calibration.py`: accuracy first drops 5% at Gaussian σ=1.0, median VoL 164. The old 55 was ~3× too permissive — past ~20% accuracy loss, which is why it never fired. | Known unfair: at 164, 61.8% of the darkest ITA bin falls below vs 38.2% of the lightest at *identical* blur (spread 0.236). **Not enforced.** |
| `sharpnessRelative` | 0.09 | **measured, rejected as a fix** | VoL/contrast² equalises light-vs-dark (1.01× vs 1.30×) but pushes mid-tones 1.8× above both extremes; overall spread worsens to 1.89×. corr(contrast, VoL) = −0.018. | Logged, not used for hints. Choice deferred to the pilot distribution. |
| `brightnessLow/High` | ~~42/222~~ → **31 / 195** | **measured** | accuracy drops 5% below luma ~31 and above ~195. Over-exposure costs far more than under-exposure (acc 0.164 at gain 2.2 vs 0.427 at gain 0.25). | Old band was too tight dark, too loose bright. |
| ~~`fill`~~ | ~~0.40~~ **removed** | **assumed → measured broken** | rejected 94–99.6% of images. Centre disc is ~20% of the frame, so uniform edge energy scores ≈0.20 and 0.40 demanded a tightly centred lesion. | **Deleted, not retuned** — the reticle makes framing declared, so the heuristic solves a problem that no longer exists. |

## Analysis constants — `ml/`

| constant | value | provenance | basis | if wrong |
|---|---|---|---|---|
| ITA annulus `r > 0.46` | 0.46 | **assumed** | "outside a typical lesion" | Lesion contamination — tested and refuted three ways (`ita_confound.py`). |
| peripheral annulus `r > 0.80` | 0.80 | **assumed** | non-circular replacement check | Gradient survives here too (+0.106, CI excludes zero), so not sensitive at this value. |
| ITA bin edges | 55/41/28/10/−30 | **inherited** | Chardon / Del Bino, standard in the literature | Real external provenance — the only constant here with any. |
| `CLEAN_MAX_OVERLAP` | 0.05 | **assumed** | intended as a clean-sample filter | **Was circular** — fallback sets the box to the whole frame, so fallback cases can never pass. Flagged `CIRCULAR_DO_NOT_USE` in the output. |
| ignorance floor | 0.3089 | **measured** | calibrated uniform prediction over 8/23 urgent classes | Derived, not chosen. Solid. |
| urgency thresholds | 0.12 / 0.32 | **assumed → superseded** | 0.12 came from an arbitrary 95%-recall target | Both **below or barely above** the ignorance floor. Superseded; see `SESSION_SUMMARY.md`. |
| `BLOCKER_FLAG_RATE` | 0.30 | **assumed** | I picked it as "high enough to block" | Blocker 1 measures 0.302 — it fires on a value I chose arbitrarily, by 0.002. **The blocker status is an artifact of this constant.** |
| entropy gate `E` | 0.70 | **assumed** | picked from a sweep by eye | Sets what fraction goes to "uncertain". Never optimised against a cost model. |
| abstention `p ≥ 0.70/0.90` | — | **assumed → invalid** | set on *uncalibrated* confidences (T=1.707) | Superseded. |

---

## The three worth acting on

1. **`minConfidence = 0.25`** — the tone-gradient finding (OPEN-1, OPEN-2) is
   P(confidence < 0.25). It has never been swept. Before treating +0.079 as a
   property of the system, confirm it is not a property of this constant.
2. **`BLOCKER_FLAG_RATE = 0.30`** — Blocker 1 fires at 0.302 against a bar I invented.
   That is not a finding; it is a coin-flip on an arbitrary cut. Should be stated as
   a measured rate, with the bar named as a judgement.
3. **distance threshold `0.55`** — inherited, drives the fallback rate directly, never
   swept.

## Pattern

Of ~20 constants, **five are now measured** (blur, exposure band, ignorance floor, and two rejected-as-unfair sharpness forms), four are inherited, and the rest I assumed.
Every defect so far came from an assumed or inherited constant that nobody checked:
`fill=0.40` (broken), `sharpness=55` (inert), `CLEAN_MAX_OVERLAP` (circular),
`t=0.12` (below the ignorance floor), `effectiveScale` (inverted semantics).

New constants should ship with a provenance label and, where the number drives a
conclusion, a sensitivity sweep.
