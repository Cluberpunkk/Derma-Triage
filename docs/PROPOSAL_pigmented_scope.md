# Proposal: decline to assess pigmented lesions

**Status: proposal, not a decision.** Written to be argued with. The counter-case is
in §4 and is not weak.

## 1. The case for

> **SUPERSEDED (ignorance floor).** This section originally opened with "71.4% of
> benign pigmented lesions are flagged as possibly urgent". That figure was measured
> at t=0.12, **below the 0.3089 ignorance floor**, where a uniform prediction also
> fires the gate. At the first above-floor threshold (t=0.32) the rate is **0.302**.
> See `data/ml/runs_final/CANONICAL.json` and `docs/ARCHITECTURE_decision.md`.
>
> **Two further corrections to this proposal's premise:** (a) the blocker is
> conditional on shipping a user-facing urgency flag, which surface #7 of the
> architecture decision says we are not; (b) under ranking, benign pigmented lesions
> reach the top-10% queue only 3.7% of the time, so they do not crowd it. Both weaken
> the case for a scope boundary further than §4 already did.

Blocker 1 is that **30.2% of benign pigmented lesions are flagged as possibly
urgent** at the first above-floor operating point (71.4% at the superseded
below-floor threshold). Three findings say this is not a
tuning problem:

- **It is not a spurious shortcut.** `ml/shortcut_test.py` refuted the pigmentation
  proxy hypothesis: darkness correlates with the urgent score at only ρ=+0.144
  within routine disease (r²≈0.02), the pigmented/non-pigmented capture gap is
  +0.026 across classes with inconsistent sign, and Grad-CAM saliency on the false
  alarms correlates with darkness at **−0.017**. The model is not "seeing dark".
- **It is genuine clinical mimicry.** The confusion is bidirectional and large —
  `SK → AK/BCC` 30×, `AK/BCC → SK` 28×. Seborrheic keratosis versus pigmented basal
  cell carcinoma is a real diagnostic problem that clinicians resolve with
  **dermoscopy** — a 10× magnified, cross-polarised view of subsurface structures
  that a phone camera does not produce.
- **The missing information is not in the input.** Milia-like cysts, comedo-like
  openings, arborising vessels: these are dermoscopic features. If the discriminative
  signal is absent from the image, no threshold, head, backbone or amount of training
  data recovers it.

**Threshold tuning implies a confidence we do not have.** Choosing an operating point
asserts we know the trade-off; a scope boundary asserts only that we know where we
are unreliable — which is the stronger and more defensible claim.

It is also simpler to explain, to a user and to a regulator: *"this app does not
assess moles and pigmented spots — see a doctor for those"* needs no calibration
argument and cannot silently degrade under distribution shift.

## 2. What "declining" would mean

1. A pigmented-lesion detector (the auxiliary head in `ml/heads.py`) routes any
   confidently-pigmented lesion **out of image scoring entirely**.
2. Those users get the `pigmented_lesion` symptom questionnaire
   (`ml/symptom_gate.py`) — change over months, bleeding, non-healing, ugly-duckling,
   new after 40 — which is how clinicians triage these *without* dermoscopy.
3. Output is a routing decision plus red-flag education. No category shortlist, no
   confidence score, no reassurance.

Note that this **keeps the escalation path and removes only the false reassurance**.

## 3. What it costs

- Melanoma and pigmented BCC no longer get an image-based flag. Today they get one:
  melanoma-folder capture at top-10% is 0.526, the best of any urgent class.
  **We would be discarding our single best-performing urgent signal.**
- A detector that mistakes an amelanotic melanoma for "not pigmented" routes it back
  into scoring — so the boundary is only as good as the detector, and its failure
  mode is the worst one available.

## 4. The counter-argument

**The strongest objection: a 71% false-alarm rate on benign pigmentation is not
obviously worse than the alternative, because the alternative is not "correct
assessment" — it is "no assessment at all".**

If a user with an early melanoma is told "we don't assess moles, see a doctor", and
does not go, the outcome is worse than a false alarm that frightens them into going.
Over-referral of benign pigmented lesions is the *safe* direction of error; the
current blocker is a trust and cost problem, not a safety one.

Three supporting points:

- Melanoma capture (0.526 at top-10%) is our best urgent signal. Discarding a
  working signal to fix a nuisance-alarm problem may be the wrong trade.
- The 71.4% figure is measured on **DermNet**, whose photography and population are
  not ours. It may be materially different on real user photos — that is exactly
  what `ml/eval_holdout.py` is for. Deciding scope before that measurement is
  deciding on data we know is unrepresentative.
- The symptom questionnaire (§2.2) may fix the blocker *without* a scope boundary:
  if `pigmented_lesion` rules can separate stable SK from changing BCC on history,
  the model's visual uncertainty stops being the binding constraint. That would give
  us both the melanoma signal and a controlled false-alarm rate.

## 5. Recommendation

**Do not adopt the scope boundary yet.** Sequence instead:

1. Measure the pigmented false-alarm rate on **real user photos** (`eval_holdout.py`).
   The DermNet number is unrepresentative in an unknown direction.
2. Measure the yield of the `pigmented_lesion` symptom rules on that same set.
3. Adopt the boundary only if the false-alarm rate stays high **and** the
   questionnaire fails to separate the cases.

Adopting it now would trade away a measured strength to fix a problem measured on
the wrong distribution.

## 6. Decision record

| | |
|---|---|
| Proposed | after the mimicry finding (`shortcut_test.py`) |
| Status | **open — not adopted; case further weakened by the ignorance floor and the ranking architecture** |
| Blocking evidence needed | holdout false-alarm rate; symptom-rule yield |
| Owner | — |
| Dermatologist review | not yet obtained |
