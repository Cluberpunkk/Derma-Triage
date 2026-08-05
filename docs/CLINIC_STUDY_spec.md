# Dermatology partner study — what we need, and why

**For a dermatology department considering a research collaboration.**
Two pages. Prepared by the DermaTriage team.

---

## What we are building, stated plainly

A **triage** tool: a patient photographs a skin problem on their phone, and the
system helps prioritise who should be seen sooner. It is **not** a diagnostic device
and does not tell anyone they are fine — it can recommend seeing a doctor, never
recommend not seeing one.

Current model performance on a public atlas, frozen test split n=3,950:
**top-1 0.571, top-3 0.772, macro-F1 0.555** across 23 coarse categories
*(operating point: top-1 argmax; no urgency threshold applied)*.

Every figure below carries its operating point, because several of ours have had to
be withdrawn for lacking one.

| figure | value | operating point | n |
|---|---|---|---|
| AK/BCC recall | **0.573** | top-1 argmax; no urgency threshold, so the ignorance floor does not apply | 286 |
| AK/BCC top-3 recall | 0.818 | top-3 shortlist; no threshold | 286 |
| malignancy-folder cases assigned to a benign category | **143 (36%)** | **urgency threshold 0.00 — answer every case. This is BELOW our 0.3089 "ignorance floor"** (see note) | 400 |
| AK/BCC recall at an above-floor operating point | 0.762 | urgency threshold 0.32 | 286 |

**The ignorance floor.** Our urgency score sums probabilities over 8 of 23 classes, so
a model outputting a *uniform* distribution — knowing nothing — still scores 0.309.
Any threshold below that cannot distinguish urgency from ignorance. The 36% figure is
measured with no threshold at all, which is the most permissive possible setting; at
an above-floor operating point AK/BCC recall is 0.762 instead of 0.573, at the cost of
missing more cases overall.

**The 36% is a coarse-category miss rate, not a melanoma miss rate.** One of the two
folders ("Melanoma Skin Cancer Nevi and Moles") merges benign naevi with melanoma in
the public dataset. We raise this ourselves because it is exactly the imprecision
review would catch.

This is why the model is confined to ranking a review queue rather than advising
patients directly.

## The problem only a clinic can solve

We have hit a hard statistical wall. To show the system performs **equally well
across skin tones** — the question that matters most for an Indian patient
population — we need enough *urgent* cases within each tone group.

| what we need to measure | images needed in a consumer app (~5% urgent) | in a clinic (~25% urgent) |
|---|---|---|
| per-tone urgent recall | **17,804** | **~3,200** |
| symptom-questionnaire yield, cellulitis | 9,604 | ~1,900 |
| symptom-questionnaire yield, severe drug reactions | 48,018 | ~9,600 |

*(80% power, α=0.05, effect sizes measured on public data; assumes balanced 50/50
recruitment across lighter and darker tones.)*

**A third deliverable from the same partnership.** The study also settles our
escalation-rule **firing-rate tolerances**: deciding whether a rule sits at its bar
(rather than merely detecting gross overshoot) needs 389–1,468 captures per rule —
comfortably inside 3,200 encounters. So one collaboration yields *sensitivity*,
*dermatologist labels*, and *calibrated tolerances* for the safety mechanism.

**Clinic prevalence is the entire difference.** A consumer app waits for urgent cases
to walk in at ~5%; a dermatology clinic sees them at five times that rate. This is
the only lever that works — collecting more consumer images does not.

Until this study happens, **we have written into our design record that no safety
claim may depend on per-tone performance**, because we cannot measure it. That
constraint is what we would like your help to lift.

## Second thing only you can provide: real labels

Our system currently has **no clinical ground truth at all**. Every label comes from
a public atlas whose categories bundle several diseases together (one folder mixes
psoriasis with lichen planus). We treat dermatologist-assigned labels as the only
valid ground truth — our software will not accept a diagnosis from a patient, from
the app, or from anyone who is not a qualified reader.

We are especially interested in **steroid-modified tinea**, which is common in Indian
practice and absent from every public dataset we know of. No public data can teach a
model to recognise it.

---

## What a partner would provide

**Cases** — target ~3,200 imaged encounters over the study period.

| tier | target share | why |
|---|---|---|
| urgent (suspected malignancy, cellulitis, blistering, systemic) | ~25% | the statistical requirement above |
| routine (eczema, psoriasis, fungal, acne, pigmentary) | ~75% | reflects real presentation mix |

**Skin tone balance** — roughly equal numbers across lighter and darker tones.
Statistical power depends on the *smaller* group, so balance is worth more than
volume. We can supply a simple tone reference card for consistent recording.

**Reader time** — the main ask.

| task | estimate |
|---|---|
| assign a diagnosis + urgency tier per case | ~1–2 min/case |
| **overlapping subset**: ~15% read independently by a second reader | ~500 cases |
| **content review of our 6 escalation rules** (`RULE_REVIEW_form.md`) | ~30–45 min per reviewer |

The overlapping subset gives us **inter-rater agreement**, which we need for an
uncomfortable but necessary reason: if two dermatologists disagree on a third of
photographs, that disagreement is the ceiling on what any model can achieve, and we
would rather know it early than discover it after building around a number that was
never attainable.

**Photographs** — taken through our capture app (a browser page, no install). It
records the phone's full-resolution image plus a short symptom questionnaire.

---

## Data handling and consent

- **Explicit, separate consent for research use.** No consent → no storage. Consent
  is recorded with a timestamp and version.
- **Right to withdraw** at any time, with deletion of that participant's images and
  records.
- **EXIF and GPS stripped** on capture; optional on-device face blurring before
  anything is transmitted.
- Images stored on institutional or partner-approved infrastructure, never publicly
  accessible, served only through authenticated routes.
- Handled in a manner appropriate to India's DPDP Act. We will work to whatever your
  ethics committee requires, and expect to submit for IRB/EC review.
- **No patient-facing output during the study.** Participants receive their normal
  clinical care; the system's output is not shown to them and does not enter their
  record.

## What the partner gets

- Named co-authorship on any publication arising from the work.
- Full access to the resulting labelled dataset for the department's own research.
- An honest performance report, including the failures. We publish the numbers that
  look bad — the 0.573 AK/BCC recall above is our own measurement, and we corrected
  our own imprecise phrasing of it before sending this.
- No obligation to deploy anything.

## What we are not asking for

- Not asking anyone to use the tool clinically, or to act on its output.
- Not asking for retrospective records or existing photographs.
- Not asking for unpaid work — reader time is budgeted and compensated.

---

## Suggested first step

A **50-case feasibility run** over two weeks, plus a **30–45 minute content review**
of our six escalation rules (`docs/RULE_REVIEW_form.md`) by two or three readers
while they are already engaged.

That answers the practical questions — does the capture flow fit clinic workflow, how
long does labelling actually take, what does inter-rater agreement look like — before
either side commits to 3,200 cases. The rule review is separately valuable: it
establishes content validity for our safety mechanism, which is the standard first
step for a clinical decision rule and the only validation reachable before the full
study.

**Contact:** *[to be completed]*
**Prepared:** from `ml/power_analysis.py` and `docs/ARCHITECTURE_decision.md`.
All performance figures are measured on a frozen test split and reproducible.
