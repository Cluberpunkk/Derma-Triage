# Clinical content review — symptom escalation rules

**For a reviewing dermatologist. Estimated 30–45 minutes.**
Six rules. For each: what triggers it, where it comes from, and four questions.

---

## Why content review, and what it can and cannot establish

Validating these rules by **outcome** — do they actually catch cellulitis? — needs
9,604 images for cellulitis and 48,018 for severe drug reactions, at realistic
prevalence. That study is not reachable soon (see `CLINIC_STUDY_spec.md`).

**Content validation** — do experts agree the rules are clinically sensible and
complete? — needs two or three dermatologists and an afternoon. It is the standard
first step for a clinical decision rule, and it is achievable today.

It establishes **face and content validity**. It does **not** establish sensitivity,
specificity, or that the rule performs in practice. We will not claim otherwise.

**What these rules do:** raise urgency, never lower it. They run *independently* of
the image model and can only add an escalation, never suppress one. So a rule that
fires too readily costs an unnecessary referral; a rule that fires too rarely simply
adds nothing. Please weigh the two accordingly.

---

## How to score

For each rule, four items:

| item | scale |
|---|---|
| **A. Clinically sensible?** | 1 = not at all · 5 = entirely |
| **B. Would it fire on a typical presenting case?** | 1 = almost never · 5 = almost always |
| **C. Threshold correct?** | too low / about right / too high |
| **D. Anything critical missing?** | free text |

Rule wording is written for patients, not clinicians. Please flag any question a
patient would misread — that is as important as the clinical content.

---

## Rule 1 — `spreading_infection` → **seek care today**

**Fires when ≥2 of:** spread noticeably in 24h · warm to touch · painful/tender · fever
**Provenance:** `validated_instrument` — spreading erythema, warmth, tenderness and
oedema (the classic cellulitis tetrad); systemic features grade severity (cf. Eron
classification).
**Our open question:** we ask these because a photograph cannot show spread over time.

A ☐1 ☐2 ☐3 ☐4 ☐5  B ☐1 ☐2 ☐3 ☐4 ☐5  C ☐too low ☐about right ☐too high
D: ______________________________________________

## Rule 2 — `systemic_involvement` → **seek care today**

**Fires when ≥1 of:** mouth/eyes/genitals affected · skin peeling in sheets · blisters
over a large area · difficulty breathing/swallowing · face, lip or tongue swelling
**Provenance:** `validated_instrument` — mucosal involvement plus detachment is the
core SJS/TEN pattern (cf. SCORTEN); Nikolsky sign; standard angioedema red flags.
**Our open question:** is *"peeling in sheets"* an adequate lay proxy for Nikolsky?

A ☐1 ☐2 ☐3 ☐4 ☐5  B ☐1 ☐2 ☐3 ☐4 ☐5  C ☐too low ☐about right ☐too high
D: ______________________________________________

## Rule 3 — `systemic_illness` → **see a doctor soon**

**Fires when ≥2 of:** fever · joint pain/swelling · unusual tiredness · unintended
weight loss · rash worse in sunlight
**Provenance:** `guideline_derived` — ACR/EULAR SLE classification domains; vasculitis
workup guidance. Used only to escalate, never to classify.
**Our open question:** is 2-of-5 an appropriate escalation threshold?

A ☐1 ☐2 ☐3 ☐4 ☐5  B ☐1 ☐2 ☐3 ☐4 ☐5  C ☐too low ☐about right ☐too high
D: ______________________________________________

## Rule 4 — `non_healing_or_changing` → **see a doctor soon**

**Fires when ≥1 of:** sore not healed >4 weeks · bleeds/crusts without trauma · mole
or spot changed in size, shape or colour
**Provenance:** `validated_instrument` — "E" of ABCDE; standard suspected-cancer
referral triggers; 4-week non-healing window from referral guidance.
**Our open question:** is 4 weeks the right window for Indian primary care?

A ☐1 ☐2 ☐3 ☐4 ☐5  B ☐1 ☐2 ☐3 ☐4 ☐5  C ☐too low ☐about right ☐too high
D: ______________________________________________

## Rule 5 — `pigmented_lesion` → **see a doctor soon**

**Fires when ≥2 of:** changed over recent months · bleeds/oozes/crusts · not healed
>4 weeks · looks different from the person's other moles · new after age 40 · itches
**Provenance:** `validated_instrument` — ugly-duckling sign; the features separating
BCC from seborrhoeic keratosis *without* dermoscopy (SK is stable for years, stuck-on).
**Why it exists:** SK↔BCC is our model's largest bidirectional confusion (30× and 28×
on the test set). We believe history separates them where the photograph cannot.
**Our open question:** does 2-of-6 over-trigger in older patients where SK is
near-universal?

A ☐1 ☐2 ☐3 ☐4 ☐5  B ☐1 ☐2 ☐3 ☐4 ☐5  C ☐too low ☐about right ☐too high
D: ______________________________________________

## Rule 6 — `vulnerable_host` → **see a doctor soon**

**Fires when ≥1 of:** immunosuppressed · baby under 3 months · pregnant
**Provenance:** `guideline_derived` — standard host-factor modifiers; rash in a
neonate is a near-universal urgent-referral trigger.
**Our open question:** should pregnancy escalate on its own, or only with another
feature?

A ☐1 ☐2 ☐3 ☐4 ☐5  B ☐1 ☐2 ☐3 ☐4 ☐5  C ☐too low ☐about right ☐too high
D: ______________________________________________

---

## Overall

**1. What is missing entirely?** Which red flag would you check in clinic that no
rule above covers? *(This is the most valuable question on the form.)*

______________________________________________________________

**2. Any rule that should be removed** as more likely to mislead than help?

______________________________________________________________

**3. Framing.** We are considering presenting these to patients as
*"what your doctor will want to know"* — visit preparation — rather than as a triage
verdict. Same questions, same escalation, different claim. Does that read as more
appropriate to you, or does it under-communicate urgency when a rule fires?

______________________________________________________________

**4. Would you be comfortable** with these questions being asked of patients by an
app that then says "please see a doctor soon"? ☐ yes ☐ yes with changes ☐ no

______________________________________________________________

---

**Reviewer:** ____________________  **Date:** __________
**Result recorded to:** `ml/urgency_groups.yaml` `reviewed_by`, and the per-rule
`[VERIFY]` markers in `ml/symptom_gate.py`, which stay in place until a named
reviewer signs each rule off.
