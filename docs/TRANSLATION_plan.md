# Translating the escalation questions — a safety task, not a localisation task

## Why this is not ordinary localisation

The 25 symptom questions in `ml/symptom_gate.py` are the **load-bearing safety
mechanism** of this system. `ARCHITECTURE_decision.md` §3b makes that explicit: per-tone
model performance is unmeasurable at any reachable sample size, so the rule-based,
escalate-only gate carries the safety case — not the model.

A mistranslation is therefore a **safety-relevant defect**, in the same class as a
wrong threshold. Our users are vernacular Tier-2/3 speakers of Telugu, Kannada and
Tamil; if *"is skin peeling off in sheets?"* renders as something closer to "is your
skin dry and flaking?", the SJS/TEN rule stops firing on the cases it exists for, and
nothing downstream notices.

`ml/symptom_gate.py` already flags the English wording as **ours and assumed** — the
lay rendering of Nikolsky sign is our translation from clinical language, and it may
already lose the sign before any other language is involved. This document is that
same risk, one layer further out.

## Scope

| | |
|---|---|
| Languages | Telugu, Kannada, Tamil (English source) |
| Items | all 25 symptom fields + 6 tier messages + consent text |
| Source of truth | `ml/symptom_gate.py` → generated `src/lib/symptomRules.js` |

Hindi is *not* on this list on purpose — add it only if the target districts need it,
rather than by default.

## Process

**1. Forward translation — two independent professional translators per language.**
Medical or health-communication experience required. Given the clinical intent of each
item, not just the English string: a translator who knows the question is screening
for a spreading infection will render it differently, and better.

**2. Reconciliation.** The two forward translations are reconciled into one by a third
bilingual reviewer, with disagreements logged rather than silently resolved.

**3. Independent back-translation.** A translator who has **not seen the English
source** renders the reconciled version back into English.

**4. Per-item comparison against the source.** Each back-translated item is scored:

| verdict | meaning | action |
|---|---|---|
| equivalent | same clinical concept | accept |
| shifted | recognisable but the clinical emphasis moved | revise |
| lost | the clinical sign is not recoverable | **reject and redo** |

Any item marked *lost* on a `tier: today` rule blocks that language from the pilot.

**5. Two separate arms, because they answer different questions.**

**5a. Paired bilingual arm — does the translation change the ANSWER?**
Bilingual participants answer **both** the English and the translated questionnaire
**about the same lesion in the same session**, with order counterbalanced.

Population is held constant, so any per-item disagreement is attributable to the
**translation**, not to the population. This is the same paired logic that cut the
required n elsewhere: an English-first design would compare a translated arm against
a baseline drawn from English-literate users — a different, non-representative
subpopulation — and confound translation with population.

Analysis is per item on **discordant pairs** (answered yes in one language, no in the
other), tested with McNemar. Roughly **n ≈ 140 bilingual participants** estimates an
item-level discordance rate to about ±5%; far fewer than two independent arms would
need. Order counterbalancing controls for the participant having already thought
about the question once.

**5b. Monolingual comprehension arm — is the wording UNDERSTOOD?**
5–8 monolingual speakers per language, drawn from the target population, not staff.
For each item: *what is this question asking you?* This tests whether a patient with
no medical background recovers the clinical intent — a different question from
whether the translation matches English, and one the bilingual arm cannot answer,
because a bilingual participant may recover the intent from the English they also
read.

Items where fewer than 6 of 8 recover the intent get rewritten.

**6. Sign-off** by a clinician who speaks the language, on the final version.

## Items we expect to be hardest

Flagging these up front so translators get extra attention where it matters:

| item | difficulty |
|---|---|
| `skin_peeling` — *"peeling off in sheets"* | Nikolsky proxy. Must not collapse into ordinary dryness or flaking. Highest risk item on the form. |
| `spreading_hours` — *"spread in the last 24 hours"* | requires a specific time comparison, not a general sense of worsening |
| `lesion_ugly_duckling` — *"different from your other moles"* | comparative across the patient's own body; easily flattened to "looks unusual" |
| `mucosal_involvement` — *"mouth, eyes or genitals"* | needs to be askable without embarrassment, or it will be answered "no" |
| `tender` vs `lesion_itches` | many languages blur pain/discomfort/itch distinctions that the rules depend on |

## Detecting failure after launch

`capture_language` and `questionnaire_version` are recorded on every capture
(`server/lib/holdout.js`). This makes a per-language firing-rate comparison possible.

**A rule firing at materially different rates across languages is a translation defect
until proven otherwise.** It is not an epidemiological finding. Disease prevalence does
not vary that sharply between neighbouring language groups, and the translation is by
far the likelier explanation.

Power for this is favourable, for the same reason as the specificity analysis: firing
rate is measured on *everyone*, not on rare true cases. At n≈300 per language a firing
rate is estimable to within a few points — enough to see a rule that has stopped firing
in one language.

## Cost and sequencing

Two translators + reconciler + back-translator + 8 testers × 3 languages, plus
clinician sign-off. Small relative to the clinic study, and it **gates the pilot** in
those languages: an English-only pilot is fine, but a Telugu pilot on unvalidated
translations would produce firing rates we could not interpret.

**Recommendation: run the English pilot first** — to validate the flow, not to
produce a translation baseline. The baseline for translation comes from the **paired
bilingual arm** (§5a), where the same people answer both versions, not from comparing
an English cohort against a Telugu cohort. Those are different populations, and the
comparison would confound translation with who was in each arm.

Post-launch per-language firing-rate monitoring (below) is a *surveillance* signal
for drift, not the primary validation — the paired arm is.

## Status

- [ ] forward translation (×2 per language)
- [ ] reconciliation
- [ ] independent back-translation
- [ ] per-item comparison scored
- [ ] paired bilingual arm (~140 participants, order counterbalanced, McNemar per item)
- [ ] monolingual comprehension testing (5–8 speakers per language)
- [ ] clinician sign-off
- [ ] `capture_language` verified populated end to end

**Nothing here is done.** The gate currently exists in English only.
