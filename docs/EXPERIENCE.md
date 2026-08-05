# EXPERIENCE.md — how an experienced dermatologist reasons, and where that plugs in

> ## Read this before using this file
>
> **This document cannot improve the image model.** A CNN maps pixels to 23 category
> scores; it has no text input. Nothing written here reaches it. Anyone claiming a
> knowledge document "trains" a vision model is mistaken about the mechanism.
>
> **What it can legitimately drive** — three places, all real:
> 1. `runTriage()` — the LLM layer, which *does* consume text and phrases the output
> 2. **The question flow** — which questions the app asks, in what order
> 3. **Deterministic rules** — escalation triggers and priors, coded and auditable
>
> **I am not a dermatologist.** This is assembled from general clinical literature
> and is a *starting draft for a dermatologist to correct*, not clinical authority.
> Every rule below is tagged `[VERIFY]` until a qualified dermatologist signs it off.
> Shipping unverified clinical rules would be worse than shipping none.
>
> **No treatment content appears here, by design.** Not a limitation to fix.

---

## 1. What experience actually is

The gap between a new graduate and a dermatologist of twenty years is **not** more
facts — both read the same textbooks. It is:

- **Pattern priors.** Knowing what is *common here*, in this population, this season.
  A ring-shaped scaly plaque in an Indian outpatient clinic is tinea until proven
  otherwise. The same lesion in Oslo has a different prior.
- **Knowing which question decides it.** Not asking twenty questions — asking the
  two that separate the live hypotheses.
- **Knowing when the photo is not enough** and reaching for KOH, dermoscopy, biopsy.
- **Calibrated doubt.** Experienced clinicians say "I don't know, let's test"
  *more* often than juniors, not less.

Only the first three are transferable to software. The fourth is what our
abstention gate imitates.

---

## 2. The consultation sequence

An experienced consultation runs roughly in this order. Our app currently collapses
most of it into one photo.

| # | Step | What it establishes | In our system |
|---|---|---|---|
| 1 | **Look before touching** | Distribution, symmetry, extent | image model |
| 2 | **Morphology** | Papule / plaque / vesicle / pustule / nodule; scale, crust | image model (weakly) |
| 3 | **Distribution** | *Where* often outweighs *what it looks like* | `body_site` — **captured, currently unused by the model** |
| 4 | **Timeline** | Days vs. years reshapes the differential entirely | `duration` |
| 5 | **Symptoms** | Itch vs. pain vs. nothing | `symptoms` |
| 6 | **Prior treatment** | Especially steroids — see §5 | `prior_steroid_use` |
| 7 | **Palpation** | Indurated? tender? rough? | **impossible from a photo** |
| 8 | **Dermoscopy / KOH / biopsy** | Confirmation | `confirmation_method` in the console |

**Steps 7–8 are the information a photo can never contain.** This is the honest
ceiling on any image-only product, and no amount of training data moves it.

---

## 3. The history that changes everything

Our model's worst confusions are visually near-identical but historically distinct.
From the actual test confusion matrix:

```
Eczema      → Psoriasis    34x
Psoriasis   → Tinea        32x
Tinea       → Psoriasis    27x
Eczema      → Tinea        24x
```

More images will not fix these. The separating information is not in the pixels.

### Tinea vs. eczema vs. psoriasis `[VERIFY]`

| | tinea | eczema | psoriasis |
|---|---|---|---|
| **edge** | active, raised, scaly; **central clearing** | ill-defined, blurs into normal skin | sharply demarcated |
| **scale** | fine, at the advancing edge | fine, variable | thick, silvery, adherent |
| **site** | groin, feet, body folds, trunk | flexures; hands; face in children | extensors — elbows, knees, scalp, sacrum |
| **itch** | usually itchy | **intensely** itchy | often minimal |
| **course** | spreads outward over weeks | relapsing–remitting, often lifelong | chronic, stable plaques |
| **steroid response** | **worsens / spreads** | improves | improves |
| **nails** | onychomycosis may coexist | usually spared | pitting, oil drops, onycholysis |
| **decisive test** | **KOH** | clinical | clinical ± biopsy |

**The single highest-yield question in this whole document:** *"Did you apply a cream,
and did it get better or worse?"* Worse-on-steroid points hard at tinea. It costs one
question and separates the three conditions that account for our largest error mass.

---

## 4. Red flags — escalate regardless of model output

These are **codeable rules**, independent of any model confidence. All `[VERIFY]`.

**Lesion-level**
- Asymmetry, irregular border, multiple colours, diameter >6 mm, **evolving** — the
  ABCDE rule. *Evolution matters most*; a changing lesion outranks a static odd one.
- A new or changing lesion in an adult over ~40
- Non-healing ulcer or sore persisting beyond ~4 weeks
- Bleeding, crusting or ulceration without trauma
- A single pigmented lesion clearly unlike the patient's others ("ugly duckling")
- Any pigmented lesion on palm, sole, or under a nail — **acral melanoma is
  proportionally more common in darker skin and is routinely missed late**

**Systemic / urgent**
- Fever with rash; rapidly spreading redness; severe pain out of proportion
- Mucosal involvement (mouth, eyes, genitals) — possible severe drug reaction
- Skin peeling in sheets, or blistering over large areas
- Rash in an immunosuppressed patient
- Any rash in a neonate

**Design rule already implemented in `ml/pipeline.py`:** escalation is evaluated
**before** abstention, so a low-confidence possible malignancy still surfaces. A
missed cancer routed to "see a clinician" is acceptable; one never shown is not.

---

## 5. Steroid-modified tinea — the India-specific case

The single most valuable thing this product could get right, and absent from every
public dataset.

**What happens.** Over-the-counter combination creams (potent steroid + antifungal
+ antibacterial) are sold widely without prescription. Applied to tinea, the steroid
suppresses inflammation — the patient sees improvement — while the fungus spreads.

**What it then looks like `[VERIFY]`**
- Loss of the classic raised scaly edge; margins become indistinct
- Wider, more extensive, often bizarre outlines
- Reduced scale; skin atrophy, telangiectasia, striae from the steroid
- Frequently multiple sites, extending to face and groin
- **Resembles eczema — which is exactly the wrong treatment**

**Why our model is blind to it.** DermNet has no such category. Our vocabulary has
`tinea_steroid_modified`, and no public dataset can populate it. It requires the
dermatologist console: KOH-confirmed cases with `prior_steroid_use = yes`.

**Codeable prior `[VERIFY]`:** `prior_steroid_use = yes` **and** `recent_change =
spreading` should raise the tinea prior substantially and *suppress* the eczema
prior — the opposite of what appearance alone suggests.

---

## 6. Skin tone changes presentation — not just performance

Widely under-appreciated, and directly relevant to a product built for Indian users.
All `[VERIFY]`.

- **Erythema is hard to see in Fitzpatrick V–VI.** Redness may read as violaceous,
  grey or simply darker. Inflammatory conditions are systematically under-recognised
  because "redness" is the taught cue.
- **Post-inflammatory hyperpigmentation (PIH)** is far more prominent and persistent,
  and is often the patient's main complaint even after the primary condition resolves.
- **Keloids and hypertrophic scars** are more common.
- **Acral melanoma** — palms, soles, nail beds — is proportionally more common and
  routinely diagnosed late.
- **Scale and lichenification** may be more visually apparent than erythema, so
  morphology cues shift.

**Consequence for us:** a model trained on light-skin-dominant atlases learns
"red = inflamed" and will under-detect inflammation in exactly our target users.
This is not fixable by tuning. It needs Fitzpatrick-labelled training data, and it
is why the skin-tone caveat stays in the app until measured on a held-out dark-skin
test set (DDI).

---

## 7. What this system must never do

Non-negotiable, and already enforced structurally in code:

- **No treatment, medicine, cream or product.** `ml/pipeline.py`'s output schema has
  no field for one. In India specifically, self-applied combination creams are a
  leading cause of the worsened, hard-to-treat infections above — a photo cannot
  safely distinguish the cases where a steroid helps from those where it harms.
- **No diagnosis.** Top-3 shortlist with confidence, or an honest refusal.
- **Never reassure.** "Probably nothing" is the one output that can kill. Absence of
  a red flag is not evidence of safety.
- **Never suppress urgency or the red-flag path** based on model confidence.

---

## 8. Dermatologist review checklist

For the reviewing dermatologist — this document is a draft until these are done:

- [ ] §3 differentiation table — correct, or over-simplified to the point of harm?
- [ ] §4 red flags — anything missing? Any threshold wrong for Indian practice?
- [ ] §5 steroid-modified tinea — does the described appearance match your OPD reality?
- [ ] §6 skin-tone claims — accurate for Fitzpatrick IV–VI?
- [ ] Which of these can be trusted as **automated escalation rules** vs. only as
      guidance to an LLM's phrasing?
- [ ] What is missing that you rely on daily and would never find in a textbook?

The last question is the point of this file. Everything above is book knowledge;
the answer to that one is the actual experience, and it is the thing no competitor
can copy.

---

## 9. Where each section is consumed

| section | consumer | status |
|---|---|---|
| §2 sequence | question flow in `src/App.jsx` | partly implemented |
| §3 history | `ml/metadata.py` fields + fusion model | **built, awaiting real data** |
| §4 red flags | `ml/pipeline.py` escalation | partly coded (can't-miss classes) |
| §5 steroid tinea | `prior_steroid_use` prior; console labelling | field captured, prior **not yet coded** |
| §6 skin tone | Fitzpatrick/Monk capture; fairness eval | captured; eval **blocked on DDI** |
| §7 prohibitions | output schema | **enforced structurally** |

Anything marked "not yet coded" is a real gap, listed here rather than implied to
be done.
