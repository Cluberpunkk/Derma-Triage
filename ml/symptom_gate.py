"""ml/symptom_gate.py
=========================================================================
A rule-based red-flag path that BYPASSES the image model entirely.

WHY THIS EXISTS
---------------
Cellulitis has the worst threshold recall (0.918) and the worst queue capture
(0.260 at top-10%) of any urgent class — and it progresses in hours. But the deeper
point is that cellulitis is not an image problem at all. It is diagnosed clinically:
a spreading margin, warmth, tenderness out of proportion, fever, systemic upset.
Several of those are invisible in a photograph, and one of them (spread over hours)
is invisible in any SINGLE photograph.

So no amount of retraining fixes it. The fix is to ask.

The same argument covers the other classes whose defining features are systemic
rather than visual: Systemic Disease, Vasculitis, Lupus and other Connective Tissue
diseases, and severe drug eruptions.

PROVENANCE - WHY THESE ARE NOT INVENTED
--------------------------------------
Every rule carries `provenance` and `source`. Most derive from published guidance or
validated instruments: the ABCDE rule, the ugly-duckling sign, SJS/TEN mucosal
criteria and Nikolsky, ACR/EULAR SLE domains, standard suspected-cancer referral
triggers. That matters because docs/CONSTANTS_audit.md found most of this system's
constants were ASSUMED, and a rule-based safety gate built on invented rules would be
no better than the constants it replaced.

  validated_instrument  the trigger is a published, named clinical sign or instrument
  guideline_derived     derived from classification or referral guidance, adapted to
                        lay wording
  invented              no source. REVIEWED FIRST. (currently: none)

WHAT IS STILL OURS, AND THEREFORE STILL ASSUMED:
  * the THRESHOLDS (2-of-4, 2-of-6). Every rule carries a [VERIFY] question naming
    exactly what a reviewer should challenge about its threshold.
  * the WORDING. Turning "Nikolsky sign" into "is skin peeling off in sheets?" is a
    lay translation that may lose or distort the sign. Called out per rule.

DESIGN
------
This gate runs BEFORE and INDEPENDENT of the model, and can only ever escalate —
it never downgrades or suppresses an image-driven escalation. Rules are declarative
so a dermatologist can edit them; every rule carries a [VERIFY] rationale.

HONEST LIMIT ON THE YIELD ESTIMATE
----------------------------------
The obvious question is: how many of our 6 cellulitis misses at t=0.12 would these
questions have caught? **That is not measurable on DermNet.** DermNet is images and
folder names — there are no symptom annotations, so there is no way to evaluate the
questionnaire against those 6 cases. Anyone reporting a number here would be
inventing it.

What we can do instead, and what this file supports:
  * `estimate` prints a literature-grounded EXPECTATION, clearly labelled as an
    assumption rather than a measurement, with the sensitivity each rule would need.
  * `score_row` runs the gate on real data, so the yield becomes measurable the
    moment the holdout set carries symptom columns — which is why
    data/holdout/labels.csv includes them.

    ml/.venv/bin/python ml/symptom_gate.py --explain
    ml/.venv/bin/python ml/symptom_gate.py --eval data/holdout/labels.csv
=========================================================================
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Each rule: any listed field being truthy contributes. `min_hits` of the group
# triggers the tier. All [VERIFY] — needs dermatologist sign-off before shipping.
# ---------------------------------------------------------------------------
RULES = [
    {
        # FIRST RULE ON PURPOSE: this is the only tier where minutes matter, so it is
        # asked first and short-circuits everything else.
        "id": "airway_emergency",
        "provenance": 'validated_instrument',
        "source": "Airway compromise in angioedema/anaphylaxis is a minutes-scale emergency; swelling of lips, tongue or throat with difficulty breathing or swallowing is the standard airway red flag, and 'call emergency services' is the standard lay instruction. SPLIT OUT of systemic_involvement deliberately: SJS/TEN is hours-to-days and angioedema is minutes, so a single 'seek care today' tier served neither, and lumping them also lumped two very different prevalences into one tolerance anchor. NOT INVENTED. [VERIFY: is 1-of-4 correct, or should throat tightness alone suffice?]",
        "tier": "emergency",
        "min_hits": 1,
        "fields": ["breathing_difficulty", "face_swelling", "throat_tightness",
                   "tongue_swelling"],
        "question_text": {
            "breathing_difficulty": "Are you having any difficulty breathing or swallowing?",
            "face_swelling": "Is your face or are your lips swelling?",
            "throat_tightness": "Does your throat feel tight or closing?",
            "tongue_swelling": "Is your tongue swelling?",
        },
        "rationale": "Airway angioedema / anaphylaxis. Minutes, not hours. Bypasses the "
                     "photo entirely and every other rule. [VERIFY]",
        "covers": ["*"],
    },
    {
        "id": "spreading_infection",
        "provenance": 'validated_instrument',
        "source": 'Cellulitis is recognised by spreading erythema, warmth, tenderness and oedema - the classic tetrad, standard in dermatology and infectious-disease texts. Systemic features grade severity (cf. the Eron classification of skin and soft-tissue infection). NOT INVENTED: these are the features a clinician elicits at the bedside. What is novel is only that we ASK them, because a photograph cannot show spread over time. [VERIFY]',
        "tier": "today",
        "min_hits": 2,
        "fields": ["spreading_hours", "warm_to_touch", "tender", "fever"],
        "question_text": {
            "spreading_hours": "Has the red area spread noticeably in the last 24 hours?",
            "warm_to_touch": "Does the area feel warm compared to the skin around it?",
            "tender": "Is it painful or tender to touch?",
            "fever": "Do you have a fever or feel unwell?",
        },
        "rationale": "Cellulitis: spreading margin + warmth + tenderness ± fever. "
                     "Spread over hours cannot be seen in one photo. [VERIFY]",
        "covers": ["Cellulitis Impetigo and other Bacterial Infections"],
    },
    {
        "id": "systemic_involvement",
        "provenance": 'validated_instrument',
        "source": "Mucosal involvement plus skin detachment is the core SJS/TEN recognition pattern; the SCORTEN severity score and consensus definitions both key on mucosal sites and body-surface detachment. Nikolsky sign is the classic bedside test, approximated here as 'peeling in sheets'. Airway features moved to the airway_emergency rule. NOT INVENTED. [VERIFY: is 'peeling in sheets' an adequate lay proxy for Nikolsky?]",
        "tier": "today",
        "min_hits": 1,
        "fields": ["mucosal_involvement", "skin_peeling", "blisters_large_area"],
        "question_text": {
            "mucosal_involvement": "Are your mouth, eyes or genitals also affected?",
            "skin_peeling": "Is skin peeling off in sheets?",
            "blisters_large_area": "Are there blisters over a large area?",
        },
        "rationale": "Mucosal involvement or sheet-like peeling suggests SJS/TEN; "
                     "airway or facial swelling suggests angioedema. Both are "
                     "emergencies and neither is reliably visible in a lesion "
                     "photo. [VERIFY]",
        "covers": ["Exanthems and Drug Eruptions", "Bullous Disease Photos"],
    },
    {
        "id": "systemic_illness",
        "provenance": 'guideline_derived',
        "source": 'Photosensitivity, arthralgia, fatigue and weight loss appear among the ACR/EULAR SLE classification domains and in vasculitis workup guidance. Used here ONLY to escalate, never to classify - classification criteria are explicitly not diagnostic criteria, and we are not attempting a diagnosis. [VERIFY: is 2-of-5 an appropriate escalation threshold?]',
        "tier": "soon",
        "min_hits": 2,
        "fields": ["fever", "joint_pain", "fatigue", "weight_loss", "photosensitivity"],
        "question_text": {
            "joint_pain": "Any joint pain or swelling?",
            "fatigue": "Unusual tiredness?",
            "weight_loss": "Unintended weight loss?",
            "photosensitivity": "Does sunlight make the rash worse?",
        },
        "rationale": "Connective-tissue disease and vasculitis present with systemic "
                     "features the skin photo cannot show. [VERIFY]",
        "covers": ["Lupus and other Connective Tissue diseases", "Vasculitis Photos",
                   "Systemic Disease"],
    },
    {
        "id": "non_healing_or_changing",
        "provenance": 'validated_instrument',
        "source": "Evolution is the 'E' of the ABCDE melanoma rule. Non-healing sore and bleeding-without-trauma are standard skin-cancer referral triggers in primary-care guidance, and the 4-week non-healing window is a common guideline threshold. NOT INVENTED. [VERIFY: is 4 weeks right for Indian primary care?]",
        "tier": "soon",
        "min_hits": 1,
        "fields": ["non_healing_4w", "bleeding_no_trauma", "changing_lesion"],
        "question_text": {
            "non_healing_4w": "Has a sore failed to heal for more than 4 weeks?",
            "bleeding_no_trauma": "Does it bleed or crust without being knocked?",
            "changing_lesion": "Has a mole or spot changed in size, shape or colour?",
        },
        "rationale": "Evolution is the strongest single malignancy signal and is by "
                     "definition invisible in one photo. [VERIFY]",
        "covers": ["Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
                   "Melanoma Skin Cancer Nevi and Moles"],
    },
    {
        "id": "pigmented_lesion",
        "provenance": 'validated_instrument',
        "source": "The 'ugly duckling' sign - a lesion unlike the patient's others - is a published melanoma-screening heuristic, complementary to ABCDE. Change over months, bleeding and failure to heal are the standard features separating BCC from seborrhoeic keratosis WITHOUT dermoscopy; SK is characteristically stable for years and stuck-on. NOT INVENTED. [VERIFY: does 2-of-6 over-trigger in older patients where SK is near-universal?]",
        "tier": "soon",
        "min_hits": 2,
        "fields": ["lesion_changing_months", "lesion_bleeds", "lesion_non_healing",
                   "lesion_ugly_duckling", "lesion_new_after_40", "lesion_itches"],
        "question_text": {
            "lesion_changing_months": "Has this spot changed in size, shape or colour "
                                      "over the last few months?",
            "lesion_bleeds": "Does it bleed, ooze or crust without being knocked?",
            "lesion_non_healing": "Has it failed to heal for more than 4 weeks?",
            "lesion_ugly_duckling": "Does it look clearly different from your other "
                                    "moles and spots?",
            "lesion_new_after_40": "Is this a new spot that appeared after age 40?",
            "lesion_itches": "Does it itch or feel different from the skin around it?",
        },
        "rationale": "Seborrheic keratosis vs pigmented BCC is the model's largest "
                     "bidirectional confusion (SK->AK/BCC 30x, AK/BCC->SK 28x) and drives "
                     "Blocker 1. Clinicians separate them WITHOUT dermoscopy mainly on "
                     "history: SK is stable for years and stuck-on; BCC changes, bleeds "
                     "and fails to heal. None of that is in a photograph, so this "
                     "questionnaire attacks the blocker where the image cannot. "
                     "'Ugly duckling' — the lesion unlike the person's others — is a "
                     "stronger malignancy signal than any single lesion feature. [VERIFY]",
        "covers": ["Seborrheic Keratoses and other Benign Tumors",
                   "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
                   "Melanoma Skin Cancer Nevi and Moles"],
    },
    {
        "id": "vulnerable_host",
        "provenance": 'guideline_derived',
        "source": 'Immunosuppression, neonatal age and pregnancy are standard host-factor modifiers in infection and drug-eruption severity assessment; rash in a neonate is a near-universal urgent-referral trigger in paediatric guidance. NOT INVENTED. [VERIFY: should pregnancy escalate alone, or only with another feature?]',
        "tier": "soon",
        "min_hits": 1,
        "fields": ["immunosuppressed", "neonate", "pregnant_with_rash"],
        "question_text": {
            "immunosuppressed": "Are you on immunosuppressants, chemotherapy or steroids "
                                "long-term?",
            "neonate": "Is this a baby under 3 months?",
            "pregnant_with_rash": "Are you pregnant?",
        },
        "rationale": "Host factors change urgency regardless of appearance. [VERIFY]",
        "covers": ["*"],
    },
]

TIER_ORDER = {"none": 0, "soon": 1, "today": 2, "emergency": 3}
ALL_FIELDS = sorted({f for r in RULES for f in r["fields"]})


def truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def score_row(row: dict) -> dict:
    """Apply the gate to one case's symptom answers. Escalation only, never downgrade."""
    fired, tier = [], "none"
    for r in RULES:
        hits = [f for f in r["fields"] if truthy(row.get(f))]
        if len(hits) >= r["min_hits"]:
            fired.append({"rule": r["id"], "tier": r["tier"], "hits": hits})
            if TIER_ORDER[r["tier"]] > TIER_ORDER[tier]:
                tier = r["tier"]
    return {"tier": tier, "rules_fired": fired,
            "bypasses_model": tier != "none",
            # emergency short-circuits the whole flow: no photo, no model, no shortlist
            "short_circuit": tier == "emergency",
            "message": {
                "emergency": "Call emergency services now.",
                "today": "Please seek care today.",
                "soon": "Please see a doctor soon.",
                "none": ""}[tier]}


def explain() -> None:
    print("SYMPTOM RED-FLAG GATE — runs before and independent of the image model.\n"
          "It can only ESCALATE. It never downgrades an image-driven escalation.\n")
    for r in RULES:
        print(f"[{r['tier'].upper():<5}] {r['id']}   (fires when >= {r['min_hits']} of "
              f"{len(r['fields'])} are yes)")
        for f in r["fields"]:
            q = r["question_text"].get(f, f)
            print(f"    - {q}")
        print(f"    rationale: {r['rationale']}")
        print(f"    covers: {', '.join(c[:44] for c in r['covers'])}\n")

    print("=" * 74)
    print("YIELD ON OUR 6 CELLULITIS MISSES: NOT MEASURABLE.")
    print("=" * 74)
    print("DermNet has images and folder names only — no symptom annotations. There is")
    print("no way to evaluate these questions against those 6 cases, and any number")
    print("quoted here would be invented.\n")
    print("What can be said, as an ASSUMPTION rather than a measurement:")
    print("  * Warmth and tenderness are near-universal in cellulitis; a 2-of-4 rule over")
    print("    {spreading, warm, tender, fever} would plausibly fire on most true cases.")
    print("  * Its value does not depend on the model at all — it fires on cases the")
    print("    model ranks LOW, which is precisely where the queue fails.")
    print("  * The cost is false escalation from any warm, tender rash, which is the")
    print("    correct direction of error for a same-day tier.")
    print("\nTo MEASURE it: data/holdout/labels.csv carries these symptom columns. Once")
    print("filled, `--eval` reports true yield. That is the only honest route to a number.")
    print(f"\nSymptom fields the holdout must record ({len(ALL_FIELDS)}):")
    print("  " + ", ".join(ALL_FIELDS))


def evaluate(csv_path: Path) -> int:
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    present = [f for f in ALL_FIELDS if any(f in r for r in rows)]
    if not present:
        print(f"{csv_path} has none of the symptom columns — nothing to evaluate.")
        print("  expected any of: " + ", ".join(ALL_FIELDS))
        return 1
    print(f"{len(rows)} rows; symptom columns present: {len(present)}/{len(ALL_FIELDS)}")

    out, by_tier, by_class = [], {"none": 0, "soon": 0, "today": 0, "emergency": 0}, {}
    for r in rows:
        g = score_row(r)
        by_tier[g["tier"]] += 1
        tc = (r.get("true_class") or "").strip()
        if tc:
            by_class.setdefault(tc, {"n": 0, "gated": 0})
            by_class[tc]["n"] += 1
            by_class[tc]["gated"] += int(g["bypasses_model"])
        out.append({"image_path": r.get("image_path"), "true_class": tc, **g})

    print(f"\ntier: emergency {by_tier['emergency']}  today {by_tier['today']}  "
          f"soon {by_tier['soon']}  none {by_tier['none']}")
    print(f"escalated by symptoms alone: "
          f"{(by_tier['emergency'] + by_tier['today'] + by_tier['soon']) / max(len(rows), 1):.1%} of cases")
    if by_class:
        print("\nper true class — caught by the symptom gate (no model involved):")
        for c, v in sorted(by_class.items(), key=lambda kv: -kv[1]["gated"] / max(kv[1]["n"], 1)):
            print(f"  {v['gated']:>3}/{v['n']:<3} = {v['gated'] / max(v['n'], 1):>5.1%}  {c[:52]}")
    p = csv_path.parent / "symptom_gate_eval.json"
    p.write_text(json.dumps({"n": len(rows), "columns_present": present,
                             "by_tier": by_tier, "by_class": by_class,
                             "per_row": out}, indent=2))
    print(f"\nWrote {p}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--eval", help="labels.csv carrying symptom columns")
    a = ap.parse_args()
    if a.eval:
        return evaluate(Path(a.eval))
    explain()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
