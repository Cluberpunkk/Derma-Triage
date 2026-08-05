/* src/lib/symptomRules.js — GENERATED. Do not edit by hand.
 *
 * Source of truth: ml/symptom_gate.py
 * Regenerate:      npm run gen:symptoms
 * Verify in CI:    ml/.venv/bin/python ml/gen_symptom_rules.py --check
 *
 * The in-app questionnaire and the server-side gate must never diverge, so the
 * rules are generated from the Python rather than transcribed. Escalate-only:
 * this can raise urgency and can never lower it, and it never touches the model.
 */

export const SYMPTOM_RULES = [
  {
    "id": "airway_emergency",
    "tier": "emergency",
    "min_hits": 1,
    "fields": [
      "breathing_difficulty",
      "face_swelling",
      "throat_tightness",
      "tongue_swelling"
    ],
    "questions": {
      "breathing_difficulty": "Are you having any difficulty breathing or swallowing?",
      "face_swelling": "Is your face or are your lips swelling?",
      "throat_tightness": "Does your throat feel tight or closing?",
      "tongue_swelling": "Is your tongue swelling?"
    },
    "rationale": "Airway angioedema / anaphylaxis. Minutes, not hours. Bypasses the photo entirely and every other rule. [VERIFY]",
    "covers": [
      "*"
    ]
  },
  {
    "id": "spreading_infection",
    "tier": "today",
    "min_hits": 2,
    "fields": [
      "spreading_hours",
      "warm_to_touch",
      "tender",
      "fever"
    ],
    "questions": {
      "spreading_hours": "Has the red area spread noticeably in the last 24 hours?",
      "warm_to_touch": "Does the area feel warm compared to the skin around it?",
      "tender": "Is it painful or tender to touch?",
      "fever": "Do you have a fever or feel unwell?"
    },
    "rationale": "Cellulitis: spreading margin + warmth + tenderness \u00b1 fever. Spread over hours cannot be seen in one photo. [VERIFY]",
    "covers": [
      "Cellulitis Impetigo and other Bacterial Infections"
    ]
  },
  {
    "id": "systemic_involvement",
    "tier": "today",
    "min_hits": 1,
    "fields": [
      "mucosal_involvement",
      "skin_peeling",
      "blisters_large_area"
    ],
    "questions": {
      "mucosal_involvement": "Are your mouth, eyes or genitals also affected?",
      "skin_peeling": "Is skin peeling off in sheets?",
      "blisters_large_area": "Are there blisters over a large area?"
    },
    "rationale": "Mucosal involvement or sheet-like peeling suggests SJS/TEN; airway or facial swelling suggests angioedema. Both are emergencies and neither is reliably visible in a lesion photo. [VERIFY]",
    "covers": [
      "Exanthems and Drug Eruptions",
      "Bullous Disease Photos"
    ]
  },
  {
    "id": "systemic_illness",
    "tier": "soon",
    "min_hits": 2,
    "fields": [
      "fever",
      "joint_pain",
      "fatigue",
      "weight_loss",
      "photosensitivity"
    ],
    "questions": {
      "joint_pain": "Any joint pain or swelling?",
      "fatigue": "Unusual tiredness?",
      "weight_loss": "Unintended weight loss?",
      "photosensitivity": "Does sunlight make the rash worse?"
    },
    "rationale": "Connective-tissue disease and vasculitis present with systemic features the skin photo cannot show. [VERIFY]",
    "covers": [
      "Lupus and other Connective Tissue diseases",
      "Vasculitis Photos",
      "Systemic Disease"
    ]
  },
  {
    "id": "non_healing_or_changing",
    "tier": "soon",
    "min_hits": 1,
    "fields": [
      "non_healing_4w",
      "bleeding_no_trauma",
      "changing_lesion"
    ],
    "questions": {
      "non_healing_4w": "Has a sore failed to heal for more than 4 weeks?",
      "bleeding_no_trauma": "Does it bleed or crust without being knocked?",
      "changing_lesion": "Has a mole or spot changed in size, shape or colour?"
    },
    "rationale": "Evolution is the strongest single malignancy signal and is by definition invisible in one photo. [VERIFY]",
    "covers": [
      "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
      "Melanoma Skin Cancer Nevi and Moles"
    ]
  },
  {
    "id": "pigmented_lesion",
    "tier": "soon",
    "min_hits": 2,
    "fields": [
      "lesion_changing_months",
      "lesion_bleeds",
      "lesion_non_healing",
      "lesion_ugly_duckling",
      "lesion_new_after_40",
      "lesion_itches"
    ],
    "questions": {
      "lesion_changing_months": "Has this spot changed in size, shape or colour over the last few months?",
      "lesion_bleeds": "Does it bleed, ooze or crust without being knocked?",
      "lesion_non_healing": "Has it failed to heal for more than 4 weeks?",
      "lesion_ugly_duckling": "Does it look clearly different from your other moles and spots?",
      "lesion_new_after_40": "Is this a new spot that appeared after age 40?",
      "lesion_itches": "Does it itch or feel different from the skin around it?"
    },
    "rationale": "Seborrheic keratosis vs pigmented BCC is the model's largest bidirectional confusion (SK->AK/BCC 30x, AK/BCC->SK 28x) and drives Blocker 1. Clinicians separate them WITHOUT dermoscopy mainly on history: SK is stable for years and stuck-on; BCC changes, bleeds and fails to heal. None of that is in a photograph, so this questionnaire attacks the blocker where the image cannot. 'Ugly duckling' \u2014 the lesion unlike the person's others \u2014 is a stronger malignancy signal than any single lesion feature. [VERIFY]",
    "covers": [
      "Seborrheic Keratoses and other Benign Tumors",
      "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
      "Melanoma Skin Cancer Nevi and Moles"
    ]
  },
  {
    "id": "vulnerable_host",
    "tier": "soon",
    "min_hits": 1,
    "fields": [
      "immunosuppressed",
      "neonate",
      "pregnant_with_rash"
    ],
    "questions": {
      "immunosuppressed": "Are you on immunosuppressants, chemotherapy or steroids long-term?",
      "neonate": "Is this a baby under 3 months?",
      "pregnant_with_rash": "Are you pregnant?"
    },
    "rationale": "Host factors change urgency regardless of appearance. [VERIFY]",
    "covers": [
      "*"
    ]
  }
];

export const ALL_SYMPTOM_FIELDS = ["bleeding_no_trauma", "blisters_large_area", "breathing_difficulty", "changing_lesion", "face_swelling", "fatigue", "fever", "immunosuppressed", "joint_pain", "lesion_bleeds", "lesion_changing_months", "lesion_itches", "lesion_new_after_40", "lesion_non_healing", "lesion_ugly_duckling", "mucosal_involvement", "neonate", "non_healing_4w", "photosensitivity", "pregnant_with_rash", "skin_peeling", "spreading_hours", "tender", "throat_tightness", "tongue_swelling", "warm_to_touch", "weight_loss"];

/** Derived from ml/symptom_gate.py TIER_ORDER — never hardcode a tier here. */
const TIER_ORDER = {"none": 0, "soon": 1, "today": 2, "emergency": 3};

/** Derived from ml/symptom_gate.py — never hardcode a message here. */
const TIER_MESSAGE = {"emergency": "Call emergency services now.", "today": "Please seek care today.", "soon": "Please see a doctor soon.", "none": ""};

/** Tiers that short-circuit the whole flow: no photo, no model, no shortlist. */
export const SHORT_CIRCUIT_TIERS = ["emergency"];

/** Apply the gate to answers. Mirrors symptom_gate.score_row exactly. */
export function scoreSymptoms(answers = {}) {
  const fired = [];
  let tier = "none";
  for (const r of SYMPTOM_RULES) {
    const hits = r.fields.filter((f) => answers[f] === true);
    if (hits.length >= r.min_hits) {
      fired.push({ rule: r.id, tier: r.tier, hits });
      if (TIER_ORDER[r.tier] > TIER_ORDER[tier]) tier = r.tier;
    }
  }
  return {
    tier,
    rulesFired: fired,
    escalates: tier !== "none",
    shortCircuit: SHORT_CIRCUIT_TIERS.includes(tier),
    message: TIER_MESSAGE[tier] ?? "",
  };
}

/** Questions in ask-order, deduplicated across rules. */
export function questionList() {
  const seen = new Set();
  const out = [];
  for (const r of SYMPTOM_RULES) {
    for (const f of r.fields) {
      if (seen.has(f)) continue;
      seen.add(f);
      out.push({ field: f, text: r.questions[f] || f, group: r.id, tier: r.tier });
    }
  }
  return out;
}
