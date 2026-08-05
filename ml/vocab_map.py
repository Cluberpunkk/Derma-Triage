"""ml/vocab_map.py
=========================================================================
DermNet's 23 folder names -> the triage controlled vocabulary in
console/shared/vocab.js.

READ THIS BEFORE TRUSTING THE MAPPING
-------------------------------------
DermNet folders are COARSE ATLAS BUCKETS, not diagnoses. "Psoriasis pictures
Lichen Planus and related diseases" is several distinct diseases in one folder.
So a model trained on these folders is a coarse CATEGORY model, never a
diagnosis model — which is exactly why the app stays triage + education.

Every mapping carries a `map_kind` so nothing silently pretends to be precise:

  direct  the folder is essentially the vocab entity
  coarse  the folder CONTAINS the vocab entity plus other diseases
          (label noise is expected; usable for pretraining, not for claims)
  none    no defensible vocab equivalent -> "other"

Where a folder mixes benign AND malignant lesions, or spans several vocab keys
at once, we deliberately map to "other" rather than guess: a wrong map poisons
the highest-stakes classes. `original_class` is preserved on every row, so the
mapping is advisory and can be revised without re-scanning the dataset.

DEFAULT TRAINING TARGET IS `original_class` (23-way, what the data supports).
The vocab mapping is opt-in via LABEL_COLUMN=confirmed_diagnosis.
=========================================================================
"""
from __future__ import annotations

# Vocab keys, kept in sync with console/shared/vocab.js DIAGNOSES.
VOCAB_KEYS = {
    "tinea", "tinea_steroid_modified", "eczema", "acne", "melasma", "pih",
    "pityriasis_versicolor", "scabies", "psoriasis", "urticaria", "vitiligo",
    "impetigo", "warts_molluscum", "suspected_malignancy", "other", "uncertain",
}

# folder name -> (vocab_key, map_kind, note)
MAPPING: dict[str, tuple[str, str, str]] = {
    "Acne and Rosacea Photos":
        ("acne", "coarse", "folder also contains rosacea"),
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions":
        ("suspected_malignancy", "coarse", "AK/BCC + other malignant lesions; refer bucket"),
    "Atopic Dermatitis Photos":
        ("eczema", "direct", ""),
    "Bullous Disease Photos":
        ("other", "none", "no bullous-disease key in vocab"),
    "Cellulitis Impetigo and other Bacterial Infections":
        ("impetigo", "coarse", "folder also contains cellulitis + other bacterial infections"),
    "Eczema Photos":
        ("eczema", "direct", ""),
    "Exanthems and Drug Eruptions":
        ("other", "none", "no drug-eruption key in vocab"),
    "Hair Loss Photos Alopecia and other Hair Diseases":
        ("other", "none", "scalp/hair disease; no vocab key"),
    "Herpes HPV and other STDs Photos":
        ("other", "none", "mixes herpes/STDs with HPV warts; mapping to warts_molluscum "
                          "would merge clinically divergent triage paths"),
    "Light Diseases and Disorders of Pigmentation":
        ("other", "none", "superset spanning vitiligo + melasma + pih + pityriasis "
                          "versicolor; not separable from the folder label alone"),
    "Lupus and other Connective Tissue diseases":
        ("other", "none", "systemic/CTD; no vocab key"),
    "Melanoma Skin Cancer Nevi and Moles":
        ("other", "none", "mixes BENIGN nevi with melanoma — mapping to "
                          "suspected_malignancy would label benign moles malignant"),
    "Nail Fungus and other Nail Disease":
        ("other", "none", "onychomycosis mixed with non-fungal nail disease; kept out of "
                          "`tinea` to protect the highest-value class"),
    "Poison Ivy Photos and other Contact Dermatitis":
        ("other", "none", "contact dermatitis is not atopic eczema; no vocab key"),
    "Psoriasis pictures Lichen Planus and related diseases":
        ("psoriasis", "coarse", "folder also contains lichen planus + related"),
    "Scabies Lyme Disease and other Infestations and Bites":
        ("scabies", "coarse", "folder also contains Lyme + other bites/infestations"),
    "Seborrheic Keratoses and other Benign Tumors":
        ("other", "none", "benign tumours; no vocab key"),
    "Systemic Disease":
        ("other", "none", "cutaneous signs of systemic disease; no vocab key"),
    "Tinea Ringworm Candidiasis and other Fungal Infections":
        ("tinea", "coarse", "folder also contains candidiasis + other mycoses"),
    "Urticaria Hives":
        ("urticaria", "direct", ""),
    "Vascular Tumors":
        ("other", "none", "no vocab key"),
    "Vasculitis Photos":
        ("other", "none", "no vocab key"),
    "Warts Molluscum and other Viral Infections":
        ("warts_molluscum", "coarse", "folder also contains other viral infections"),
}

# Body site is only asserted when the folder name states it outright; otherwise unknown.
# Values must exist in vocab.js BODY_SITES.
BODY_SITE_HINT = {
    "Nail Fungus and other Nail Disease": "Nails",
    "Hair Loss Photos Alopecia and other Hair Diseases": "Scalp",
}

# Vocab keys DermNet can never teach — they need the dermatologist-labelled corpus
# from the console. Note these are exactly the pigmentation / steroid-modified
# presentations that matter most in Indian outpatient practice.
UNCOVERED_KEYS = sorted(
    VOCAB_KEYS
    - {"other", "uncertain"}
    - {v[0] for v in MAPPING.values()}
)


def map_class(folder_name: str) -> tuple[str, str, str]:
    """(vocab_key, map_kind, note) for a DermNet folder name.

    Unknown folder names are NOT a crash — a mirror with extra/renamed folders still
    works — but they are reported as map_kind "unknown_folder" so the run surfaces them.
    """
    if folder_name in MAPPING:
        return MAPPING[folder_name]
    return ("other", "unknown_folder", "folder name not in the DermNet mapping table")


def body_site(folder_name: str) -> str:
    return BODY_SITE_HINT.get(folder_name, "")


if __name__ == "__main__":
    kinds: dict[str, int] = {}
    for folder, (key, kind, note) in sorted(MAPPING.items()):
        kinds[kind] = kinds.get(kind, 0) + 1
        print(f"{kind:<7} {folder}\n        -> {key}" + (f"   ({note})" if note else ""))
    print(f"\n{len(MAPPING)} folders: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    print("vocab keys DermNet cannot teach: " + ", ".join(UNCOVERED_KEYS))
