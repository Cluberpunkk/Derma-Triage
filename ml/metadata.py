"""ml/metadata.py
=========================================================================
Encodes the structured clinical fields the app ALREADY collects into a fixed
vector the model can consume alongside the image.

WHY THIS EXISTS
---------------
Our worst test confusions are visually near-identical but clinically separable:

    Eczema    -> Psoriasis   34x
    Psoriasis -> Tinea       32x
    Tinea     -> Psoriasis   27x

No amount of extra images fixes those, because the discriminating information is
not in the pixels. It is in the history a dermatologist asks for in ten seconds:

    groin/foot + itchy + spreading + prior steroid cream   -> tinea
    elbows/knees + chronic + silvery scale                 -> psoriasis
    flexures + childhood onset + intensely itchy           -> eczema

Image-only models (including every competitor's) discard this. That is the gap.

MISSINGNESS IS FIRST-CLASS
--------------------------
Public atlases (DermNet) have NO metadata, and real users skip questions. So every
field carries an explicit "unknown" slot, and `present_mask` tells the model which
fields were actually answered. Combined with modality dropout at training time
(see fusion.py) this keeps the model working image-only rather than collapsing when
metadata is absent.

Field vocabularies mirror console/shared/vocab.js exactly — one source of truth.
=========================================================================
"""
from __future__ import annotations

# Mirrors console/shared/vocab.js. Keep in sync; validate_vocab_sync() checks it.
BODY_SITES = ["Face", "Scalp", "Neck", "Chest", "Back", "Abdomen",
              "Groin", "Buttocks", "Arm", "Hand", "Leg", "Foot", "Nails", "Trunk", "Other"]
PRIOR_STEROID = ["yes", "no", "unknown"]
FITZPATRICK = ["I", "II", "III", "IV", "V", "VI"]
MONK = [str(i) for i in range(1, 11)]

# Free-text-ish app fields, bucketed. These come from the public capture flow
# (src/App.jsx) and are NOT yet in the manifest export — the encoder tolerates
# their absence so the schema can grow without breaking old manifests.
DURATION = ["days", "weeks", "months", "years"]
SYMPTOMS = ["itch", "pain", "burning", "none"]          # multi-hot: several can apply
RECENT_CHANGE = ["spreading", "shrinking", "same", "new"]

# (column name in the manifest, vocabulary, multi_hot?)
FIELDS: list[tuple[str, list[str], bool]] = [
    ("body_site", BODY_SITES, False),
    ("prior_steroid_use", PRIOR_STEROID, False),
    ("fitzpatrick", FITZPATRICK, False),
    ("monk", MONK, False),
    ("duration", DURATION, False),
    ("symptoms", SYMPTOMS, True),
    ("recent_change", RECENT_CHANGE, False),
]

# one-hot slots + one "unknown" slot per field, then a present-mask bit per field
_ONEHOT_DIM = sum(len(v) + 1 for _, v, _ in FIELDS)
FEATURE_DIM = _ONEHOT_DIM + len(FIELDS)
FIELD_NAMES = [f for f, _, _ in FIELDS]


def _norm(value) -> str:
    return "" if value is None else str(value).strip()


def encode(row: dict) -> list[float]:
    """Manifest row (or any dict of the same field names) -> fixed-length float vector.

    Unknown / missing / out-of-vocabulary values all land in the field's "unknown"
    slot and leave its present-mask bit at 0, so the model can tell "not asked" from
    "answered as unknown-ish".
    """
    vec: list[float] = []
    mask: list[float] = []
    for col, vocab, multi in FIELDS:
        slots = [0.0] * (len(vocab) + 1)      # last slot = unknown/missing
        raw = _norm(row.get(col))
        answered = False

        if raw and raw.lower() not in ("unknown", "none_given", "na", "n/a"):
            values = [v.strip() for v in raw.split("|")] if multi else [raw]
            for v in values:
                # case-insensitive match so "face"/"Face" and 3/"3" both work
                hit = next((i for i, k in enumerate(vocab) if k.lower() == v.lower()), None)
                if hit is not None:
                    slots[hit] = 1.0
                    answered = True
        if not answered:
            slots[-1] = 1.0
        vec.extend(slots)
        mask.append(1.0 if answered else 0.0)
    return vec + mask


def empty() -> list[float]:
    """The all-missing vector — what a DermNet row encodes to."""
    return encode({})


def describe(vec: list[float]) -> dict:
    """Inverse-ish view of a vector, for debugging and for showing what the model saw."""
    out, i = {}, 0
    for col, vocab, _ in FIELDS:
        slots = vec[i:i + len(vocab) + 1]
        i += len(vocab) + 1
        picked = [vocab[j] for j, s in enumerate(slots[:-1]) if s > 0]
        out[col] = picked or ("unknown" if slots[-1] > 0 else None)
    out["_present_mask"] = dict(zip(FIELD_NAMES, vec[_ONEHOT_DIM:]))
    return out


def validate_vocab_sync(vocab_js_path) -> list[str]:
    """Cheap guard against this file drifting from console/shared/vocab.js."""
    import re
    from pathlib import Path
    txt = Path(vocab_js_path).read_text()
    problems = []

    def js_list(name):
        m = re.search(rf"export const {name} = \[(.*?)\];", txt, re.S)
        return [] if not m else [s.strip().strip('"\'') for s in m.group(1).split(",") if s.strip()]

    for name, mine in (("BODY_SITES", BODY_SITES), ("PRIOR_STEROID", PRIOR_STEROID),
                       ("FITZPATRICK", FITZPATRICK)):
        theirs = js_list(name)
        if theirs and theirs != mine:
            problems.append(f"{name} differs: js={theirs} py={mine}")
    return problems


if __name__ == "__main__":
    from pathlib import Path
    print(f"FEATURE_DIM = {FEATURE_DIM}  ({len(FIELDS)} fields)")
    for col, vocab, multi in FIELDS:
        print(f"  {col:<18} {len(vocab)+1:>3} slots{'  (multi-hot)' if multi else ''}")

    print("\nDermNet row (no metadata at all):")
    print("  ", describe(empty()))

    print("\nA console case — classic steroid-modified tinea presentation:")
    row = {"body_site": "Groin", "prior_steroid_use": "yes", "fitzpatrick": "V",
           "monk": "8", "duration": "months", "symptoms": "itch|burning",
           "recent_change": "spreading"}
    print("  ", describe(encode(row)))

    js = Path(__file__).resolve().parent.parent / "console" / "shared" / "vocab.js"
    if js.exists():
        probs = validate_vocab_sync(js)
        print("\nvocab sync vs console/shared/vocab.js: " +
              ("OK" if not probs else "DRIFT -> " + "; ".join(probs)))
