# Session summary — what was built, what was claimed, what was superseded

One page for a stakeholder. Every superseded claim is named with its current value.
Numbers regenerate from `data/ml/runs_final/CANONICAL.json`
(`ml/regenerate_report.py`), frozen test split `ce6761fd5b40`, n=3950.

---

## What exists now

| | |
|---|---|
| **Model** | resnet50 @224px, 20 epochs, ImageNet-pretrained, trained locally on Apple MPS |
| **Test** | top-1 **0.571** · top-3 **0.772** · macro-F1 **0.555** · balanced-acc 0.570 |
| **Urgency (derived)** | AUROC **0.878** · AUPRC **0.745** (no-skill 0.264) |
| **Queue (the shipping metric)** | top-5% captures 17.5% of urgent (3.5× random); top-10% captures 33.0% |
| **Data** | DermNet, 19,559 images → train 12,201 / val 2,151 / test 3,950 |

Progression: macro-F1 **0.229** (resnet18 baseline) → **0.491** (@192px) → **0.555** (@224px, final).

Built: dataset adapter with dedup, evaluation harness, abstention, OOD detection,
metadata fusion, decision curves, symptom gate, full 5-gate pipeline, and a
regeneration script that rewrites docs from measured values.

---

## Claims made and later superseded

| # | Claim | Status | Current value |
|---|---|---|---|
| 1 | Kaggle GPU training would take ~40 min | **wrong** | Account had **no GPU and no internet**; all training ran locally on MPS |
| 2 | The weight-download failure was a HuggingFace client bug | **wrong** | It was **no network at all** (DNS unresolved) — same root cause as #1 |
| 3 | `enable_gpu` failed because I passed a boolean not a string | **wrong** | Kaggle **stored** `enable_gpu: true` and ignored it; cause is account-level |
| 4 | `map_kind` results show label-noise (direct > coarse) | **inverted** | With a trained model: none 0.608 > coarse 0.553 > direct 0.531. Cause is visual distinctiveness, not label cleanliness |
| 5 | The model uses pigmentation as a proxy for malignancy | **refuted** | ρ=+0.144 within routine (r²≈0.02); Grad-CAM↔darkness on false alarms **−0.017** |
| 6 | Resolution ablation shows no resolution sensitivity | **invalid** | FixRes artifact — head not retrained per resolution; top-1 collapsed 0.575→0.311 |
| 7 | Source-resolution bucketing will settle it | **unanswerable** | 97% of DermNet sits in one bucket (median 477px). Deferred to holdout |
| 8 | A sum of calibrated marginals is not calibrated | **overstated** | Measured: already near-calibrated (binary ECE 0.041 → 0.021 after Platt) |
| 9 | Abstention thresholds 0.70 / 0.90 | **invalid basis** | Set on **uncalibrated** confidences (model was over-confident, T=1.707) |
| 10 | **Blocker 1: 71.4% of benign pigmentation flagged urgent** | **superseded** | **30.2%** above the ignorance floor; and it does **not bind** under ranking (3.7% queue occupancy) |
| 11 | **588 false alarms per missed cancer** | **superseded** | **49:1** at the first above-floor threshold |
| 12 | Cellulitis ranks significantly worse than aggregate | **overstated** | CI 0.17–0.37 includes aggregate 0.330. The real problem is **absolute**: only 26% in top-10% |
| 13 | Floor-corrected score is a monotone transform | **wrong in detail** | Clipping at zero breaks monotonicity; AUROC 0.878 → **0.828**. Use log-odds, don't clip |
| 14 | max-urgent-prob may dominate the sum | **refuted** | Worse at matched recall (0.402 vs 0.360 inflammatory flag) |
| 15 | Urgency score is not conflated with uncertainty | **wrong** (my automated verdict) | **Confirmed conflated**: a uniform prediction scores 0.309 vs a 0.12 threshold; 70.6% of flagged inflammatory cases have a *routine* top-1 |

**Pattern worth noting:** most reversals came from measuring something I had asserted.
Four of the fifteen were my own hypotheses failing; three were measurement errors of
mine; the rest were inherited assumptions that had never been checked.

---

## The ignorance floor — the single most consequential finding

A uniform prediction over 23 classes puts 8/23 into the urgent group: **0.3089
calibrated**. Every operational number produced at t=0.12 was measured *below* it,
where a blank prediction fires the gate. Re-baselining above the floor changed
Blocker 1 from 0.714 to 0.302 and the stakeholder ratio from 588:1 to 49:1.

The cost is stated, not hidden: urgent recall falls 0.946 → 0.763, and the loss
concentrates on the **time-critical** classes (cellulitis −0.233, bullous −0.239)
rather than on melanoma (−0.123).

---

## What actually blocks the product

| blocker | binds? | why |
|---|---|---|
| **1 — pigmentation false alarms** | **No** | Constrains a user-facing urgency flag we are not shipping; 3.7% queue occupancy under ranking |
| **2 — time-critical classes rank low** | **Yes** | Measured on *rank*, which is what we ship. Cellulitis 0.260 capture @top-10%. Not fixable by ranking — needs the symptom questionnaire |
| **3 — no brown-skin, no phone-photo evidence** | **Hardest** | Every number above is from curated atlas photos of a mostly light-skinned population. `fairness_measurable: false` in all outputs |

---

## Honest position

The model is a **coarse 23-category atlas classifier**. It cannot name a disease
reliably, cannot assess faces or "skin quality", and **must never be used to
reassure** — it misses 36% of malignancies when answering everything.

What it can support: **ranking a review queue for a human** (3.5× lift at top-5%),
refusing bad or non-skin input, and a shortlist to bring to a clinician.

The next evidence that matters is not a better model. It is 150–300 real phone photos
of Indian skin, collected through the production capture path, with Fitzpatrick tone
and symptom answers recorded.

---

*Unreviewed by any dermatologist. All clinical groupings carry `reviewed_by: null`.*
