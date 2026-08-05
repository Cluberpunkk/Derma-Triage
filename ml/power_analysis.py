"""ml/power_analysis.py
=========================================================================
What can 150-300 holdout images actually answer?

The subsample accident is the motivating evidence: the tone gradient did not hold
its SIGN at n=1200. If an effect is that fragile at 1200 curated images, planning to
detect it in 300 real ones is planning to collect data that cannot answer the
question.

For every question the holdout is supposed to answer, this computes the n needed for
80% power at alpha=0.05, using effect sizes measured on DermNet where we have them.
Then it states plainly which questions 300 images answer, which need thousands, and
which need a different DESIGN rather than more samples.

THE MULTIPLIER THAT DOMINATES EVERYTHING
----------------------------------------
Per-tone URGENT metrics need urgent cases within each tone group. DermNet's urgent
prevalence is 26%, but a consumer app's will be ~1-5% (see traffic_weighted.py). So
the images-needed figure is n_urgent / prevalence, and a 5% prevalence multiplies the
requirement twentyfold. This is why "collect 300 and measure fairness" does not work,
and no amount of careful analysis fixes it.

DESIGN ALTERNATIVES, when n is the wrong lever:
  paired within-subject   same lesion, same session, two conditions (e.g. reticle vs
                          upload path). Removes between-person variance, so the
                          required n drops by roughly the intra-class correlation.
  stratified oversampling deliberately recruit darker tones to equal group sizes
                          instead of convenience sampling. Power depends on the
                          SMALLER group, so balancing beats collecting more.
  enrichment              recruit through a dermatology clinic for the urgent arm,
                          rather than waiting for 5%-prevalence walk-ins.

    ml/.venv/bin/python ml/power_analysis.py
=========================================================================
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import runmeta  # noqa: E402

Z_ALPHA = 1.959964      # two-sided 0.05
Z_BETA = 0.841621       # 80% power
PLANNED = [150, 300]

# Real-world urgent prevalence: DermNet's 26.4% is an artifact of atlas curation.
APP_URGENT_PREVALENCE = 0.05
DARK_TONE_SHARE = 0.45   # assumed share of an Indian user base in the darker ITA bins


def n_two_proportions(p1: float, p2: float) -> float:
    """Per-group n to detect p1 vs p2 at 80% power, alpha=0.05."""
    d = abs(p1 - p2)
    if d < 1e-9:
        return float("inf")
    return (Z_ALPHA + Z_BETA) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / d ** 2


def n_for_ci_width(p: float, half_width: float) -> float:
    """n to estimate a single proportion to +/- half_width."""
    return (Z_ALPHA ** 2) * p * (1 - p) / half_width ** 2


def n_two_means(d_cohen: float) -> float:
    """Per-group n to detect a standardised mean difference."""
    if d_cohen <= 0:
        return float("inf")
    return 2 * (Z_ALPHA + Z_BETA) ** 2 / d_cohen ** 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prevalence", type=float, default=APP_URGENT_PREVALENCE)
    a = ap.parse_args()

    meta = runmeta.manifest(tool="power_analysis.py",
                            checkpoint=config.out_dir() / "runs_final" / "best.pt",
                            extra={"alpha": 0.05, "power": 0.80,
                                   "assumed_app_urgent_prevalence": a.prevalence,
                                   "assumed_dark_tone_share": DARK_TONE_SHARE})
    runmeta.print_header(meta)
    print(f"\nalpha=0.05, power=80%. Effect sizes from DermNet where measured.")
    print(f"Assumed real-world urgent prevalence: {a.prevalence:.0%} "
          f"(DermNet's 26.4% is an artifact of curation).\n")

    Q = []

    # --- 1. tone gradient in localiser fallback (the effect that failed at n=1200) ---
    n_grp = n_two_proportions(0.124, 0.203)
    Q.append({"question": "per-tone localiser fallback rate (light vs dark)",
              "effect": "0.124 vs 0.203 (measured, full split)",
              "n_per_group": n_grp,
              "images_needed": n_grp * 2 / DARK_TONE_SHARE / (1 / DARK_TONE_SHARE),
              "images_total": n_grp / DARK_TONE_SHARE,
              "note": "needs only tone labels, not urgency — the cheapest tone question"})

    # --- 2. per-tone quality-gate rejection ---
    n_grp = n_two_proportions(0.382, 0.618)     # measured at VoL 164, identical blur
    Q.append({"question": "per-tone quality-gate rejection (blur)",
              "effect": "0.382 vs 0.618 (measured, synthetic blur at VoL 164)",
              "n_per_group": n_grp, "images_total": n_grp / DARK_TONE_SHARE,
              "note": "large effect -> cheapest to detect; needs only tone + gate output"})

    # --- 3. per-tone urgent recall — the prevalence multiplier bites here ---
    n_grp = n_two_proportions(0.736, 0.818)     # urgRec by ITA bin, measured
    Q.append({"question": "per-tone urgent RECALL (light vs dark)",
              "effect": "0.736 vs 0.818 (measured by ITA bin)",
              "n_per_group": n_grp,
              "n_urgent_needed": n_grp * 2,
              "images_total": n_grp / DARK_TONE_SHARE / a.prevalence,
              "note": "needs URGENT cases within each tone group — prevalence multiplier"})

    # --- 4. per-tone top-k capture ---
    n_grp = n_two_proportions(0.328, 0.319)     # cap@10 light vs dark, measured
    Q.append({"question": "per-tone top-10% capture (light vs dark)",
              "effect": "0.328 vs 0.319 (measured — effectively zero)",
              "n_per_group": n_grp,
              "images_total": (n_grp / DARK_TONE_SHARE / a.prevalence
                               if math.isfinite(n_grp) else float("inf")),
              "note": "measured difference is ~0.009; detecting it is not a realistic goal"})

    # --- 5. symptom-gate yield (estimation, not comparison) ---
    n = n_for_ci_width(0.5, 0.10)
    Q.append({"question": "symptom-gate yield (estimate to +/-10%)",
              "effect": "estimation, worst-case p=0.5",
              "n_per_group": n, "images_total": n,
              "note": "ESTIMATION not comparison — much cheaper. +/-5% would need "
                      f"{n_for_ci_width(0.5, 0.05):.0f}"})

    # --- 6. distribution shift vs DermNet ---
    n = n_two_means(0.8)
    Q.append({"question": "distribution shift vs DermNet (feature distance)",
              "effect": "assumed large, Cohen d=0.8",
              "n_per_group": n, "images_total": n,
              "note": "one-sample against a fixed reference -> the cheapest question here"})

    print(f"{'question':<52} {'n/group':>9} {'IMAGES':>10}  verdict")
    for q in Q:
        tot = q["images_total"]
        v = ("300 IS ENOUGH" if tot <= 300 else
             "needs ~1k" if tot <= 1500 else
             "needs THOUSANDS" if tot <= 20000 else
             "NOT ANSWERABLE by sample size")
        q["verdict"] = v
        print(f"{q['question'][:52]:<52} {q['n_per_group']:>9.0f} "
              f"{tot:>10,.0f}  {v}")
        print(f"{'':<52} {'':>9} {'':>10}  {q['note']}")

    print(f"\n=== WHAT 150-300 IMAGES ACTUALLY BUYS ===")
    can = [q for q in Q if q["images_total"] <= 300]
    cant = [q for q in Q if q["images_total"] > 300]
    print(f"  ANSWERABLE at 300 ({len(can)}):")
    for q in can:
        print(f"    - {q['question']}")
    print(f"  NOT ANSWERABLE at 300 ({len(cant)}):")
    for q in cant:
        print(f"    - {q['question']}  (needs ~{q['images_total']:,.0f})")

    print(f"\n=== WHY FAIRNESS EVALUATION IS THE EXPENSIVE ONE ===")
    n_rec = n_two_proportions(0.736, 0.818)
    print(f"  per-tone urgent recall needs {n_rec:.0f} URGENT cases per tone group.")
    print(f"  At {a.prevalence:.0%} urgent prevalence and {DARK_TONE_SHARE:.0%} darker-tone "
          f"share, that is {n_rec / DARK_TONE_SHARE / a.prevalence:,.0f} images.")
    print(f"  At DermNet's 26.4% prevalence it would be "
          f"{n_rec / DARK_TONE_SHARE / 0.264:,.0f} — the prevalence multiplier is the "
          f"whole difference.")

    # ---------------- per-RULE-GROUP symptom yield ----------------
    # The gate is now the load-bearing safety mechanism, so its yield is the primary
    # validation target. Aggregate yield can hide a group that never fires, so each
    # group must be powered separately — and yield is measured AMONG the cases the
    # group is meant to catch, which reintroduces a prevalence multiplier per condition.
    print(f"\n=== SYMPTOM-GATE YIELD, PER RULE GROUP ===")
    print(f"  yield is measured among the cases a group should catch, so each group")
    print(f"  inherits the prevalence of ITS conditions.")
    n_est = n_for_ci_width(0.5, 0.10)
    GROUP_PREV = {           # share of app traffic, assumed
        "spreading_infection (cellulitis)": 0.010,
        "systemic_involvement (SJS/TEN, angioedema)": 0.002,
        "systemic_illness (lupus, vasculitis)": 0.005,
        "non_healing_or_changing (malignancy)": 0.010,
        "pigmented_lesion (SK vs BCC)": 0.080,
        "vulnerable_host": 0.050,
    }
    groups_out = []
    print(f"  {'rule group':<44} {'assumed prev':>12} {'IMAGES for +/-10%':>18}")
    for g, prev in GROUP_PREV.items():
        need = n_est / prev
        groups_out.append({"group": g, "assumed_prevalence": prev,
                           "cases_needed": n_est, "images_needed": need,
                           "answerable_at_300": bool(need <= 300)})
        print(f"  {g:<44} {prev:>11.1%} {need:>18,.0f}"
              + ("  <- OK at 300" if need <= 300 else ""))
    print(f"  Aggregate yield at +/-10% needs only {n_est:.0f} images — but that number")
    print(f"  can be dominated by pigmented_lesion and hide a time-critical group that")
    print(f"  never fires. Per-group is the measurement that matters.")

    # ---------------- per-rule FIRING RATE on the general population ----------------
    # The asymmetry that makes 300 captures worth collecting:
    #
    #   SENSITIVITY needs true cases, which are rare  -> 9,604 to 48,018 images
    #   SPECIFICITY needs NON-cases, which are abundant -> a few hundred
    #
    # So 300 general-population captures cannot tell us whether the gate CATCHES
    # cellulitis, but they tell us precisely how often it fires on people who do not
    # have it. That matters on its own: a rule firing on a third of ordinary users is
    # a product failure whatever its sensitivity turns out to be, and it is the direct
    # test of the 2-of-4 / 2-of-6 thresholds, which are ours and unvalidated.
    print(f"\n=== PER-RULE FIRING RATE ON GENERAL POPULATION (specificity side) ===")
    print(f"  needs NON-cases, which every capture provides -> cheap at n=300")
    # Tolerance is ANCHORED TO CONDITION PREVALENCE, not to a round number.
    #
    # An earlier version used a flat 5%/15% by tier. That is the same defect as the
    # retired BLOCKER_FLAG_RATE: a bar invented rather than derived, against which an
    # observed rate would "pass" or "fail" for no reason connected to the condition.
    #
    # Instead: for each rule, state the expected presenting prevalence of its target
    # condition, and set tolerance as a CHOSEN MULTIPLE of it. Over-referral is the
    # safe direction, so multiples are deliberately > 1. The multiple is larger where
    # missing the condition is catastrophic (SJS/TEN) and smaller where the rule fires
    # at a tier that disrupts the user's day (a "today" escalation costs more per
    # firing than a "soon" one).
    #
    # PREVALENCE FIGURES ARE ASSUMED, NOT CITED. I do not have sourced presenting-
    # prevalence estimates for an Indian teledermatology population, and inventing
    # citations would be worse than saying so. They are review items on
    # docs/RULE_REVIEW_form.md — a partner clinic can replace them with their own
    # case mix, at which point every tolerance below updates automatically.
    RULES_ANCHORED = {
        "spreading_infection": {
            "plausibility": 'moderate',
            "plausibility_note": 'Cellulitis incidence is roughly 200/100k/year, but many present to a GP or ED rather than a photo app. 1% of app traffic is plausible, possibly high.',
            "tier": "today", "condition": "cellulitis / spreading bacterial infection",
            "prevalence": 0.010, "multiple": 5,
            "why": "5x: progresses in hours, so over-referral is cheap relative to a "
                   "miss; but 'today' disrupts the user's day, so not 20x"},
        "systemic_involvement": {
            "plausibility": 'VERY WEAK - LIKELY WRONG BY ORDERS OF MAGNITUDE',
            "plausibility_note": 'SJS/TEN incidence is ~1-2 per MILLION per year, and those patients are hospitalised, not photographing a rash. Presenting prevalence in this app is plausibly <0.01%, i.e. 20x lower than assumed - and if so the effective multiple is ~500x, so the rule would pass at almost any firing rate. That INVERTS the intent: the rule meant to get the strictest scrutiny gets the most permissive bar. The 0.2% figure is partly carried by angioedema, which is commoner than SJS/TEN, but the two are lumped and should not be. TOP PRIORITY for the rule-review form.',
            "tier": "today", "condition": "SJS/TEN, angioedema",
            "prevalence": 0.002, "multiple": 25,
            "why": "25x: catastrophic and rapidly fatal if missed. The highest multiple "
                   "on the form, deliberately — at 0.2% prevalence even 5% firing is "
                   "25 false alarms per true case, and that is the right trade here"},
        "systemic_illness": {
            "plausibility": 'weak',
            "plausibility_note": 'SLE prevalence is ~20-150/100k (0.02-0.15%); vasculitis is rarer. 0.5% is likely high by ~5x, which makes this tolerance correspondingly permissive.',
            "tier": "soon", "condition": "lupus / vasculitis / connective tissue disease",
            "prevalence": 0.005, "multiple": 10,
            "why": "10x: serious but not same-day; 'soon' tier is cheap per firing"},
        "non_healing_or_changing": {
            "plausibility": 'weak, and India-specific',
            "plausibility_note": 'Skin malignancy is far less common in Indian than Western populations, and most published presenting-prevalence figures are Western. 1% is plausibly high by 2-10x for this population.',
            "tier": "soon", "condition": "skin malignancy",
            "prevalence": 0.010, "multiple": 10,
            "why": "10x: our worst-performing model class (AK/BCC recall 0.573), so the "
                   "questionnaire is carrying more of the load here than elsewhere"},
        "pigmented_lesion": {
            "plausibility": 'moderate',
            "plausibility_note": 'Pigmented-lesion complaints are genuinely common; 8% of a dermatology-app case mix is defensible. The most plausible figure on the form.',
            "tier": "soon", "condition": "pigmented lesion needing SK-vs-BCC judgement",
            "prevalence": 0.080, "multiple": 2,
            "why": "2x only: the target is COMMON, so a large multiple would fire on a "
                   "big fraction of all users. This is the rule most at risk of "
                   "over-firing, and its 2-of-6 threshold is the one to watch"},
        "vulnerable_host": {
            "plausibility": 'moderate',
            "plausibility_note": 'Pregnancy alone accounts for a few percent of adult women at any time; adding immunosuppression and neonates, 5% is reasonable.',
            "tier": "soon", "condition": "immunosuppressed / neonate / pregnant",
            "prevalence": 0.050, "multiple": 3,
            "why": "3x: host factors are self-reported and common; a high multiple "
                   "would escalate a large share of ordinary users"},
    }
    print(f"  tolerance = condition prevalence x a CHOSEN multiple (rationale per rule)")
    print(f"  prevalence figures are ASSUMED — review items, not citations\n")
    def min_detectable_overshoot(tol: float, n: int = 300) -> float:
        """Smallest TRUE firing rate whose CI lower bound at n excludes the tolerance.

        This is what the pilot can actually REJECT. A tolerance of 5% with a minimum
        detectable overshoot of 9% means the pilot can prove a rule fires too often
        only once it is firing at nearly twice the bar — it cannot confirm the rule
        sits AT the bar, and reporting "decidable: yes" would have implied it could.
        """
        pp = tol
        while pp < 0.99:
            if pp - Z_ALPHA * ((pp * (1 - pp)) / n) ** 0.5 > tol:
                return pp
            pp += 0.001
        return float("nan")

    print(f"  {'rule':<24} {'prev':>6} {'x':>3} {'tol':>7} {'min detectable':>15} "
          f"{'as x tol':>9} {'plausibility':>12}")
    firing = []
    for rule, d in RULES_ANCHORED.items():
        tol = min(0.95, d["prevalence"] * d["multiple"])
        hw300 = Z_ALPHA * ((tol * (1 - tol)) / 300) ** 0.5
        mdo = min_detectable_overshoot(tol)
        # n to distinguish the bar itself from 1.5x the bar
        n_decide_bar = n_two_proportions(tol, tol * 1.5)
        firing.append({"rule": rule, "tier": d["tier"], "condition": d["condition"],
                       "assumed_prevalence": d["prevalence"],
                       "plausibility": d["plausibility"],
                       "plausibility_note": d["plausibility_note"],
                       "multiple": d["multiple"], "tolerable_rate": tol,
                       "rationale": d["why"], "ci_halfwidth_at_300": hw300,
                       "min_detectable_overshoot_at_300": mdo,
                       "overshoot_as_multiple_of_tolerance": mdo / tol if tol else None,
                       "n_to_decide_the_bar_itself": n_decide_bar})
        short = d["plausibility"].split(" ")[0][:12]
        print(f"  {rule:<24} {d['prevalence']:>5.1%} {d['multiple']:>3}x {tol:>7.1%} "
              f"{mdo:>14.1%} {mdo / tol:>8.1f}x {short:>12}")
    print(f"\n  'min detectable' = the smallest true firing rate the pilot (n=300) can")
    print(f"  prove exceeds the tolerance. NOT 'the pilot settles this' — at these n the")
    print(f"  pilot rejects only gross overshoot, typically ~2x the bar.")
    print(f"  To decide whether a rule sits AT its bar (bar vs 1.5x bar) needs:")
    for f in firing:
        print(f"    {f['rule']:<24} n = {f['n_to_decide_the_bar_itself']:>7,.0f}")

    print(f"\n  PLAUSIBILITY OF THE PREVALENCE ANCHORS — a prevalence set too HIGH")
    print(f"  produces a tolerance too PERMISSIVE, so errors here fail in the unsafe")
    print(f"  direction (the rule passes when it should not).")
    for f in firing:
        print(f"    [{f['plausibility']}] {f['rule']}")
        print(f"      {f['plausibility_note']}")

    for f in firing:
        print(f"\n  {f['rule']}: {f['rationale']}")
    print(f"\n  At n=300 every rule's firing rate is estimable to within a few points —")
    print(f"  enough to tell a rule firing at 2% from one firing at 30%. That is the")
    print(f"  question 300 captures CAN answer about the safety mechanism.")
    print(f"  Report per rule with CIs and flag any rule above its tolerance.")

    # ---------------- continuous fairness substitute ----------------
    # Recall is a binary outcome on rare events -> 17,804 images. A CONTINUOUS
    # comparison of the urgency score between tone groups is far cheaper because every
    # image contributes information, not just the urgent ones.
    print(f"\n=== CONTINUOUS FAIRNESS SUBSTITUTE (score distribution by tone) ===")
    print(f"  {'effect (Cohen d)':>17} {'n/group':>9} {'IMAGES (50/50)':>15}  paired (ICC 0.5)")
    cont = []
    for d in (0.15, 0.20, 0.30, 0.40, 0.50):
        n = n_two_means(d)
        paired = n * (1 - 0.5)      # within-subject removes between-person variance
        cont.append({"cohen_d": d, "n_per_group": n, "images_unpaired": n * 2,
                     "images_paired_icc0.5": paired * 2})
        print(f"  {d:>17.2f} {n:>9.0f} {n * 2:>15,.0f}  {paired * 2:>,.0f}")
    print(f"  A systematic score depression on darker skin of d=0.3 is detectable with")
    print(f"  ~{n_two_means(0.3) * 2:,.0f} images, or ~{n_two_means(0.3):,.0f} paired.")
    print(f"  NOT A FAIRNESS GUARANTEE: equal score distributions do not imply equal")
    print(f"  recall. It is a SENSITIVITY SIGNAL that detects systematic depression at")
    print(f"  achievable n, and it is the only tone metric in that category.")

    print(f"\n=== DESIGN CHANGES THAT BEAT COLLECTING MORE ===")
    print(f"  1. STRATIFIED OVERSAMPLING of darker tones. Power depends on the SMALLER")
    print(f"     group, so balancing 50/50 beats convenience sampling at any total n.")
    print(f"  2. PAIRED WITHIN-SUBJECT for path comparisons (reticle vs upload, burst vs")
    print(f"     single). Same lesion, same lighting, same person removes between-person")
    print(f"     variance — typically a 3-10x reduction in required n.")
    print(f"  3. CLINIC ENRICHMENT for the urgent arm. Recruiting urgent cases directly")
    print(f"     defeats the prevalence multiplier; it is the ONLY thing that does.")
    print(f"  4. Report CIs and decline to conclude, rather than running an underpowered")
    print(f"     test and reading its point estimate.")

    print(f"\n=== RECOMMENDED PILOT SCOPE ===")
    print(f"  300 images, stratified ~50/50 light/dark, answers:")
    print(f"    - PER-RULE FIRING RATE on the general population, with CIs (the direct")
    print(f"      test of our assumed 2-of-4 / 2-of-6 thresholds)")
    print(f"    - distribution shift vs DermNet (the binding uncertainty)")
    print(f"    - AGGREGATE symptom-gate yield to about +/-10% (NOT per-group)")
    print(f"    - quality-gate rejection RATE and its tone gradient (large effect)")
    print(f"    - whether the pipeline works end to end at all")
    print(f"  It does NOT answer per-tone urgent recall or per-tone capture. Those need")
    print(f"  clinic enrichment, and should be planned as a separate study rather than")
    print(f"  hoped for from the same 300.")

    out = {**meta, "questions": Q,
           "symptom_yield_per_group": groups_out,
           "per_rule_firing_rate": firing,
           "continuous_fairness_substitute": cont,
           "answerable_at_300": [q["question"] for q in can],
           "not_answerable_at_300": [q["question"] for q in cant],
           "planned_sizes": PLANNED,
           "caveat": "Effect sizes are from DermNet and may not transfer. Where the real "
                     "effect is smaller, required n rises as the inverse square."}
    p = config.out_dir() / "runs_final" / "power_analysis.json"
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
