# DermaTriage — prototype

A skin **triage + education** prototype for Indian skin. It takes a photo and a few
patient details and returns: how urgently to see a doctor, possibilities to discuss
with a dermatologist, red flags, and questions to bring to the visit.

It is **not a diagnosis and not a medical device.**

---

## ⚠️ Read this before you build on it

The "intelligence" here is a **general vision model used as a placeholder.** It is
NOT a trained dermatology classifier. It will be impressive on some cases and
confidently wrong on others — especially on darker skin, and especially on
steroid-modified tinea, the most common (and most disguised) presentation in India.

A polished demo proves the **experience**, not the **medicine**. Don't let a good
demo on a placeholder engine convince you the model is ready. It isn't — see Roadmap.

---

## Architecture

```
Browser (React)  →  your backend proxy (server/proxy.js)  →  vision model
```

- `src/App.jsx` — the UI and triage flow.
- `src/lib/clinicalPrompt.js` — **the clinical knowledge layer** (the rules the model follows).
- `src/lib/triageEngine.js` — calls the proxy, parses the JSON result.
- `server/proxy.js` — holds your API key, forwards to the model.
  Never call the model API directly from the browser: it fails CORS and exposes your key.

---

## Where dermatology expertise actually enters this system

You keep asking how to "feed it 15 years + a PhD." Here is the honest map — there are
three places expertise can enter, and **only one of them is typing knowledge in.**

**1. The system prompt — `src/lib/clinicalPrompt.js` (do this now, in Claude Code).**
Encode clinical *reasoning*: how to weigh a differential, what forces escalation,
India-specific patterns. This is real and useful. **Ceiling:** it instructs a general
model that still can't reliably *see* dermatology. A good triage scaffold, not a
diagnostic engine.

**2. A retrieval knowledge base (optional).**
Ground the *educational text* in curated references — **your own clinical notes, open
guidelines, or licensed content.** Do NOT scrape copyrighted textbooks (Rook's,
Bolognia, Fitzpatrick's, journals) — that's IP infringement and legally radioactive
for a medical product. Retrieval improves the *words*, not the *eyes*.

**3. Training data with your labels — the real model (the months-of-work path).**
This is where 15 years + a PhD actually *becomes* the model: not as text you type,
but as the **expert labels you put on thousands of Indian-skin images.** Your
expertise is the ground truth that teaches the model to *see*. That is the only lever
that fixes visual accuracy.

So in Claude Code: sharpen levers 1 and 2. But your expertise enters the **model**
through *labelling data*, not prompting.

---

## Run it locally

Requires Node 18+ and an Anthropic API key.

1. `npm install`
2. `export ANTHROPIC_API_KEY=sk-ant-...`  (Windows PowerShell: `$env:ANTHROPIC_API_KEY="sk-ant-..."`)
3. In one terminal: `npm run proxy`   — starts the backend on `:8787`
4. In another terminal: `npm run dev` — starts the app; open the printed URL

The Vite dev server proxies `/api` → the backend, so your key never reaches the browser.

---

## Data collection + expert-labelling pipeline (`console/`)

This turns real, **consented** usage into a confirmed-label, ML-ready dataset. It is a
**separate, auth-gated system** from the public app and is not reachable from the public
capture/triage flow — the public app deploys and collects data without it.

**Two decoupled systems, one data substrate.**

- **Public app + `server/proxy.js` (:8787)** — adds a *training-consent* screen before
  capture and a `POST /api/case` ingestion endpoint. On submit, if the user consented, the
  image (EXIF/GPS stripped; optional on-device face blur) is written to `data/images/` and a
  row is inserted in SQLite. No consent → nothing stored, triage still works.
- **Labelling console — `console/server.js` (:8788) + `console/web/`** — its own Express
  backend and its own Vite app, gated by a single admin credential. Fast review queue
  (hotkeys, image prefetch, autosave), the full expert schema, adjudication, a data-quality
  dashboard, and a reproducible stratified export. Images are streamed only through an
  auth-gated route — never a public URL.

Shared substrate only: `server/lib/{db,storage,images}.js` and the gitignored `data/` dir.

**Run the console (local):**

1. `npm run console:build` — builds the console web app to `console/web/dist`
2. `export CONSOLE_PASSWORD=choose-a-strong-secret`
3. `npm run console:server` — serves the console + API on `:8788`; open `http://localhost:8788`
   - For hot-reload dev instead of the built app: `npm run console:web` (Vite on `:5174`,
     proxying `/console-api` → `:8788`).

Optional env: `DATA_DIR` (default `./data`), `DB_PATH`, `CONSOLE_PORT`, `CONSOLE_SECRET`.

**Ground-truth rules (enforced, not cosmetic):**

- No case enters the training export without a **dermatologist-confirmed** label.
  `confirmation_method` and skin tone (**Fitzpatrick + Monk**) are mandatory on every label.
- Self-reported / app-generated values are shown as context only and are **never** treated
  as ground truth.
- Splits are **stratified by skin tone × diagnosis** with a **fixed seed** (reproducible), so
  the held-out test set carries Fitzpatrick V–VI representation. Only confirmed, gradable
  cases are placed in train/val/test.
- Consent-gated, EXIF-stripped, images never publicly exposed; right-to-withdraw supported
  (withdrawal removes a case from queue and export) — a DPDP-appropriate posture.

The controlled vocabulary lives in `console/shared/vocab.js` and is meant to be edited.

---

## Public-atlas pretraining track (`ml/`)

A **research-only** track that adapts the DermNet public atlas into the same manifest
schema and trains a coarse category classifier. Not wired into the app.

```bash
python3 -m venv ml/.venv && ml/.venv/bin/pip install -r ml/requirements.txt
ml/.venv/bin/python ml/config.py             # show resolved paths/settings
ml/.venv/bin/python ml/dermnet_adapter.py    # class folders -> manifest.csv
ml/.venv/bin/python ml/train.py --smoke      # CPU sanity run (~5 s)
```

Dataset path is configuration, not code: `DERMNET_DIR` (default `./data`,
`/kaggle/input/dermnet` on Kaggle) with `image_path` stored relative to it, so one
manifest runs on both. GPU instructions in [`ml/KAGGLE.md`](ml/KAGGLE.md).

<!-- AUTOGEN:blocker1 -->
> ### ⚠️ BLOCKER 1 — CONDITIONAL, does not currently bind: benign pigmentation false alarms
>
> **Ignorance floor = 0.3089.** A uniform prediction over
> 3,950 test cases scores that, because 8 of 23 classes are in the urgent
> group. No threshold below it separates urgency from ignorance.
>
> | | t=0.12 (**below floor**) | t=0.32 (above floor) |
> |---|---|---|
> | benign-pigmented flag rate | 0.714 | **0.302** |
> | inflammatory flag rate | 0.506 | 0.179 |
> | urgent recall | 0.946 | 0.763 |
> | missed urgent | 56 | 247 |
> | false alarms : missed cancer | 588:1 | 49:1 |
>
> **SUPERSEDED TWICE.** (1) The earlier record quoted 0.714 and 588:1, both measured
> at t=0.12 — **below the ignorance floor**, where a blank prediction also fires the
> gate. Above the floor the rate is 0.302 and the ratio
> 49:1. (2) The old pass/fail rule was a **0.30 bar I
> invented**, which this fired against by 0.002 — that decided nothing. It is replaced
> by the architecture test: *a blocker binds iff a shipping surface consumes the
> metric*. This one is consumed only by user-facing urgency flag (ARCHITECTURE_decision.md surface #7), which does not ship,
> so it is **conditional**. Under the ranking architecture we do ship, benign pigmented
> lesions reach the top-10% queue 3.7% of the time. If a user-facing urgency flag is
> ever added, this binds immediately.
>
> The cost of moving above the floor is real and is not hidden: urgent recall falls
> 0.946 → 0.763 and missed urgent cases rise
> 56 → 247.
>
> Regenerated by `ml/regenerate_report.py` from the frozen test split
> `ce6761fd5b40`.
<!-- /AUTOGEN:blocker1 -->


<!-- AUTOGEN:blocker2 -->
> ### ⛔ BLOCKER 2: time-critical classes rank and recall worst
>
> Per-class urgent recall, worst first — and the trade that moving above the
> ignorance floor imposes:
>
> | class | recall @0.12 (below floor) | recall @0.32 (above) | missed |
> |---|---|---|---|
> | Cellulitis Impetigo and other Bacterial In | 0.918 | 0.685 [0.56–0.78] | 23 |
> | Bullous Disease Photos | 0.929 | 0.690 [0.61–0.77] | 35 |
> | Lupus and other Connective Tissue diseases | 0.933 | 0.695 [0.61–0.78] | 32 |
> | Exanthems and Drug Eruptions | 0.929 | 0.717 [0.64–0.80] | 28 |
> | Actinic Keratosis Basal Cell Carcinoma and | 0.944 | 0.762 [0.72–0.81] | 68 |
>
> **The trade is explicit: fixing Blocker 1 by moving above the floor makes Blocker 2
> worse, and it does so hardest on the time-critical classes.** Aggregate recall falls
> 0.946 → 0.763, but the loss
> concentrates in cellulitis, bullous disease and exanthems — the classes that
> progress fastest.
>
> Queue capture at top-10% (threshold-free, so **unaffected by the floor**):
>
> | class | capture @top-10% |
> |---|---|
> | Bullous Disease Photos | 0.257 [0.19–0.34] |
> | Cellulitis Impetigo and other Bacterial In | 0.260 [0.16–0.36] |
> | Actinic Keratosis Basal Cell Carcinoma and | 0.266 [0.22–0.32] |
> | Lupus and other Connective Tissue diseases | 0.286 [0.20–0.37] |
> | *aggregate* | *0.330* |
>
> Image-score ranking alone cannot be the triage mechanism for fast-progressing
> disease. That needs the symptom questionnaire (`ml/symptom_gate.py`), which is
> model-independent and escalate-only.
>
> Regenerated by `ml/regenerate_report.py`.
<!-- /AUTOGEN:blocker2 -->


**No fairness claim is measurable.** Every evaluation in `ml/` prints
`fitzpatrick: unknown` and `fairness_measurable: false`, because this corpus carries
no skin-tone labels. No per-skin-tone metric exists for any model trained here, and
none can be inferred from overall accuracy. That stays true until Fitzpatrick-labelled
data is added — Fitzpatrick17k for training, DDI (biopsy-confirmed, dark-skin-weighted)
held out as the fairness test set. Until then the app's skin-tone reliability caveat is
a statement of fact, not caution.

**This corpus is not, and cannot become, clinical ground truth.** DermNet labels are
folder-level atlas categories with **no skin-tone labels**, so every row carries
`label_authority=public_atlas`, `confirmation_method=none`, `fitzpatrick=unknown`,
`gradable=ungraded`. `validateLabel()` rejects those values and `ml/config.py` refuses
to write into `data/exports/`, so atlas rows cannot leak into the confirmed-label
export. It also means **no fairness metric across skin tones is computable here** —
the skin-tone caveat in the app stands unchanged. Details and the folder→vocabulary
mapping (with per-folder confidence) are in [`ml/README.md`](ml/README.md).

---

## What it deliberately will NOT do (and why)

- **No diagnosis** — only possibilities to discuss with a clinician.
- **No treatment or product suggestions** — in India, self-applied steroid/combination
  creams are a leading cause of worsened, hard-to-treat infections, and a photo can't
  safely tell those apart. This rule is load-bearing safety, not a limitation to remove.
- **Always escalates red flags**, never reassures away anything that looks serious.
- **States its skin-tone limitation** openly.

If you "enhance" it by removing these, you are building a less safe product, not a better one.

---

## Roadmap to a real (not placeholder) engine

1. **Data pipeline** — confirmed-label, skin-tone-diverse, India-disease-mix images.
   Bootstrap with DermaCon-IN; hold out DDI (biopsy-confirmed, dark skin) for fairness
   testing; use Fitzpatrick17k with caution (noisy, fair-skin-skewed labels).
2. **Expert labelling** — your clinical ground truth; KOH/culture confirmation for
   fungal cases; capture prior-steroid-use as a field.
3. **Train + evaluate** a vision model on that data; hold out a Fitzpatrick V–VI test
   set and measure failure honestly.
4. **Swap the engine** in `server/proxy.js` from the placeholder to your model.
5. **Regulatory + clinical validation** — a tool that diagnoses serious conditions is
   CDSCO Class C/D in India. Keep the product scoped to triage until validated.

Only after step 3 holds up should you talk go-to-market. A great demo on a placeholder
engine is the most common way these projects fool their own founders.
