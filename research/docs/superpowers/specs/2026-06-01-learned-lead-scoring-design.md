---
title: Learned lead scoring vs rules — bootstrapped from synthetic calls
status: draft
created: 2026-06-01
competition: JMLC (ITMO master's admission)
repo: JMLC (research/DS exhibit)
related: FirstContact prod service (product/engineering exhibit)
---

# Learned lead scoring — design

## 1. Problem & research question

The *FirstContact* voice agent qualifies inbound edtech leads and, today, scores
each call's quality with a **hand-written rule scorer** (`lib/scoring.js`:
keyword/regex signals with hand-guessed weights). A sales team can only call back
a fraction of leads, so the operational question is a **ranking** one:

> Of everyone who called, **who do we call back first?**

**Research question.** Can a model that *learns* lead quality from data beat the
hand-written rules at this ranking — and, because we must bootstrap from
**LLM-generated** calls (no real data exists yet), **how do we know the learned
signal is real and not an artifact of the generator?**

The second clause is the actual contribution. This is a **cold-start** study: a
reusable pipeline to bootstrap and *stress-test* a lead scorer before any real
data exists, with an honest sim-to-real bridge.

## 2. ML task

- **Input:** one call = transcript (turns) + captured qualification fields.
- **Target:** `enrolled ∈ {0,1}` — would this lead convert. Derived from latent
  traits with noise (§4), so it is a genuine prediction problem with irreducible
  error, not a deterministic function of the text.
- **Prediction:** `P(enrolled) ∈ [0,1]` — used as the call-back priority score.
- **Primary framing:** **ranking** (who to call first), evaluated with
  precision@k / NDCG; secondary framing: probability quality (AUC, calibration).

## 3. Baselines (what the model must beat)

1. **Rule baseline** — faithful Python port of prod `lib/scoring.js`
   (`ml/baseline.py`), parity-tested against fixtures. This is the honest,
   non-strawman baseline (real shipped code).
2. **LLM zero-shot scorer** *(stretch)* — prompt an LLM to rate the lead 0–100.
   A strong, modern baseline.

## 4. Synthetic data — the core method (anti-leakage by design)

We generate the **label in code first**, then have an LLM **role-play the persona
blind to the label**. Per synthetic lead:

1. **Sample latent traits** θ (plain Python RNG), e.g. `readiness, urgency,
   budget_fit, program_fit, engagement` ∈ [0,1], optionally correlated.
2. **Compute the label by a formula we control, with noise:**
   `p = sigmoid(w · θ − b + ε)`, `enrolled ~ Bernoulli(p)`. The noise ε gives
   genuine irreducible error (look-alike leads can differ).
3. **Render a behavior brief in natural language — no numbers, no label**
   (e.g. high readiness → "already decided, asks about start dates"; low
   budget_fit → "anxious about price, asks for installments").
4. **LLM generates the dialogue** (София persona from `assets/scenarios.json` +
   the behavior brief). The model is **never** shown θ or the label.
5. **Extract fields** from the transcript exactly as the live product does.
6. **Store** `{ transcript, fields, scenario_id, theta (held out for analysis
   only), enrolled }` as JSONL.

**Why this is rigorous:** the label depends on θ with noise; the transcript
reflects θ only through stochastic behavior/language; θ and the label are never
fed to the generator's surface → predicting the label cannot be generator
inversion. θ is retained **only** for analysis (e.g. "does the model recover
urgency?") and **never** used as a model feature.

**Datasets produced:**
- `train.jsonl` (~1500) + `test.jsonl` held-out split (same generator = gen-A).
- `ood.jsonl` (~300–500) from **gen-B**: different wording style / temperature
  (and, if available, a different model). Same latent→label process, different
  *surface*. Tests generalization beyond the generator's accent.

## 5. Honest validation layers

1. **Leakage guard (automated, a test):** a deliberately dumb keyword/bag-of-words
   model must **not** approach perfect on held-out data. If it does, the answer
   leaked into obvious tokens → make personas subtler / increase noise. Enforced
   in `tests/`.
2. **Distribution-shift test:** evaluate the trained model on `ood.jsonl`. A large
   drop ⇒ it memorized gen-A's style, not signal.
3. **Reality check (real calls):** ~20–50 **genuinely collected** calls via the
   live app (author + friends; auto-captured by the prod CRM, exported to
   `data/real/`). Hand-labeled "would realistically enroll?" (or caller
   self-rates at the end). Reported with its **true size + labeling method**.

> **Integrity:** synthetic is always labeled synthetic; generated calls are never
> reported as real. The small real set is the honest sim-to-real bridge, and its
> limitations are stated out loud. (Mirrored in `CLAUDE.md`.)

## 6. Features & models

- **Text features:** TF-IDF over transcript (word + char n-grams, for Russian
  morphology). **Optional/stretch:** multilingual sentence embeddings (LaBSE /
  paraphrase-multilingual-MiniLM) as an ablation.
- **Structured features:** field completeness, #turns, transcript length,
  duration, simple per-field presence flags.
- **Models (sklearn):** Logistic Regression and Gradient Boosting
  (HistGradientBoosting). Compare; keep the pipeline simple and reproducible.

## 7. Experiments & metrics

- **Split & CV:** stratified train/test + 5-fold CV on train.
- **Classification:** ROC-AUC, PR-AUC, confusion matrix at a chosen threshold.
- **Ranking (= the product goal):** precision@k and NDCG@k, with k as a realistic
  call-back budget (e.g. top 20%). Rule baseline ranked on the *same* metric.
- **Calibration:** Brier score + reliability curve (the score must be a usable
  probability, not just a rank).
- **Robustness:** all metrics re-reported on `ood.jsonl` and on `data/real/`.
- **Error analysis:** where the model disagrees with the rules; FP/FN exemplars;
  feature importance; (stretch) recovery of latent θ.
- **Ablations (stretch):** TF-IDF vs embeddings; text-only vs +structured;
  learning curve (metric vs train-set size — speaks directly to "how much
  synthetic data is worth generating").

## 8. Engineering & MLOps (criterion 1)

- **Reproducible pipeline:** `make data | train | eval | report`; deterministic
  seeds; config in one place.
- **Experiment tracking:** MLflow (local file store) logs params, metrics, and
  artifacts (plots) per run.
- **Containerization:** `Dockerfile` for an identical run environment.
- **CI:** GitHub Actions runs `pytest` on every push (+ a tiny smoke generation
  using a mocked LLM client, so CI needs no API key).
- **Conscious use of ready-made solutions:** we *buy* speech/LLM (hosted) rather
  than *build* it, and document the trade-off; sklearn over bespoke models.

## 9. AI application (criterion 3)

- **LLM as data generator** (the bootstrap engine).
- **LLM as a zero-shot baseline** scorer (stretch).
- **The product itself** is an AI voice agent.
- **Agentic development:** built with Claude Code under the superpowers
  spec→plan→TDD workflow — documented as evidence.

## 10. Product thinking (criterion 4)

- **Problem & audience:** edtech schools lose their best leads because callbacks
  are slow and untriaged; audience = admissions/sales teams.
- **Competitors:** comparison table (Bland, Air.ai, Synthflow, Retell, Vapi; RU:
  Dasha.ai, Voximplant) — positioning of a *qualify-and-triage* agent.
- **Hypothesis:** better ranking → more enrollments per advisor-hour.
- **MVP:** the working voice agent + CRM (prod repo).
- **Impact:** estimated directly from precision@k — "with a callback budget of
  X%, the model captures Y% more of the true enrollers than the rules."
- **Feedback:** the real-call sprint doubles as first user feedback.

## 11. Scope (5–6 day sprint)

**MVP (must-have — a complete, defensible entry even if days slip):**
synthetic generator (latent design) + EDA; rule baseline ported & parity-tested;
one learned model; eval = AUC + precision@k + calibration + 5-fold CV vs baseline;
leakage guard; OOD test; ~20+ real calls; report narrative; Docker + CI + MLflow;
competitor table + impact estimate; **limitations** section.

**Stretch:** LLM zero-shot baseline; embeddings ablation; learning curve; SHAP;
latent-θ recovery; live model served back into the CRM (FastAPI behind a flag).

**Day-by-day (indicative):**
| Day | Focus |
|---|---|
| 1 | Generator (latent design) + dataset + EDA |
| 2 | Port baseline to Python + features + first model + metric harness |
| 3 | Full experiments (CV, calibration, ranking) + leakage/OOD + error analysis |
| 4 | MLOps (Docker, CI, MLflow) + real-call sprint (parallel) + stretch |
| 5 | Report notebook + product (competitors/impact) + figures + limitations |
| 6 | Buffer, polish, presentation, defense Q&A dry-run |

## 12. Out of scope (YAGNI)

Real enrollment-outcome tracking (impossible without live conversions);
multi-tenant; deep-learning training from scratch; production model serving beyond
the optional flag; hyperparameter mega-search.

## 13. Risks & mitigations

- **"You just modeled your own generator."** → leakage guard + OOD (gen-B) + real
  calls + explicit limitations. This *is* the thesis.
- **Real-call set too small.** → report true N proudly; frame as next step; lean on
  OOD for breadth.
- **Time overrun.** → MVP/stretch split front-loads the gradable core.
- **LLM cost/rate limits.** → debug on ~300 samples; cache generations to JSONL;
  mock the client in tests/CI.

## 14. Repo (already scaffolded)

`ml/` · `notebooks/` · `tests/` · `docs/superpowers/{specs,plans}` · `docs/report`
· `docs/issues` · `assets/scenarios.json` · `data/` (gitignored). Standalone git
repo, separate from prod; the report links to prod as the product/engineering
exhibit.

## 15. JMLC criteria coverage (summary)

| Criterion | Where it's earned |
|---|---|
| Dev & engineering | reproducible pipeline, Docker, CI, MLflow, tests, conscious build-vs-buy |
| Data Science | EDA, preprocessing, model choice, metrics (AUC/precision@k/calibration), CV + OOD + real validation |
| AI application | LLM data generation, LLM zero-shot baseline, AI voice-agent product, agentic dev |
| Product thinking | problem/audience, competitor table, hypothesis, MVP, precision@k impact, real-call feedback |
