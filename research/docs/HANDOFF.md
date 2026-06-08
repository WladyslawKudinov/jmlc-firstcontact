# Handoff — start here

You're in the **JMLC research repo**. `CLAUDE.md` (auto-loaded) has the
conventions + the **integrity rule** (synthetic is always labeled synthetic).

## Current state (2026-06-01)

- ✅ Repo scaffolded; superpowers conventions wired (specs/plans, pytest, Docker, CI).
- ✅ Design **spec** written + committed.
- ✅ Full TDD **build plan** written + committed.
- ✅ **Pipeline implemented & unit-tested** — `ml/{config,baseline,personas,llm,generate,features,metrics,train,evaluate}.py`, `make test` → **23 passing** (TDD, one module per task).
- ✅ **Report notebook** (`notebooks/eda.ipynb`) renders end-to-end via `make report` →
  `docs/report/eda.html` (gitignored build artifact).
- ⬜ **Real LLM generation** not yet run (needs an API key) — notebook currently falls
  back to **OFFLINE-SMOKE** stub data (clearly labeled; illustrative only).
- ✅ **Real-call set collected (N=31)** + integrated → `data/real/real.jsonl`, `make eval` `real`
  block. **The simulation-trained model does NOT transfer** (AUC 0.39; beaten by the rules 0.66, the
  prod verdict 0.75, and the **prod LLM scorer 0.80**; rules hold ~0.65 on sim *and* real) — the
  honest sim-to-real gap. **Phase 5:** rules on **LLM-extracted** fields reach **0.76**, so the gap
  to the LLM is mostly *extraction*, not the algorithm. See `docs/report/REPORT.md` §6.5–6.6.

## What this is

Learned lead-scoring vs the hand-written rule baseline, bootstrapped from
LLM-generated calls. **Research question:** can the learned model rank "who to
call back first" better than the rules — and how do we know the signal is real,
not a generator artifact?

## Next action

The build plan is **executed** (Tasks 1–10 + the offline report in Task 11). The
remaining work needs resources Claude can't supply autonomously:

1. **Real LLM generation** — set a key + model and run the real study:
   ```bash
   cp .env.example .env        # set GEN_PROVIDER, GEN_MODEL (+ GEN_MODEL_OOD for a
   set -a; source .env; set +a #   model-level OOD shift) + the matching key
   make data                  # resumable: safe to Ctrl-C / re-run (tops up to n).
                              #   force fresh: python -m ml.generate --no-resume
   make train && make eval    # -> docs/report/metrics.json (check leakage_auc < 1.0)
   make report                # re-render the notebook against the real LLM data
   ```
   The generator retries transient 429/5xx errors with backoff, writes each call
   incrementally (a crash never loses progress), skips an unparseable response, and
   aborts a systemically-broken run (bad model/key) instead of looping forever.
2. **Real-call set** — ✅ done (N=31 at `data/real/real.jsonl`, label = real `sale_result`;
   `make eval` `real` block; fields hand-extracted → `data/real/real_fields.json`, so both the
   prod `verdict` and the exact `lib/scoring.js` baselines run). Model fails to transfer; both
   rules beat it (§6.5). **Remaining:** grow N beyond 31 (wide CIs).
3. **Stretch** (spec §11): LLM zero-shot baseline, embeddings ablation, learning
   curve, SHAP, latent-θ recovery, live model served back into the CRM.

## Key references

- Spec: `docs/superpowers/specs/2026-06-01-learned-lead-scoring-design.md`
- Plan: `docs/superpowers/plans/2026-06-01-learned-lead-scoring.md`
- Persona + baseline source: `assets/scenarios.json` (snapshot from prod)

## Locked decisions

- Binary `enrolled` label from latent traits + noise — label made **in code**, the
  LLM role-plays the persona **blind** (anti-leakage).
- Baseline = Python port of prod `lib/scoring.js`.
- Primary metric = precision@k / NDCG (who to call first) + AUC + calibration.
- Validation = leakage check + OOD (gen-B) + a small **real** call set (true N).
- MLflow local; Docker + GitHub Actions CI.

## Open knobs

- `GEN_PROVIDER` / `GEN_MODEL` (defaults to xAI/Grok).
- Stretch: GBM + embeddings ablation, LLM zero-shot baseline, live model in the CRM.

## Time box

~5–6 days. The MVP (gradable core) is front-loaded; stretch items are bonus.
