# JMLC — Learned lead scoring for a voice agent

**Research question:** Can a model trained on **LLM-generated** sales calls beat
hand-written rules at ranking *"which lead do we call back first?"* — and how do
we know the signal is real and not an artifact of the generator?

This is the **research / Data-Science exhibit** for the JMLC entry. The
**product / engineering exhibit** is the separate *FirstContact* voice-agent +
CRM service (linked below); this repo does not duplicate it.

## The experiment in one picture

A voice agent talks to ~200 prospects; the team can only call back ~40. *Which 40?*

- **Baseline:** the hand-written rule scorer shipped in the prod service
  (`lib/scoring.js`), ported here to Python.
- **Model:** a classifier trained to output `P(enrolls)` from the transcript +
  captured fields.
- **Metric = the product goal:** of the people who *actually* would enroll, how
  many land in each method's top-k? (precision@k / NDCG, plus AUC and calibration.)

## The honest part (the actual contribution)

We have no real calls yet — a classic **cold-start** problem. We bootstrap with
synthetic data, then *prove it isn't circular*:

1. Labels are generated **in code** from hidden latent traits (with noise), then
   an LLM role-plays the persona **blind** to the label — so predicting the label
   is a real task, not generator-inversion.
2. **Leakage check** — a deliberately dumb keyword model must *not* score ~perfect.
3. **Distribution-shift test** — evaluate on calls written in a different style.
4. **Reality check** — a small set of genuinely real calls (collected via the live
   app), reported with its true size and labeling method.

> **Integrity:** synthetic data is always labeled synthetic. Generated calls are
> never presented as real. See `CLAUDE.md`.

## Results

The pipeline is implemented and unit-tested (`make test` → 23 passing). The report
notebook (`notebooks/eda.ipynb`, rendered to `docs/report/eda.html` via
`make report`) runs the full machinery — features → model → leakage guard → OOD →
calibration — end to end.

By default it runs in **OFFLINE-SMOKE** mode: a deterministic Python stub role-plays
each lead (no LLM, no network), so every number is **illustrative scaffolding, not a
result**. In that mode the learned model beats the rule baseline in-distribution on
the product metric (`precision@20%` ≈ 0.72 vs 0.62) and generalizes to the
distribution-shift (gen-B) split, while the leakage guard stays well under 1.0 — but
the absolute AUCs sit near a ~0.65 ceiling that's baked in by the generator's
deliberate label noise. **To produce the real study**, set an LLM key (below) and run
`make data && make train && make eval`; the notebook then renders against the
LLM-generated calls, and the `real`-call row appears once `data/real/real.jsonl` is
collected.

## Quickstart

```bash
make setup                 # install deps
make report                # render the OFFLINE-SMOKE report (no key needed)
make test                  # run the test suite (23 tests)

# --- the real study (needs an LLM key) ---
cp .env.example .env        # set GEN_PROVIDER, GEN_MODEL + the matching key
set -a; source .env; set +a
make data                  # LLM-generated gen-A + gen-B  -> data/raw/
make train && make eval    # train + compare vs the rule baseline + leakage/OOD
```

## Layout

```
ml/         data generation, baseline, features, train, evaluate
notebooks/  EDA + the report narrative
tests/      pytest (generator sanity, baseline parity, metrics)
docs/       superpowers/specs · superpowers/plans · report · issues
assets/     scenarios.json (persona snapshot from the prod service)
data/        generated + real sets (gitignored; small sample tracked)
```

## Related

- **FirstContact** (product/engineering exhibit) — the Grok-powered voice agent +
  mini-CRM this study builds on. _(link)_
