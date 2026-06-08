# JMLC — project notes for Claude

## What this is

Research / Data-Science entry for the **JMLC** competition (ITMO master's
admission). It builds a **learned lead-qualification scorer** for the
*FirstContact* voice agent and asks: *can a model trained on LLM-generated calls
beat the hand-written rule scorer at ranking "who to call back first" — and how
do we know the signal is real?*

This repo is the **research exhibit**. The **product / engineering exhibit** is
the separate *FirstContact* prod repo (the voice agent + CRM). Link to it; do not
duplicate it here.

## Integrity (non-negotiable)

Synthetic data is **always** labeled synthetic. We **never** present generated
calls as real. The "real-call" set is genuinely collected (the author + friends
via the live app) and reported with its true (small) size and labeling method.
The honesty of the data story **is** the project's thesis — do not undermine it,
even under time pressure.

## Research log — keep it current (non-optional)

[`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md) is the **running journal** of this
research: the thesis *spine*, the chronological *вопрос → что сделали → что нашли →
решение/почему* arc, open threads, and the **presentation skeleton**. We assemble the
deck from it.

**You MUST keep it current.** Whenever a step, finding, decision, or change of direction
lands — including mid-discussion, as we go deeper — append/revise the relevant entry **in
the same turn**, and update the *spine* + key-result table if they moved, **before**
treating the work as done. A stale log is a broken deliverable. Keep `REPORT.md` (polished
result) and `HANDOFF.md` (state) consistent with it.

## Stack

- **Python 3.11**, pandas / scikit-learn, scipy, matplotlib.
- **MLflow** (local file tracking) for experiments.
- **LLM** (xAI/Grok, or OpenAI/Anthropic) for synthetic-call generation and a
  zero-shot baseline scorer.
- Multilingual sentence embeddings are **optional** (`requirements-embeddings.txt`,
  pulls in torch); the TF-IDF path needs only `requirements.txt`.

## How to run

```bash
make setup     # pip install -r requirements.txt
make data      # generate synthetic dataset -> data/raw/
make train     # train models -> models/
make eval      # model vs rule baseline -> metrics + plots
make test      # pytest
make report    # render docs/report/eda.html
```

## Structure

- `ml/` — `generate.py` (dice -> brief -> LLM -> row), `personas.py` (latent
  traits + behavior brief), `baseline.py` (Python port of prod `lib/scoring.js`),
  `features.py`, `train.py`, `evaluate.py`.
- `notebooks/` — EDA + report narrative.
- `tests/` — pytest: generator sanity, baseline parity, metrics, leakage guard.
- **`docs/RESEARCH_LOG.md`** — running research journal (keep current; see above).
- `docs/superpowers/specs/` · `docs/superpowers/plans/` · `docs/report/` · `docs/issues/`.
- `assets/scenarios.json` — persona snapshot copied from the prod repo.
- `data/` — generated + real sets (gitignored; small curated sample under `data/sample/`).

## Workflow guardrails (superpowers)

- **Specs** → `docs/superpowers/specs/` ; **plans** → `docs/superpowers/plans/`
  (brainstorming / writing-plans defaults).
- **TDD** with `pytest`. **Verification-before-completion:** run `make test` and
  show the output before claiming anything passes.
- Execute plans **task-by-task** (executing-plans / subagent-driven-development).
- Bugs / behaviour issues: one file per issue in `docs/issues/` (same pattern as
  prod), checked before re-fixing.

## Data provenance

- `assets/scenarios.json` — snapshot from the prod repo (persona definitions that
  seed the generator).
- `ml/baseline.py` — a faithful Python port of prod `lib/scoring.js` (the rule
  baseline we compare against).
- `data/real/` — real-call transcripts exported from the prod CRM's
  export-JSON endpoint.

## Git

Standalone repo, **separate from the prod service** (which has its own
`origin`/`public` remotes — unrelated to this one). Commit/push only when asked.
