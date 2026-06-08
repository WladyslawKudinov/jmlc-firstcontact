# Learned Lead Scoring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible study showing whether a model trained on LLM-generated calls beats the hand-written rule scorer at ranking "who to call back first", with honest leakage/OOD/real-call validation.

**Architecture:** Pure, testable Python modules under `ml/` (config, baseline rules port, latent-variable persona generator, LLM client, dataset generation, features, metrics, train, evaluate). The LLM is isolated behind an injectable `complete_fn` so the whole pipeline is unit-testable with **no API key**. MLflow tracks runs; a notebook narrates EDA + results.

**Tech Stack:** Python 3.11, scikit-learn, scipy, numpy, pandas, matplotlib, MLflow, httpx, pytest.

**Spec:** `docs/superpowers/specs/2026-06-01-learned-lead-scoring-design.md`

**Conventions:** Execute from inside the JMLC repo (so plain `pytest`/`git` work). TDD: write the failing test, run it red, implement, run it green, commit. Seed everything (`numpy.random.default_rng`).

---

## File structure

**Create:**
- `pyproject.toml` — pytest config (`pythonpath="."`).
- `ml/__init__.py`, `tests/__init__.py` — packages.
- `ml/config.py` — paths, seed, scenario loading helpers.
- `ml/baseline.py` — Python port of prod `lib/scoring.js` (`score_lead`).
- `ml/personas.py` — latent traits, label sampling, behavior brief.
- `ml/llm.py` — `build_messages`, `parse_generation` (pure), `make_complete_fn` (httpx).
- `ml/generate.py` — `make_record`, `generate_dataset(complete_fn)`, CLI.
- `ml/features.py` — structured + TF-IDF features.
- `ml/metrics.py` — `precision_at_k`, `ndcg_at_k`, `evaluate_scores`.
- `ml/train.py` — train + persist models.
- `ml/evaluate.py` — `baseline_scores`, `compare`, leakage AUC, CLI (+ MLflow + plots).
- `tests/test_*.py` — one per module.
- `notebooks/eda.ipynb`, `docs/HANDOFF.md`.

---

## Task 1: Python project setup

**Files:** Create `pyproject.toml`, `ml/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
addopts = "-q"
testpaths = ["tests"]
```

- [ ] **Step 2: Create empty packages**

Create `ml/__init__.py` containing `# ml package` and `tests/__init__.py` containing `# tests package`.

- [ ] **Step 3: Verify the runner works**

Run: `pytest`
Expected: exits 0, `no tests ran`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml ml/__init__.py tests/__init__.py
git commit -m "chore: python project + pytest config"
```

---

## Task 2: Config & scenario loading

**Files:** Create `ml/config.py`, `tests/test_config.py`

- [ ] **Step 1: Write the failing test** (`tests/test_config.py`)

```python
from ml.config import load_scenarios, scenarios_by_id, field_ids

def test_load_scenarios_has_three_with_collects():
    scs = load_scenarios()
    ids = {s["id"] for s in scs}
    assert {"english-courses-qualify", "obscheedelo-site", "obscheedelo-tg"} <= ids
    eng = scenarios_by_id(scs)["english-courses-qualify"]
    assert field_ids(eng) == ["situation", "level", "past_attempts", "hours_per_week", "start_date"]
```

- [ ] **Step 2: Run red** — `pytest tests/test_config.py` → FAIL (no module `ml.config`).

- [ ] **Step 3: Implement `ml/config.py`**

```python
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
DATA = ROOT / "data"
RAW = DATA / "raw"
MODELS = ROOT / "models"
REPORT = ROOT / "docs" / "report"

GLOBAL_SEED = 42

def load_scenarios(path=None):
    path = Path(path) if path else (ASSETS / "scenarios.json")
    return json.loads(path.read_text(encoding="utf-8"))

def scenarios_by_id(scenarios):
    return {s["id"]: s for s in scenarios}

def field_ids(scenario):
    return [c["id"] for c in scenario.get("collects", []) if isinstance(c, dict)]
```

- [ ] **Step 4: Run green** — `pytest tests/test_config.py` → PASS.

- [ ] **Step 5: Commit**

```bash
git add ml/config.py tests/test_config.py
git commit -m "feat: scenario loading + path config"
```

---

## Task 3: Rule baseline (port of prod `lib/scoring.js`)

**Files:** Create `ml/baseline.py`, `tests/test_baseline.py`

- [ ] **Step 1: Write the failing test** (`tests/test_baseline.py`)

```python
from ml.baseline import score_lead
from ml.config import load_scenarios, scenarios_by_id

ENG = scenarios_by_id(load_scenarios())["english-courses-qualify"]

def test_empty_fields_zero():
    out = score_lead(ENG, {})
    assert out["fit"] == 0 and out["completeness"] == 0 and out["breakdown"] == []

def test_partial_completeness_plus_one_signal():
    out = score_lead(ENG, {"level": "средний", "situation": "работа"})
    assert out["completeness"] == 2          # 2/5 * 50 = 20
    assert out["fit"] == 35                    # 20 + situation(+15)
    assert any(b["label"] == "конкретный повод" and b["points"] == 15 for b in out["breakdown"])

def test_hot_lead_high_fit():
    out = score_lead(ENG, {
        "situation": "работа, клиенты", "level": "средний", "past_attempts": "курсы",
        "hours_per_week": "5", "start_date": "на этой неделе",
    })
    assert out["fit"] >= 80
    assert any(b["label"] == "старт на этой неделе" for b in out["breakdown"])

def test_atleast_needs_integer_threshold():
    out = score_lead(ENG, {"hours_per_week": "2 часа"})
    assert not any(b["label"] == "реалистичные часы" for b in out["breakdown"])
```

- [ ] **Step 2: Run red** — FAIL (no module).

- [ ] **Step 3: Implement `ml/baseline.py`** (mirrors prod scoring exactly)

```python
import re

COMPLETENESS_WEIGHT = 50
SIGNAL_CAP = 50
SIGNAL_FLOOR = -25

def _is_filled(v):
    return v is not None and str(v).strip() != ""

def _matches(signal, value):
    if not _is_filled(value):
        return False
    s = str(value)
    if isinstance(signal.get("match"), list):
        return any(re.search(p, s, re.IGNORECASE) for p in signal["match"])
    if isinstance(signal.get("atLeast"), (int, float)):
        m = re.search(r"-?\d+", s)
        return (int(m.group()) >= signal["atLeast"]) if m else False
    return False

def score_lead(scenario, fields):
    fields = fields or {}
    ids = [c["id"] for c in scenario.get("collects", []) if isinstance(c, dict)]
    total = len(ids) or 1
    filled = sum(1 for i in ids if _is_filled(fields.get(i)))
    completeness_points = round(filled / total * COMPLETENESS_WEIGHT)
    breakdown = []
    if filled > 0:
        breakdown.append({"label": f"полнота {filled}/{total}", "points": completeness_points})
    signal_sum = 0
    for sig in scenario.get("scoring", {}).get("signals", []):
        if _matches(sig, fields.get(sig["field"])):
            signal_sum += sig["points"]
            breakdown.append({"label": sig["label"], "points": sig["points"]})
    signal_points = max(SIGNAL_FLOOR, min(SIGNAL_CAP, signal_sum))
    fit = max(0, min(100, completeness_points + signal_points))
    return {"fit": fit, "completeness": filled, "breakdown": breakdown}
```

- [ ] **Step 4: Run green** — `pytest tests/test_baseline.py` → PASS (4).

- [ ] **Step 5: Commit**

```bash
git add ml/baseline.py tests/test_baseline.py
git commit -m "feat: port rule-based lead scorer from prod"
```

---

## Task 4: Latent personas (label generator, anti-leakage)

**Files:** Create `ml/personas.py`, `tests/test_personas.py`

- [ ] **Step 1: Write the failing test** (`tests/test_personas.py`)

```python
import numpy as np
from ml.personas import (
    LATENT_TRAITS, sample_latent, label_probability, sample_label, behavior_brief,
)

def test_latent_in_unit_range_and_seeded():
    a = sample_latent(np.random.default_rng(0))
    b = sample_latent(np.random.default_rng(0))
    assert set(a) == set(LATENT_TRAITS)
    assert a == b
    assert all(0.0 <= v <= 1.0 for v in a.values())

def test_label_probability_monotonic_in_readiness():
    base = {t: 0.5 for t in LATENT_TRAITS}
    hi = dict(base, readiness=0.95)
    lo = dict(base, readiness=0.05)
    assert label_probability(hi) > label_probability(lo)

def test_higher_traits_more_positive_labels():
    rng = np.random.default_rng(1)
    hi = {t: 0.85 for t in LATENT_TRAITS}
    lo = {t: 0.15 for t in LATENT_TRAITS}
    mean_hi = np.mean([sample_label(hi, rng) for _ in range(400)])
    mean_lo = np.mean([sample_label(lo, rng) for _ in range(400)])
    assert mean_hi > mean_lo + 0.2

def test_brief_leaks_no_numbers_or_label():
    brief = behavior_brief({t: 0.9 for t in LATENT_TRAITS})
    assert not any(ch.isdigit() for ch in brief)
    low = behavior_brief({t: 0.1 for t in LATENT_TRAITS})
    assert brief != low                       # behavior differs by traits
    assert "решил" in brief                    # high readiness phrase present
```

- [ ] **Step 2: Run red** — FAIL.

- [ ] **Step 3: Implement `ml/personas.py`**

```python
import numpy as np

LATENT_TRAITS = ["readiness", "urgency", "budget_fit", "program_fit", "engagement"]
TRAIT_WEIGHTS = {"readiness": 2.2, "urgency": 1.4, "budget_fit": 1.2, "program_fit": 1.0, "engagement": 0.6}
BIAS = 3.2
LABEL_NOISE_SD = 0.8

def sample_latent(rng):
    return {t: float(rng.beta(2.0, 2.0)) for t in LATENT_TRAITS}

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _logit(theta):
    return sum(TRAIT_WEIGHTS[t] * theta[t] for t in LATENT_TRAITS) - BIAS

def label_probability(theta):
    return float(_sigmoid(_logit(theta)))

def sample_label(theta, rng):
    noisy = _logit(theta) + float(rng.normal(0.0, LABEL_NOISE_SD))
    return int(rng.random() < _sigmoid(noisy))

_BANDS = {
    "readiness": ("ты просто собираешь информацию, решения пока нет",
                  "ты присматриваешься, но окончательно не решил(а)",
                  "ты уже почти решил(а) учиться, спрашиваешь про старт"),
    "urgency": ("со сроками не торопишься, «когда-нибудь потом»",
                "готов(а) начать в ближайший месяц",
                "хочешь начать как можно скорее, буквально на этой неделе"),
    "budget_fit": ("бюджет ограничен, переживаешь о цене, просишь рассрочку/скидку",
                   "цена важна, спросишь про стоимость и варианты",
                   "цена для тебя не проблема, про деньги не переживаешь"),
    "program_fit": ("твой запрос лишь частично совпадает с программой школы",
                    "твой запрос в целом подходит школе",
                    "твой запрос точно совпадает с тем, что школа предлагает"),
    "engagement": ("отвечаешь односложно, неохотно, коротко",
                   "отвечаешь по делу, нейтрально",
                   "отвечаешь развёрнуто, охотно, задаёшь встречные вопросы"),
}

def _band(v):
    return 0 if v < 0.34 else (2 if v > 0.66 else 1)

def behavior_brief(theta):
    return "\n".join("- " + _BANDS[t][_band(theta[t])] for t in LATENT_TRAITS)
```

- [ ] **Step 4: Run green** — `pytest tests/test_personas.py` → PASS (4).

- [ ] **Step 5: Commit**

```bash
git add ml/personas.py tests/test_personas.py
git commit -m "feat: latent-variable persona + label generator"
```

---

## Task 5: LLM client (messages + parsing, injectable)

**Files:** Create `ml/llm.py`, `tests/test_llm.py`

- [ ] **Step 1: Write the failing test** (`tests/test_llm.py`)

```python
from ml.llm import build_messages, parse_generation, LLMError
from ml.config import load_scenarios, scenarios_by_id, field_ids
import pytest

ENG = scenarios_by_id(load_scenarios())["english-courses-qualify"]

def test_build_messages_includes_brief_and_field_names():
    system, user = build_messages(ENG, "- ты уже почти решил(а)")
    assert system == ENG["system_prompt"]
    assert "ты уже почти решил" in user
    assert "start_date" in user

def test_parse_generation_extracts_turns_and_fields_even_with_fence():
    raw = '```json\n{"transcript":[{"role":"assistant","text":"Здравствуйте"},' \
          '{"role":"user","text":"Да"},{"role":"user","text":""}],' \
          '"fields":{"situation":"работа","level":null}}\n```'
    transcript, fields = parse_generation(raw, field_ids(ENG))
    assert transcript == [{"role": "assistant", "text": "Здравствуйте"}, {"role": "user", "text": "Да"}]
    assert fields["situation"] == "работа"
    assert fields["level"] is None and fields["start_date"] is None   # missing -> null

def test_parse_generation_raises_without_json():
    with pytest.raises(LLMError):
        parse_generation("no json here", field_ids(ENG))
```

- [ ] **Step 2: Run red** — FAIL.

- [ ] **Step 3: Implement `ml/llm.py`**

```python
import os
import json
import httpx

BASE_URLS = {"xai": "https://api.x.ai/v1", "openai": "https://api.openai.com/v1"}
KEY_ENV = {"xai": "XAI_API_KEY", "openai": "OPENAI_API_KEY"}

class LLMError(RuntimeError):
    pass

def build_messages(scenario, brief):
    fields = [c["id"] for c in scenario["collects"]]
    nulls = ", ".join(f'"{f}": null' for f in fields)
    user = (
        "Сгенерируй РЕАЛИСТИЧНЫЙ короткий телефонный диалог между ассистентом «Софья» "
        "и лидом по сценарию. Софья следует системному промпту. Лид ведёт себя так:\n"
        + brief + "\n\n"
        "Верни СТРОГО JSON без markdown:\n"
        '{"transcript":[{"role":"assistant","text":"..."},{"role":"user","text":"..."}],'
        f'"fields":{{{nulls}}}}}\n'
        f"В fields заполни значения по диалогу (поля: {', '.join(fields)}); null если не прозвучало. "
        "Ничего кроме JSON."
    )
    return scenario["system_prompt"], user

def parse_generation(raw, field_ids):
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LLMError("no JSON object found in generation")
    obj = json.loads(raw[start:end + 1])
    transcript = []
    for turn in obj.get("transcript", []) or []:
        text = str(turn.get("text", "")).strip()
        if not text:
            continue
        role = "user" if turn.get("role") == "user" else "assistant"
        transcript.append({"role": role, "text": text})
    raw_fields = obj.get("fields", {}) or {}
    fields = {}
    for fid in field_ids:
        v = raw_fields.get(fid)
        fields[fid] = None if v in ("", None) else v
    return transcript, fields

def make_complete_fn(provider=None, model=None, timeout=60.0):
    provider = provider or os.environ.get("GEN_PROVIDER", "xai")
    model = model or os.environ.get("GEN_MODEL")
    if not model:
        raise LLMError("GEN_MODEL not set")
    key = os.environ.get(KEY_ENV[provider])
    if not key:
        raise LLMError(f"{KEY_ENV[provider]} not set")
    base = BASE_URLS[provider]

    def complete(system, user, temperature=0.9):
        resp = httpx.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "temperature": temperature,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}]},
            timeout=timeout,
        )
        if resp.status_code != 200:
            raise LLMError(f"{resp.status_code}: {resp.text[:200]}")
        return resp.json()["choices"][0]["message"]["content"]

    return complete
```

- [ ] **Step 4: Run green** — `pytest tests/test_llm.py` → PASS (3).

- [ ] **Step 5: Commit**

```bash
git add ml/llm.py tests/test_llm.py
git commit -m "feat: LLM message builder + robust generation parser"
```

---

## Task 6: Dataset generation (injectable complete_fn) + CLI

**Files:** Create `ml/generate.py`, `tests/test_generate.py`

- [ ] **Step 1: Write the failing test** (`tests/test_generate.py`)

```python
import numpy as np
from ml.generate import make_record, transcript_text, generate_dataset
from ml.config import load_scenarios

SCS = load_scenarios()

def _stub_complete(system, user, temperature=0.9):
    return ('{"transcript":[{"role":"assistant","text":"Здравствуйте"},'
            '{"role":"user","text":"Мне срочно для работы, на этой неделе"}],'
            '"fields":{"situation":"работа","start_date":"на этой неделе"}}')

def test_transcript_text_flattens_with_speakers():
    txt = transcript_text([{"role": "assistant", "text": "Привет"}, {"role": "user", "text": "Да"}])
    assert "Софья: Привет" in txt and "Лид: Да" in txt

def test_make_record_shape():
    rec = make_record(SCS[0], {"readiness": 0.9}, 1, [{"role": "user", "text": "Да"}],
                      {"situation": "работа"}, "A")
    assert rec["label"] == 1 and rec["generator"] == "A"
    assert rec["scenario_id"] == SCS[0]["id"]
    assert "theta" in rec and "transcript_text" in rec

def test_generate_dataset_uses_stub_no_network():
    recs = generate_dataset(SCS, n=6, rng=np.random.default_rng(0),
                            complete_fn=_stub_complete, temperature=0.9, generator="A")
    assert len(recs) == 6
    assert all(r["label"] in (0, 1) for r in recs)
    assert all(len(r["transcript"]) >= 1 for r in recs)
    assert {r["scenario_id"] for r in recs} <= {s["id"] for s in SCS}
```

- [ ] **Step 2: Run red** — FAIL.

- [ ] **Step 3: Implement `ml/generate.py`**

```python
import argparse
import json
import uuid
import numpy as np

from .config import load_scenarios, RAW, GLOBAL_SEED, field_ids
from .personas import sample_latent, sample_label, behavior_brief
from .llm import build_messages, parse_generation, make_complete_fn

def transcript_text(transcript):
    return "\n".join(
        f"{'Лид' if t['role'] == 'user' else 'Софья'}: {t['text']}" for t in transcript
    )

def make_record(scenario, theta, label, transcript, fields, generator):
    return {
        "call_id": str(uuid.uuid4()),
        "scenario_id": scenario["id"],
        "transcript": transcript,
        "transcript_text": transcript_text(transcript),
        "fields": fields,
        "theta": theta,
        "label": int(label),
        "generator": generator,
    }

def generate_dataset(scenarios, n, rng, complete_fn, temperature, generator):
    records = []
    for i in range(n):
        scenario = scenarios[i % len(scenarios)]
        theta = sample_latent(rng)
        label = sample_label(theta, rng)
        system, user = build_messages(scenario, behavior_brief(theta))
        raw = complete_fn(system=system, user=user, temperature=temperature)
        transcript, fields = parse_generation(raw, field_ids(scenario))
        records.append(make_record(scenario, theta, label, transcript, fields, generator))
    return records

def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200, help="gen-A total (train+test)")
    ap.add_argument("--n-ood", type=int, default=300, help="gen-B (distribution shift)")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--out", default=str(RAW))
    args = ap.parse_args()

    from pathlib import Path
    out = Path(args.out)
    scenarios = load_scenarios()
    rng = np.random.default_rng(GLOBAL_SEED)

    complete_a = make_complete_fn()
    a = generate_dataset(scenarios, args.n, rng, complete_a, temperature=0.9, generator="A")
    rng.shuffle(a)
    n_test = int(len(a) * args.test_frac)
    _write_jsonl(out / "test.jsonl", a[:n_test])
    _write_jsonl(out / "train.jsonl", a[n_test:])

    import os
    ood_model = os.environ.get("GEN_MODEL_OOD") or os.environ.get("GEN_MODEL")
    ood_temp = float(os.environ.get("GEN_TEMPERATURE_OOD", "1.1"))
    complete_b = make_complete_fn(model=ood_model)
    b = generate_dataset(scenarios, args.n_ood, rng, complete_b, temperature=ood_temp, generator="B")
    _write_jsonl(out / "ood.jsonl", b)

    print(f"train={len(a) - n_test} test={n_test} ood={len(b)} -> {out}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run green** — `pytest tests/test_generate.py` → PASS (3).

- [ ] **Step 5: Commit**

```bash
git add ml/generate.py tests/test_generate.py
git commit -m "feat: synthetic dataset generation (injectable LLM)"
```

---

## Task 7: Features

**Files:** Create `ml/features.py`, `tests/test_features.py`

- [ ] **Step 1: Write the failing test** (`tests/test_features.py`)

```python
from ml.features import structured_features, make_vectorizer, featurize

REC = {
    "transcript": [{"role": "assistant", "text": "Здравствуйте"},
                   {"role": "user", "text": "Да, мне срочно для работы"}],
    "transcript_text": "Софья: Здравствуйте\nЛид: Да, мне срочно для работы",
    "fields": {"situation": "работа", "level": None, "start_date": "на этой неделе"},
}

def test_structured_features_length_and_completeness():
    f = structured_features(REC)
    assert len(f) == 5
    assert abs(f[0] - 2 / 3) < 1e-9        # 2 of 3 fields filled

def test_featurize_fit_then_transform_same_width():
    vec = make_vectorizer()
    X1 = featurize([REC, REC], vec, fit=True)
    X2 = featurize([REC], vec, fit=False)
    assert X1.shape[1] == X2.shape[1]
    assert X2.shape[0] == 1
```

- [ ] **Step 2: Run red** — FAIL.

- [ ] **Step 3: Implement `ml/features.py`**

```python
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

def structured_features(record):
    transcript = record.get("transcript", []) or []
    user_turns = [t for t in transcript if t.get("role") == "user"]
    fields = record.get("fields", {}) or {}
    n_fields = len(fields)
    filled = sum(1 for v in fields.values() if v not in (None, ""))
    completeness = (filled / n_fields) if n_fields else 0.0
    text = record.get("transcript_text", "")
    avg_user_len = float(np.mean([len(t["text"]) for t in user_turns])) if user_turns else 0.0
    return [completeness, float(len(transcript)), float(len(user_turns)),
            float(len(text)), avg_user_len]

def make_vectorizer():
    # char n-grams handle Russian morphology without a tokenizer
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=20000)

def _texts(records):
    return [r.get("transcript_text", "") for r in records]

def featurize(records, vectorizer, fit=False):
    X_text = vectorizer.fit_transform(_texts(records)) if fit else vectorizer.transform(_texts(records))
    X_struct = csr_matrix(np.array([structured_features(r) for r in records], dtype=float))
    return hstack([X_text, X_struct]).tocsr()

def labels(records):
    return np.array([int(r["label"]) for r in records])
```

- [ ] **Step 4: Run green** — `pytest tests/test_features.py` → PASS (2).

- [ ] **Step 5: Commit**

```bash
git add ml/features.py tests/test_features.py
git commit -m "feat: TF-IDF + structured feature extraction"
```

---

## Task 8: Ranking & probability metrics

**Files:** Create `ml/metrics.py`, `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test** (`tests/test_metrics.py`)

```python
from ml.metrics import precision_at_k, ndcg_at_k, evaluate_scores

def test_precision_at_k_perfect_and_worst():
    assert precision_at_k([1, 1, 0, 0], [0.9, 0.8, 0.1, 0.2], 0.5) == 1.0
    assert precision_at_k([0, 0, 1, 1], [0.9, 0.8, 0.1, 0.2], 0.5) == 0.0

def test_ndcg_perfect_is_one():
    assert abs(ndcg_at_k([1, 1, 0, 0], [0.9, 0.8, 0.1, 0.2], 1.0) - 1.0) < 1e-9

def test_evaluate_scores_keys():
    out = evaluate_scores([0, 1, 0, 1], [0.2, 0.9, 0.3, 0.8])
    assert {"auc", "ap", "precision_at_20pct", "ndcg_at_20pct"} <= set(out)
    assert out["auc"] == 1.0
```

- [ ] **Step 2: Run red** — FAIL.

- [ ] **Step 3: Implement `ml/metrics.py`**

```python
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def _topk_idx(scores, k_frac):
    n = len(scores)
    k = max(1, int(round(n * k_frac)))
    order = np.argsort(-np.asarray(scores, dtype=float))
    return order[:k], k

def precision_at_k(y_true, scores, k_frac=0.2):
    idx, k = _topk_idx(scores, k_frac)
    return float(np.asarray(y_true)[idx].sum() / k)

def ndcg_at_k(y_true, scores, k_frac=0.2):
    idx, k = _topk_idx(scores, k_frac)
    gains = np.asarray(y_true, dtype=float)[idx]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((gains * discounts).sum())
    ideal = np.sort(np.asarray(y_true, dtype=float))[::-1][:k]
    idcg = float((ideal * discounts).sum())
    return (dcg / idcg) if idcg > 0 else 0.0

def evaluate_scores(y_true, scores, k_frac=0.2):
    y_true = list(y_true)
    multiclass = len(set(y_true)) > 1
    return {
        "auc": float(roc_auc_score(y_true, scores)) if multiclass else float("nan"),
        "ap": float(average_precision_score(y_true, scores)) if multiclass else float("nan"),
        "precision_at_20pct": precision_at_k(y_true, scores, k_frac),
        "ndcg_at_20pct": ndcg_at_k(y_true, scores, k_frac),
    }
```

- [ ] **Step 4: Run green** — `pytest tests/test_metrics.py` → PASS (3).

- [ ] **Step 5: Commit**

```bash
git add ml/metrics.py tests/test_metrics.py
git commit -m "feat: precision@k / NDCG / probability metrics"
```

---

## Task 9: Training

**Files:** Create `ml/train.py`, `tests/test_train.py`

- [ ] **Step 1: Write the failing test** (`tests/test_train.py`)

```python
import numpy as np
from ml.train import train_model

def _toy(n=40, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for _ in range(n):
        pos = rng.random() < 0.5
        text = ("срочно работа на этой неделе готов начать" if pos
                else "просто смотрю подумаю потом не уверен")
        recs.append({
            "transcript": [{"role": "user", "text": text}],
            "transcript_text": "Лид: " + text,
            "fields": {"situation": "работа" if pos else None, "start_date": None},
            "label": int(pos),
        })
    return recs

def test_train_model_returns_fitted_estimator_and_vectorizer():
    model, vec = train_model(_toy())
    from ml.features import featurize
    X = featurize(_toy(n=8, seed=9), vec, fit=False)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (8,)
    assert ((proba >= 0) & (proba <= 1)).all()
```

- [ ] **Step 2: Run red** — FAIL.

- [ ] **Step 3: Implement `ml/train.py`**

```python
import argparse
import json
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression

from .config import MODELS, RAW
from .features import featurize, labels, make_vectorizer

def train_model(records):
    vectorizer = make_vectorizer()
    X = featurize(records, vectorizer, fit=True)
    y = labels(records)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X, y)
    return model, vectorizer

def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(RAW / "train.jsonl"))
    ap.add_argument("--out", default=str(MODELS))
    args = ap.parse_args()
    records = _read_jsonl(args.data)
    model, vectorizer = train_model(records)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    joblib.dump(vectorizer, out / "vectorizer.joblib")
    print(f"trained on {len(records)} -> {out}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run green** — `pytest tests/test_train.py` → PASS.

- [ ] **Step 5: Commit**

```bash
git add ml/train.py tests/test_train.py
git commit -m "feat: train + persist logistic-regression scorer"
```

---

## Task 10: Evaluation — model vs baseline, leakage, OOD (+ CLI/MLflow/plots)

**Files:** Create `ml/evaluate.py`, `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing test** (`tests/test_evaluate.py`)

```python
import numpy as np
from ml.evaluate import baseline_scores, compare
from ml.train import train_model
from ml.config import load_scenarios

SCS_BY_ID = {s["id"]: s for s in load_scenarios()}

def _toy(n=40, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for _ in range(n):
        pos = rng.random() < 0.5
        text = "срочно работа на этой неделе" if pos else "просто смотрю потом"
        recs.append({
            "scenario_id": "english-courses-qualify",
            "transcript": [{"role": "user", "text": text}],
            "transcript_text": "Лид: " + text,
            "fields": {"situation": "работа" if pos else None,
                       "start_date": "на этой неделе" if pos else None},
            "label": int(pos),
        })
    return recs

def test_baseline_scores_align_with_labels():
    recs = _toy()
    bs = baseline_scores(recs, SCS_BY_ID)
    assert len(bs) == len(recs)
    # positives carry hot signals -> higher mean baseline fit
    y = np.array([r["label"] for r in recs])
    assert bs[y == 1].mean() > bs[y == 0].mean()

def test_compare_returns_model_and_baseline_metrics():
    train = _toy(seed=1)
    test = _toy(seed=2)
    model, vec = train_model(train)
    out = compare(test, model, vec, SCS_BY_ID)
    assert "model" in out and "baseline" in out
    assert "auc" in out["model"] and "precision_at_20pct" in out["baseline"]
```

- [ ] **Step 2: Run red** — FAIL.

- [ ] **Step 3: Implement `ml/evaluate.py`**

```python
import argparse
import json
from pathlib import Path

import numpy as np

from .baseline import score_lead
from .config import MODELS, RAW, REPORT, load_scenarios, scenarios_by_id
from .features import featurize
from .metrics import evaluate_scores

def baseline_scores(records, scs_by_id):
    return np.array([score_lead(scs_by_id[r["scenario_id"]], r.get("fields", {}))["fit"]
                     for r in records], dtype=float)

def model_scores(records, model, vectorizer):
    X = featurize(records, vectorizer, fit=False)
    return model.predict_proba(X)[:, 1]

def compare(records, model, vectorizer, scs_by_id):
    y = [int(r["label"]) for r in records]
    return {
        "n": len(records),
        "model": evaluate_scores(y, model_scores(records, model, vectorizer)),
        "baseline": evaluate_scores(y, baseline_scores(records, scs_by_id)),
    }

def leakage_auc(records):
    """A deliberately dumb word-count model. Should be well below ~1.0 if no leak."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    texts = [r["transcript_text"] for r in records]
    y = np.array([int(r["label"]) for r in records])
    X = CountVectorizer().fit_transform(texts)
    proba = cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=5, method="predict_proba")[:, 1]
    return float(evaluate_scores(y, proba)["auc"])

def _read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

def main():
    import joblib
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(MODELS))
    ap.add_argument("--data", default=str(RAW / "test.jsonl"))
    ap.add_argument("--ood", default=str(RAW / "ood.jsonl"))
    ap.add_argument("--real", default=str(Path("data/real/real.jsonl")))
    args = ap.parse_args()

    scs_by_id = scenarios_by_id(load_scenarios())
    model = joblib.load(Path(args.models) / "model.joblib")
    vec = joblib.load(Path(args.models) / "vectorizer.joblib")

    results = {"test": compare(_read_jsonl(args.data), model, vec, scs_by_id)}
    for name, path in [("ood", args.ood), ("real", args.real)]:
        recs = _read_jsonl(path)
        if recs:
            results[name] = compare(recs, model, vec, scs_by_id)
    results["leakage_auc_dumb_model"] = leakage_auc(_read_jsonl(args.data))

    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))

    # MLflow (best-effort; never fail the run on tracking issues)
    try:
        import mlflow
        mlflow.set_tracking_uri(f"file://{Path('mlruns').resolve()}")
        with mlflow.start_run():
            for split, blk in results.items():
                if isinstance(blk, dict) and "model" in blk:
                    for k, v in blk["model"].items():
                        mlflow.log_metric(f"{split}_model_{k}", v)
                    for k, v in blk["baseline"].items():
                        mlflow.log_metric(f"{split}_baseline_{k}", v)
            mlflow.log_metric("leakage_auc", results["leakage_auc_dumb_model"])
            mlflow.log_artifact(str(REPORT / "metrics.json"))
    except Exception as e:  # noqa: BLE001
        print(f"[mlflow skipped] {e}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run green** — `pytest tests/test_evaluate.py` → PASS (2).

- [ ] **Step 5: Full suite green** — `pytest` → all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add ml/evaluate.py tests/test_evaluate.py
git commit -m "feat: evaluate model vs baseline + leakage/OOD/real harness"
```

---

## Task 11: End-to-end run, EDA notebook, real-call import, handoff doc

**Files:** Create `notebooks/eda.ipynb`, `docs/HANDOFF.md`; Modify `README.md`

- [ ] **Step 1: Generate a small real dataset (no API key needed)**

Smoke the whole pipeline offline first:

```bash
python -c "import numpy as np, json; from ml.generate import generate_dataset; from ml.config import load_scenarios; \
recs = generate_dataset(load_scenarios(), 12, np.random.default_rng(0), \
 lambda system,user,temperature=0.9: '{\"transcript\":[{\"role\":\"user\",\"text\":\"срочно для работы на этой неделе\"}],\"fields\":{\"situation\":\"работа\",\"start_date\":\"на этой неделе\"}}', 0.9, 'A'); \
print(len(recs), recs[0]['label'])"
```
Expected: prints `12 0` or `12 1` (no network used).

- [ ] **Step 2: Real generation (needs an LLM key)**

```bash
cp .env.example .env            # set GEN_PROVIDER, GEN_MODEL, and the matching key
set -a; source .env; set +a
python -m ml.generate --n 1200 --n-ood 300
```
Expected: `train=900 test=300 ood=300 -> .../data/raw`. (Debug with `--n 60 --n-ood 15` first.)

- [ ] **Step 3: Train + evaluate**

```bash
python -m ml.train --data data/raw/train.jsonl
python -m ml.evaluate --data data/raw/test.jsonl
```
Expected: a metrics block where **`leakage_auc_dumb_model` is clearly < 1.0** (e.g. < 0.9 — if ~1.0, increase `LABEL_NOISE_SD` / soften `behavior_brief` and regenerate), and `model` ≥ `baseline` on `precision_at_20pct` / `auc`.

- [ ] **Step 4: Real-call set (honest sim-to-real bridge)**

Collect ~20–50 real calls via the prod app, export the CRM JSON, and convert to
`data/real/real.jsonl` (same row shape: `scenario_id`, `transcript`,
`transcript_text`, `fields`, hand-labeled `label`). Re-run Step 3 — the `real`
block now appears. Report its **true N** and labeling method; never inflate it.

- [ ] **Step 5: EDA + report notebook**

Create `notebooks/eda.ipynb` covering: class balance; transcript-length and
completeness distributions by label; baseline-fit separation by label; the
results table (model vs baseline on test/OOD/real); reliability curve; and a
**Limitations** cell (synthetic labels; small real N). Render it:

```bash
make report
```
Expected: `docs/report/eda.html` exists.

- [ ] **Step 6: Write `docs/HANDOFF.md`**

```markdown
# Handoff — start here

**State:** spec + plan committed; pipeline implemented & unit-tested (`pytest`).
**Next:** run Task 11 (generate → train → evaluate), collect the real-call set,
fill the notebook, write the report.

- Spec: docs/superpowers/specs/2026-06-01-learned-lead-scoring-design.md
- Plan: docs/superpowers/plans/2026-06-01-learned-lead-scoring.md
- Decisions: binary `enrolled` label from latent traits + noise; baseline = ported
  rule scorer; primary metric precision@k; MLflow local; integrity rule in CLAUDE.md.
- Open knobs: GEN_PROVIDER/GEN_MODEL; embeddings ablation (requirements-embeddings.txt) is stretch.
```

- [ ] **Step 7: Commit**

```bash
git add notebooks/eda.ipynb docs/HANDOFF.md README.md docs/report/metrics.json
git commit -m "docs: EDA/report notebook, results, and handoff"
```

---

## Self-review (done while writing)

- **Spec coverage:** synthetic generator+latent design (T4,T6) · baseline (T3) ·
  features (T7) · models (T9) · metrics incl. precision@k/calibration-inputs (T8) ·
  leakage+OOD+real validation (T10,T11) · MLOps Docker/CI/MLflow (scaffold + T10) ·
  EDA/report + limitations (T11) · product/impact lives in the report (T11/notebook). ✓
- **No placeholders:** every code/test step has complete code and exact commands. ✓
- **Type consistency:** record schema (`scenario_id, transcript, transcript_text, fields, theta, label, generator`) and signatures (`featurize(records, vectorizer, fit)`, `compare(records, model, vectorizer, scs_by_id)`) are consistent across T6–T11. ✓
- Note: GBM/embedding ablations and the LLM zero-shot baseline are **stretch** (spec §11) — add as T12+ if time allows; the MVP entry is complete without them.
