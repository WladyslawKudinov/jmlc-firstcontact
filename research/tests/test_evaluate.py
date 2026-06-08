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


def test_compare_real_uses_verdict_as_baseline():
    from ml.evaluate import compare_real
    model, vec = train_model(_toy(seed=1))
    real = [
        {"scenario_id": "obscheedelo-site", "fields": {}, "label": 1, "verdict": "горячий",
         "transcript": [{"role": "user", "text": "да, призёрство, на этой неделе"}],
         "transcript_text": "Лид: да, призёрство, на этой неделе"},
        {"scenario_id": "obscheedelo-site", "fields": {}, "label": 0, "verdict": "холодный",
         "transcript": [{"role": "user", "text": "просто смотрю, потом"}],
         "transcript_text": "Лид: просто смотрю, потом"},
    ]
    out = compare_real(real, model, vec)
    assert out["n"] == 2
    assert "model" in out and "baseline_verdict" in out
    assert "auc" in out["baseline_verdict"]


def test_compare_real_adds_rule_baseline_when_scenarios_given():
    from ml.evaluate import compare_real
    model, vec = train_model(_toy(seed=1))
    real = [
        {"scenario_id": "obscheedelo-site", "label": 1, "verdict": "горячий",
         "fields": {"grade": "10 класс", "goal": "поступление через БВИ", "start_date": "на этой неделе",
                    "olympiad_experience": "муниципальный этап", "subject": "история"},
         "transcript": [{"role": "user", "text": "поступление, на этой неделе"}],
         "transcript_text": "Лид: поступление, на этой неделе"},
        {"scenario_id": "obscheedelo-site", "label": 0, "verdict": "холодный", "fields": {},
         "transcript": [{"role": "user", "text": "просто смотрю"}],
         "transcript_text": "Лид: просто смотрю"},
    ]
    out = compare_real(real, model, vec, SCS_BY_ID)
    assert "baseline_rules" in out and "auc" in out["baseline_rules"]
