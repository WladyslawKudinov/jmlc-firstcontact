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
