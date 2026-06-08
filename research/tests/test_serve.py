"""Serving stub: the `rules` path is offline; the `llm` path takes an injected complete_fn."""
import pytest

from ml.serve import priority_score

SCEN = {
    "collects": [{"id": "grade", "label": "Класс"}],
    "scoring": {"signals": [{"field": "grade", "match": [r"\b10\b"], "points": 7, "label": "10-11"}]},
}


def test_rules_scorer_is_offline_and_in_unit_interval():
    p = priority_score(SCEN, {"grade": "10"}, scorer="rules")
    assert 0.0 <= p <= 1.0
    assert p > 0  # a filled field + a matched signal -> positive priority


def test_rules_scorer_empty_lead_is_zero():
    assert priority_score(SCEN, {}, scorer="rules") == 0.0


def test_llm_scorer_uses_injected_complete_fn():
    p = priority_score(
        SCEN, {"grade": "10"}, transcript=[{"role": "user", "text": "я в 10"}],
        scorer="llm", complete=lambda system, user, temp: '{"fit": 90, "verdict": "горячий"}',
    )
    assert abs(p - 0.90) < 1e-9


def test_unknown_scorer_raises():
    with pytest.raises(ValueError):
        priority_score(SCEN, {}, scorer="bogus")
