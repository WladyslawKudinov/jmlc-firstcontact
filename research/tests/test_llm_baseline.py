"""Phase-5 LLM-baseline module: pure builders/parsers + run() with an injected
complete_fn (no API key needed)."""
from ml.llm_baseline import (
    build_assess_messages,
    build_extract_messages,
    parse_assess,
    parse_extract,
    run,
)


def test_parse_assess_and_extract():
    assert parse_assess('{"fit": 80, "verdict": "горячий"}') == (80, "горячий")
    fields, fit = parse_extract('prefix {"fields": {"grade": "10", "goal": null}, "fit": 70} suffix')
    assert fields == {"grade": "10"}  # null/empty dropped
    assert fit == 70


def test_assess_fit_is_clamped():
    assert parse_assess('{"fit": 250, "verdict": "горячий"}')[0] == 100
    assert parse_assess('{"fit": -5, "verdict": "холодный"}')[0] == 0


def test_build_messages_carry_rubric_and_transcript():
    sc = {"school": "Общее Дело", "name": "OD", "collects": [{"id": "grade", "label": "Класс"}]}
    tr = [{"role": "assistant", "text": "привет"}, {"role": "user", "text": "я в 10 классе"}]
    sys_a, user_a = build_assess_messages(sc, {"grade": "10"}, tr)
    assert "Общее Дело" in sys_a and "verdict" in sys_a
    assert "я в 10 классе" in user_a and "- grade: 10" in user_a
    sys_e, user_e = build_extract_messages(sc, tr)
    assert "grade — Класс" in sys_e and "я в 10 классе" in user_e


def test_run_with_injected_complete_fn():
    def fake(system, user, temperature):
        if '"fields"' in system:  # extract prompt
            return '{"fields": {"grade": "10", "goal": "поступление по БВИ"}, "fit": 85}'
        return '{"fit": 90, "verdict": "горячий"}'

    scs = {"obscheedelo-site": {
        "school": "OD", "name": "OD",
        "collects": [{"id": "grade", "label": "Класс"}, {"id": "goal", "label": "Цель"}],
        "scoring": {"signals": [{"field": "grade", "match": [r"\b10\b"], "points": 7, "label": "10-11"}]},
    }}
    recs = [{"transcript": [{"role": "assistant", "text": "вы оставили заявку с сайта"}], "sale_result": "Sold"}]
    res = run(recs, scs, fake)
    assert res["label"] == [1]
    assert res["assess_fit"] == [90]
    assert res["verdict"] == ["горячий"]
    assert res["rules_gpt"][0] > 0  # grade 10 -> completeness + signal points
