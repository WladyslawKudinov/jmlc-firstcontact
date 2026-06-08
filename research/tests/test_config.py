from ml.config import load_scenarios, scenarios_by_id, field_ids

def test_load_scenarios_has_three_with_collects():
    scs = load_scenarios()
    ids = {s["id"] for s in scs}
    assert {"english-courses-qualify", "obscheedelo-site", "obscheedelo-tg"} <= ids
    eng = scenarios_by_id(scs)["english-courses-qualify"]
    assert field_ids(eng) == ["situation", "level", "past_attempts", "hours_per_week", "start_date"]
