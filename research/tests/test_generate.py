import json
import numpy as np
import pytest
from ml.generate import make_record, transcript_text, generate_dataset, generate_to_jsonl
from ml.config import load_scenarios
from ml.llm import LLMError

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


# --- generator hardening: incremental, resumable, skip-on-failure writes ---

def _read_lines(path):
    return [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

def test_generate_to_jsonl_writes_each_record(tmp_path):
    out = tmp_path / "gen.jsonl"
    made, skipped = generate_to_jsonl(SCS, 5, np.random.default_rng(0),
                                      _stub_complete, 0.9, "A", out, resume=True)
    assert (made, skipped) == (5, 0)
    recs = [json.loads(l) for l in _read_lines(out)]
    assert len(recs) == 5
    assert all(r["label"] in (0, 1) and r["generator"] == "A" for r in recs)

def test_generate_to_jsonl_resumes_from_existing(tmp_path):
    out = tmp_path / "gen.jsonl"
    out.write_text('{"a":1}\n{"a":2}\n{"a":3}\n', encoding="utf-8")   # 3 already done
    calls = {"n": 0}
    def counting(system, user, temperature=0.9):
        calls["n"] += 1
        return _stub_complete(system, user, temperature)
    made, skipped = generate_to_jsonl(SCS, 5, np.random.default_rng(0),
                                      counting, 0.9, "A", out, resume=True)
    assert made == 5 and skipped == 0
    assert calls["n"] == 2                       # only the missing 2 were generated
    assert len(_read_lines(out)) == 5

def test_generate_to_jsonl_skips_failed_generations(tmp_path):
    out = tmp_path / "gen.jsonl"
    calls = {"n": 0}
    def flaky(system, user, temperature=0.9):
        calls["n"] += 1
        if calls["n"] == 1:                      # first call fails even after retries
            raise LLMError("transient failure exhausted")
        return _stub_complete(system, user, temperature)
    made, skipped = generate_to_jsonl(SCS, 3, np.random.default_rng(0),
                                      flaky, 0.9, "A", out, resume=True)
    assert made == 3 and skipped == 1            # one skipped, three still produced
    assert len(_read_lines(out)) == 3

def test_generate_to_jsonl_fresh_overwrites_when_not_resume(tmp_path):
    out = tmp_path / "gen.jsonl"
    out.write_text('{"old":1}\n{"old":2}\n', encoding="utf-8")
    made, _ = generate_to_jsonl(SCS, 3, np.random.default_rng(0),
                                _stub_complete, 0.9, "B", out, resume=False)
    assert made == 3
    recs = [json.loads(l) for l in _read_lines(out)]
    assert len(recs) == 3 and all(r["generator"] == "B" for r in recs)   # old rows gone

def test_generate_to_jsonl_skips_unparseable_response(tmp_path):
    # a malformed-JSON LLM response (the real failure that crashed a 1500-call run)
    # must be skipped, not crash the whole batch.
    out = tmp_path / "gen.jsonl"
    calls = {"n": 0}
    def flaky_json(system, user, temperature=0.9):
        calls["n"] += 1
        if calls["n"] == 1:
            return '{"transcript":[] "fields":{}}'   # missing comma -> JSONDecodeError
        return _stub_complete(system, user, temperature)
    made, skipped = generate_to_jsonl(SCS, 2, np.random.default_rng(0),
                                      flaky_json, 0.9, "A", out, resume=True)
    assert made == 2 and skipped == 1

def test_generate_to_jsonl_aborts_on_persistent_failure(tmp_path):
    # a systemically-broken model (e.g. wrong name -> 404 every call) must NOT loop
    # forever skipping; it should abort after N consecutive failures.
    out = tmp_path / "gen.jsonl"
    def always_fail(system, user, temperature=0.9):
        raise LLMError("persistent failure (e.g. 404 model not found)")
    with pytest.raises(LLMError):
        generate_to_jsonl(SCS, 5, np.random.default_rng(0),
                          always_fail, 0.9, "A", out, resume=True, max_consecutive_skips=4)
