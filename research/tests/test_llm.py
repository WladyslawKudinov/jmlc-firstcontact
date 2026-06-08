from ml.llm import build_messages, parse_generation, LLMError, retrying, make_complete_fn
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

def test_parse_generation_raises_llmerror_on_malformed_json():
    # braces present but invalid JSON (missing comma) -> must surface as LLMError,
    # NOT a raw json.JSONDecodeError, so the generator's skip-logic can catch it.
    raw = '{"transcript":[] "fields":{}}'
    with pytest.raises(LLMError):
        parse_generation(raw, ["situation"])


# --- generator hardening: retry/backoff + provider guard ---

def test_llmerror_carries_retry_metadata():
    e = LLMError("boom", status=429, retryable=True, retry_after=2.0)
    assert e.status == 429 and e.retryable is True and e.retry_after == 2.0
    plain = LLMError("nope")
    assert plain.retryable is False and plain.status is None and plain.retry_after is None


def test_retrying_returns_after_transient_failures():
    sleeps, calls = [], {"n": 0}
    def do_once():
        calls["n"] += 1
        if calls["n"] < 3:
            raise LLMError("429", status=429, retryable=True)
        return "ok"
    out = retrying(do_once, max_retries=5, base_delay=0.1, sleep=sleeps.append)
    assert out == "ok"
    assert calls["n"] == 3
    assert len(sleeps) == 2          # two backoffs before the success


def test_retrying_raises_after_exhausting_retries():
    sleeps = []
    def do_once():
        raise LLMError("503", status=503, retryable=True)
    with pytest.raises(LLMError):
        retrying(do_once, max_retries=3, base_delay=0.1, sleep=sleeps.append)
    assert len(sleeps) == 3           # max_retries backoffs, then give up


def test_retrying_does_not_retry_non_retryable():
    sleeps, calls = [], {"n": 0}
    def do_once():
        calls["n"] += 1
        raise LLMError("400", status=400, retryable=False)
    with pytest.raises(LLMError):
        retrying(do_once, max_retries=5, base_delay=0.1, sleep=sleeps.append)
    assert calls["n"] == 1            # no retries on a non-retryable error
    assert sleeps == []


def test_retrying_honors_retry_after():
    sleeps, calls = [], {"n": 0}
    def do_once():
        calls["n"] += 1
        if calls["n"] == 1:
            raise LLMError("429", status=429, retryable=True, retry_after=2.0)
        return "ok"
    out = retrying(do_once, max_retries=5, base_delay=0.1, sleep=sleeps.append)
    assert out == "ok"
    assert sleeps == [2.0]            # server-specified Retry-After used verbatim


def test_make_complete_fn_unknown_provider_raises_llmerror():
    # anthropic is advertised in .env.example but not implemented -> must fail clearly
    with pytest.raises(LLMError):
        make_complete_fn(provider="anthropic", model="whatever")
