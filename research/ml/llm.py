import os
import sys
import json
import time
import httpx

BASE_URLS = {"xai": "https://api.x.ai/v1", "openai": "https://api.openai.com/v1"}
KEY_ENV = {"xai": "XAI_API_KEY", "openai": "OPENAI_API_KEY"}
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

class LLMError(RuntimeError):
    def __init__(self, message="", *, status=None, retryable=False, retry_after=None):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.retry_after = retry_after

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
    try:
        obj = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as e:
        raise LLMError(f"malformed JSON in generation: {e}") from e
    if not isinstance(obj, dict):
        raise LLMError("generation JSON is not an object")
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

def _parse_retry_after(value):
    """Retry-After header -> seconds. Handles the integer-seconds form (common on 429);
    falls back to computed backoff (None) for the HTTP-date form."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def retrying(do_once, *, max_retries=5, base_delay=1.0, max_delay=30.0,
             sleep=time.sleep, on_retry=None):
    """Call do_once(); on a *retryable* LLMError, back off and retry up to max_retries.
    Honors LLMError.retry_after when set, else exponential backoff. Non-retryable
    errors propagate immediately. sleep is injectable for tests."""
    attempt = 0
    while True:
        try:
            return do_once()
        except LLMError as e:
            if not e.retryable or attempt >= max_retries:
                raise
            delay = e.retry_after or min(max_delay, base_delay * (2 ** attempt))
            if on_retry is not None:
                on_retry(attempt + 1, delay, e)
            sleep(delay)
            attempt += 1

def _request_once(base, key, model, system, user, temperature, timeout):
    try:
        resp = httpx.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "temperature": temperature,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}]},
            timeout=timeout,
        )
    except httpx.TimeoutException as e:
        raise LLMError(f"timeout: {e}", retryable=True)
    except httpx.TransportError as e:
        raise LLMError(f"transport error: {e}", retryable=True)
    if resp.status_code != 200:
        raise LLMError(
            f"{resp.status_code}: {resp.text[:200]}",
            status=resp.status_code,
            retryable=resp.status_code in _RETRYABLE_STATUS,
            retry_after=_parse_retry_after(resp.headers.get("retry-after")),
        )
    return resp.json()["choices"][0]["message"]["content"]

def make_complete_fn(provider=None, model=None, timeout=60.0,
                     max_retries=5, base_delay=1.0):
    provider = provider or os.environ.get("GEN_PROVIDER", "xai")
    if provider not in BASE_URLS:
        raise LLMError(f"unknown provider '{provider}'; supported: {', '.join(sorted(BASE_URLS))}")
    model = model or os.environ.get("GEN_MODEL")
    if not model:
        raise LLMError("GEN_MODEL not set")
    key = os.environ.get(KEY_ENV[provider])
    if not key:
        raise LLMError(f"{KEY_ENV[provider]} not set")
    base = BASE_URLS[provider]

    def _log_retry(attempt, delay, err):
        print(f"[retry {attempt}/{max_retries}] {err} -> sleeping {delay:.1f}s", file=sys.stderr)

    def complete(system, user, temperature=0.9):
        return retrying(
            lambda: _request_once(base, key, model, system, user, temperature, timeout),
            max_retries=max_retries, base_delay=base_delay, on_retry=_log_retry,
        )

    return complete
