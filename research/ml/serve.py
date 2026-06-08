"""Callback-priority scoring service — the integration point back into the CRM.

The study's recommendation (Phase 5): rank callbacks by the **LLM scorer** (best on real calls,
AUC 0.80); the **rule scorer** is the cheap, offline fallback. Selected behind a `CALLBACK_SCORER`
flag (``llm`` | ``rules``). This is a stub — it returns a priority in [0, 1] for one qualified
lead; wiring it into the live prod queue is out of scope for the research exhibit.

    CALLBACK_SCORER=rules python -m ml.serve --scenario obscheedelo-site lead.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

from .baseline import score_lead

DEFAULT_SCORER = "rules"


def priority_score(scenario, fields, transcript=None, scorer=None, complete=None):
    """Callback priority in [0, 1] for one lead.

    `scorer`: 'rules' (offline, no key) or 'llm' (needs an LLM). `complete` is an injectable
    `(system, user, temperature) -> str` LLM function — required only for scorer='llm' (so this
    stays unit-testable without a key).
    """
    scorer = (scorer or os.environ.get("CALLBACK_SCORER", DEFAULT_SCORER)).lower()
    if scorer == "rules":
        return score_lead(scenario, fields or {})["fit"] / 100.0
    if scorer == "llm":
        from .llm_baseline import build_assess_messages, parse_assess
        if complete is None:
            from .llm import make_complete_fn
            complete = make_complete_fn(provider="openai", model="gpt-4o-mini")
        fit, _ = parse_assess(complete(*build_assess_messages(scenario, fields or {}, transcript or []), 0.2))
        return (fit or 0) / 100.0
    raise ValueError(f"unknown CALLBACK_SCORER '{scorer}' (use 'llm' or 'rules')")


def main():
    ap = argparse.ArgumentParser(description="Score one lead's callback priority (stub).")
    ap.add_argument("--scorer", default=None, help="llm | rules (default: $CALLBACK_SCORER or rules)")
    ap.add_argument("--scenario", required=True, help="scenario id from assets/scenarios.json")
    ap.add_argument("call", nargs="?", default="-", help="lead JSON ({fields, transcript}); '-' = stdin")
    args = ap.parse_args()
    from .config import load_scenarios, scenarios_by_id
    sc = scenarios_by_id(load_scenarios()).get(args.scenario, {})
    raw = sys.stdin.read() if args.call == "-" else Path(args.call).read_text(encoding="utf-8")
    lead = json.loads(raw)
    p = priority_score(sc, lead.get("fields"), lead.get("transcript"), args.scorer)
    print(json.dumps({"scenario": args.scenario, "priority": round(p, 4)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
