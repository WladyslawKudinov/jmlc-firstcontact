"""Prod LLM lead-scorer as a real-call baseline (Phase 5).

Faithfully reproduces the production `lib/assess-lead.js` (gpt-4o-mini): a holistic
LLM `fit` (0-100) + `verdict` over the whole conversation, and the extraction path
(`extractLead`) that pulls the qualification fields. We then run those LLM-extracted
fields back through the `lib/scoring.js` rule scorer (`ml.baseline.score_lead`) to
ablate **features (extraction) vs the rule algorithm**.

Findings on the 31 real calls (`docs/report/llm_baseline.json`):
  GPT assess-fit  0.80  >  rules on GPT-extracted fields  0.76  >  rules on hand fields  0.66
  i.e. the gap between the rules and the LLM is ~70% **extraction**, not the algorithm.

The prompt builders and parsers are pure (no key) so they unit-test with an injected
`complete_fn`. `main()` needs `OPENAI_API_KEY` and hits the API (31 calls x2).
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from .baseline import score_lead
from .config import load_scenarios, scenarios_by_id
from .llm import make_complete_fn
from .metrics import bootstrap_auc_ci, evaluate_scores
from .real_data import infer_scenario_id

VERDICT_ORDER = {"горячий": 2.0, "холодный": 1.0, "не подходящий": 0.0}


def _convo(transcript, limit):
    return "\n".join(
        f"{'Лид' if t.get('role') == 'user' else 'София'}: {t.get('text', '')}"
        for t in (transcript or [])
    )[:limit]


def build_assess_messages(scenario, fields, transcript):
    """Port of assess-lead.js buildAssessMessages -> (system, user)."""
    ft = "\n".join(f"- {k}: {v}" for k, v in (fields or {}).items() if v not in (None, "")) or "(поля не собраны)"
    system = (
        f"Ты — аналитик отдела продаж онлайн-школы «{scenario.get('school', '')}» ({scenario.get('name', '')}). "
        "По разговору голосового квалификатора с лидом оцени КАЧЕСТВО лида для менеджера и поставь балл 0–100 "
        "(та же шкала, что и у правил отдела: полнота анкеты + сила сигналов). "
        "Критерии «горячести»: чёткая цель (поступление в вуз, БВИ, призёрство), готовность стартовать в ближайшие 1–3 месяца, "
        "наличие олимпиадного опыта, подходящий класс (8–11) и гуманитарный предмет. "
        "Размытая/слабая мотивация или далёкий/неопределённый старт → низкий балл (холодный). "
        "Не тот класс/предмет, отказ, троллинг → очень низкий балл. Учитывай весь контекст и нюансы, а не отдельные слова. "
        "Также определи verdict — ОБЯЗАТЕЛЬНО одно из: «горячий» / «холодный» / «не подходящий». "
        "«Не подходящий» в т.ч. если цель НЕ про олимпиады (ОГЭ/ЕГЭ/ВПР/школьные — не профиль), не тот класс/предмет, троллинг или отказ. "
        'Верни СТРОГО JSON: {"fit": <целое 0-100>, "verdict": "<горячий|холодный|не подходящий>"}.'
    )
    user = f"Собранные поля:\n{ft}\n\nТранскрипт разговора:\n{_convo(transcript, 6000)}"
    return system, user


def build_extract_messages(scenario, transcript):
    """Port of assess-lead.js buildExtractMessages -> (system, user)."""
    collects = [c for c in scenario.get("collects", []) if isinstance(c, dict)]
    field_list = "\n".join(
        f"- {c['id']} — {c.get('label', '')}" + (": " + c["description"] if c.get("description") else "")
        for c in collects
    )
    system = (
        f"Ты — аналитик онлайн-школы «{scenario.get('school', '')}». По транскрипту разговора Софии (квалификатор) с лидом "
        "извлеки поля анкеты ТОЧНО по тому, что сказал лид, и поставь балл качества 0–100 (полнота + сила сигналов: "
        "чёткая цель, близкий старт, олимпиадный опыт, подходящий класс/предмет). Правила:\n"
        "- Для каждого поля верни короткое значение по-русски. null ТОЛЬКО если поле в разговоре вообще не прозвучало. "
        "Если лид ответил — обязательно извлеки (напр. «в десятом» → «10»; «поступление по БВИ» → «поступление по БВИ»).\n"
        "- Ничего не выдумывай.\n"
        "- red_flag — ТОЛЬКО при явной грубости/угрозах или прямом отказе говорить, иначе null.\n"
        "- open_question — только реальный вопрос лида про оффер, иначе null.\n"
        "- timezone — только если лид явно назвал город/пояс, иначе null.\n"
        "- verdict — ОБЯЗАТЕЛЬНО одно из: «горячий» / «холодный» / «не подходящий».\n"
        f"Поля:\n{field_list}\n\n"
        'Верни СТРОГО JSON: {"fields": { <id>: <строка|null>, ... }, "fit": <целое 0-100>}.'
    )
    return system, f"Транскрипт:\n{_convo(transcript, 8000)}"


def _clamp_fit(v):
    try:
        return max(0, min(100, int(round(float(v)))))
    except (TypeError, ValueError):
        return None


def parse_assess(text):
    o = json.loads(text[text.find("{"):text.rfind("}") + 1])
    return _clamp_fit(o.get("fit")), str(o.get("verdict", "")).strip()


def parse_extract(text):
    o = json.loads(text[text.find("{"):text.rfind("}") + 1])
    fields = o.get("fields") if isinstance(o.get("fields"), dict) else {}
    return {k: v for k, v in fields.items() if v not in (None, "", "null")}, _clamp_fit(o.get("fit"))


def run(records, scs_by_id, complete, hand_fields=None):
    """Score each real call. `complete(system, user, temperature)` is injectable."""
    hand_fields = hand_fields or []
    y, assess_fit, verdict, rules_gpt, rules_hand, extract_fit = [], [], [], [], [], []
    for i, r in enumerate(records):
        tr = r.get("transcript", [])
        sc = scs_by_id.get(infer_scenario_id(tr), {})
        try:
            af, vd = parse_assess(complete(*build_assess_messages(sc, hand_fields[i] if i < len(hand_fields) else {}, tr), 0.2))
        except Exception:  # noqa: BLE001
            af, vd = None, ""
        try:
            gfields, ef = parse_extract(complete(*build_extract_messages(sc, tr), 0.2))
        except Exception:  # noqa: BLE001
            gfields, ef = {}, None
        y.append(1 if r.get("sale_result") == "Sold" else 0)
        assess_fit.append(af)
        verdict.append(vd)
        extract_fit.append(ef)
        rules_gpt.append(score_lead(sc, gfields)["fit"])
        rules_hand.append(score_lead(sc, hand_fields[i] if i < len(hand_fields) else {})["fit"])
    return {"label": y, "assess_fit": assess_fit, "verdict": verdict,
            "extract_fit": extract_fit, "rules_gpt": rules_gpt, "rules_hand": rules_hand}


def _auc_block(y, scores):
    s = np.array([x if x is not None else np.nan for x in scores], float)
    m = ~np.isnan(s)
    yy, ss = np.array(y)[m].tolist(), s[m].tolist()
    b = evaluate_scores(yy, ss)
    return {"auc": round(b["auc"], 4), "ap": round(b["ap"], 4),
            "auc_ci95": [round(c, 4) for c in bootstrap_auc_ci(yy, ss)], "n": int(m.sum())}


def summarize(res):
    y = res["label"]
    return {
        "n": len(y), "label_pos": int(sum(y)),
        "gpt_assess_fit": _auc_block(y, res["assess_fit"]),
        "gpt_verdict_rerun": _auc_block(y, [VERDICT_ORDER.get(v, 0.0) for v in res["verdict"]]),
        "gpt_extract_fit": _auc_block(y, res["extract_fit"]),
        "rules_on_gpt_fields": _auc_block(y, res["rules_gpt"]),
        "rules_on_hand_fields": _auc_block(y, res["rules_hand"]),
        "per_call": res,
    }


def main():
    ap = argparse.ArgumentParser(description="LLM lead-scorer baseline on the real calls")
    ap.add_argument("--src", default="data/real/leads-final.raw.json")
    ap.add_argument("--fields", default="data/real/real_fields.json")
    ap.add_argument("--out", default="docs/report/llm_baseline.json")
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    for ln in Path(".env").read_text(encoding="utf-8").splitlines() if Path(".env").exists() else []:
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    records = json.loads(Path(args.src).read_text(encoding="utf-8"))
    hand = json.loads(Path(args.fields).read_text(encoding="utf-8")) if Path(args.fields).exists() else []
    scs = scenarios_by_id(load_scenarios())
    complete = make_complete_fn(provider="openai", model=args.model)

    out = summarize(run(records, scs, complete, hand))
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for k in ("gpt_assess_fit", "rules_on_gpt_fields", "rules_on_hand_fields"):
        print(f"{k:22} AUC={out[k]['auc']}  CI={out[k]['auc_ci95']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
