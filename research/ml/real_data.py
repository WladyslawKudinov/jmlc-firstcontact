"""Adapter for the genuinely-collected real-call set (prod CRM export).

The real export (`leads-final.json`) carries `transcript` (with ASR `t_sec`
timings), the agent's `verdict`/`red_flag`/`open_question`, and the real
downstream `sale_result` — but NOT the extracted qualification `fields` or a
`scenario_id`. This maps it into the pipeline row shape so the learned model
can be scored on it, with the REAL outcome as the label.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .generate import transcript_text

# The prod agent's hot/cold call, as an ordinal lead score (the real-data baseline).
_VERDICT_ORDER = {"горячий": 2.0, "холодный": 1.0, "не подходящий": 0.0}


def infer_scenario_id(transcript):
    """Real OD calls announce their source in the opener ('с сайта' / 'в боте')."""
    opener = (transcript[0]["text"] if transcript else "").lower()
    if "сайт" in opener:
        return "obscheedelo-site"
    if "бот" in opener:
        return "obscheedelo-tg"
    return "obscheedelo-tg"


def to_rows(raw_records, fields_list=None):
    """Map prod CRM export records into pipeline rows.

    Label = the REAL outcome `sale_result` (Sold=1 / Failed=0), NOT the agent's
    call-time verdict. The export has no extracted `fields`; pass `fields_list`
    (a by-index sidecar) to attach them so the lib/scoring.js baseline can run.
    """
    rows = []
    for i, r in enumerate(raw_records):
        tr = r.get("transcript", []) or []
        fields = dict(r.get("fields", {}) or {})
        if fields_list and i < len(fields_list) and fields_list[i]:
            fields.update({k: v for k, v in fields_list[i].items() if v not in (None, "")})
        rows.append({
            "call_id": r.get("call_id", f"real-{i:03d}"),
            "scenario_id": infer_scenario_id(tr),
            "transcript": tr,
            "transcript_text": transcript_text(tr),
            "fields": fields,
            "label": 1 if r.get("sale_result") == "Sold" else 0,
            "verdict": r.get("verdict"),
            "sale_result": r.get("sale_result"),
            "open_question": r.get("open_question"),
            "red_flag": r.get("red_flag"),
        })
    return rows


def verdict_scores(records):
    """Ordinal score from the prod agent's verdict (the real-call rule baseline)."""
    return np.array([_VERDICT_ORDER.get(r.get("verdict"), 0.0) for r in records], dtype=float)


def main():
    ap = argparse.ArgumentParser(description="Adapt a prod CRM call export into real.jsonl")
    ap.add_argument("--src", required=True, help="raw export JSON (list of calls)")
    ap.add_argument("--fields", default=None, help="optional by-index sidecar JSON of extracted fields")
    ap.add_argument("--out", default="data/real/real.jsonl")
    args = ap.parse_args()
    raw = json.loads(Path(args.src).read_text(encoding="utf-8"))
    fields_list = json.loads(Path(args.fields).read_text(encoding="utf-8")) if args.fields else None
    rows = to_rows(raw, fields_list)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    pos = sum(r["label"] for r in rows)
    print(f"wrote {len(rows)} real rows -> {out}  ({pos} Sold / {len(rows) - pos} Failed)")


if __name__ == "__main__":
    main()
