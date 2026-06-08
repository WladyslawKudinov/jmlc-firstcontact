import re

COMPLETENESS_WEIGHT = 50
SIGNAL_CAP = 50
SIGNAL_FLOOR = -25

def _is_filled(v):
    return v is not None and str(v).strip() != ""

def _matches(signal, value):
    if not _is_filled(value):
        return False
    s = str(value)
    if isinstance(signal.get("match"), list):
        return any(re.search(p, s, re.IGNORECASE) for p in signal["match"])
    if isinstance(signal.get("atLeast"), (int, float)):
        m = re.search(r"-?\d+", s)
        return (int(m.group()) >= signal["atLeast"]) if m else False
    return False

def score_lead(scenario, fields):
    fields = fields or {}
    ids = [c["id"] for c in scenario.get("collects", []) if isinstance(c, dict)]
    total = len(ids) or 1
    filled = sum(1 for i in ids if _is_filled(fields.get(i)))
    completeness_points = round(filled / total * COMPLETENESS_WEIGHT)
    breakdown = []
    if filled > 0:
        breakdown.append({"label": f"полнота {filled}/{total}", "points": completeness_points})
    signal_sum = 0
    for sig in scenario.get("scoring", {}).get("signals", []):
        if _matches(sig, fields.get(sig["field"])):
            signal_sum += sig["points"]
            breakdown.append({"label": sig["label"], "points": sig["points"]})
    signal_points = max(SIGNAL_FLOOR, min(SIGNAL_CAP, signal_sum))
    fit = max(0, min(100, completeness_points + signal_points))
    return {"fit": fit, "completeness": filled, "breakdown": breakdown}
