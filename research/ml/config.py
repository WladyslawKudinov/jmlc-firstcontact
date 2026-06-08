import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
DATA = ROOT / "data"
RAW = DATA / "raw"
MODELS = ROOT / "models"
REPORT = ROOT / "docs" / "report"

GLOBAL_SEED = 42

def load_scenarios(path=None):
    path = Path(path) if path else (ASSETS / "scenarios.json")
    return json.loads(path.read_text(encoding="utf-8"))

def scenarios_by_id(scenarios):
    return {s["id"]: s for s in scenarios}

def field_ids(scenario):
    return [c["id"] for c in scenario.get("collects", []) if isinstance(c, dict)]
