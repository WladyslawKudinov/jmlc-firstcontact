import argparse
import json
from pathlib import Path

import numpy as np

from .baseline import score_lead
from .config import MODELS, RAW, REPORT, load_scenarios, scenarios_by_id
from .features import featurize
from .metrics import bootstrap_auc_ci, evaluate_scores
from .real_data import verdict_scores

def baseline_scores(records, scs_by_id):
    return np.array([score_lead(scs_by_id[r["scenario_id"]], r.get("fields", {}))["fit"]
                     for r in records], dtype=float)

def model_scores(records, model, vectorizer):
    X = featurize(records, vectorizer, fit=False)
    return model.predict_proba(X)[:, 1]

def compare(records, model, vectorizer, scs_by_id):
    y = [int(r["label"]) for r in records]
    return {
        "n": len(records),
        "model": evaluate_scores(y, model_scores(records, model, vectorizer)),
        "baseline": evaluate_scores(y, baseline_scores(records, scs_by_id)),
    }

def compare_real(records, model, vectorizer, scs_by_id=None):
    """Real calls: learned model vs the prod agent's `verdict` (the rule lead-score
    actually used in production). If `scs_by_id` is given AND fields were extracted
    onto the rows, also report the exact lib/scoring.js rule baseline."""
    y = [int(r["label"]) for r in records]
    ms = model_scores(records, model, vectorizer)
    vs = verdict_scores(records)
    model_blk = evaluate_scores(y, ms)
    model_blk["auc_ci95"] = list(bootstrap_auc_ci(y, ms))
    verdict_blk = evaluate_scores(y, vs)
    verdict_blk["auc_ci95"] = list(bootstrap_auc_ci(y, vs))
    out = {"n": len(records), "model": model_blk, "baseline_verdict": verdict_blk}
    if scs_by_id:
        rs = baseline_scores(records, scs_by_id)
        rules_blk = evaluate_scores(y, rs)
        rules_blk["auc_ci95"] = list(bootstrap_auc_ci(y, rs))
        out["baseline_rules"] = rules_blk
    return out

def leakage_auc(records):
    """A deliberately dumb word-count model. Should be well below ~1.0 if no leak."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    texts = [r["transcript_text"] for r in records]
    y = np.array([int(r["label"]) for r in records])
    X = CountVectorizer().fit_transform(texts)
    proba = cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=5, method="predict_proba")[:, 1]
    return float(evaluate_scores(y, proba)["auc"])

def _read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

def main():
    import joblib
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(MODELS))
    ap.add_argument("--data", default=str(RAW / "test.jsonl"))
    ap.add_argument("--ood", default=str(RAW / "ood.jsonl"))
    ap.add_argument("--real", default=str(Path("data/real/real.jsonl")))
    args = ap.parse_args()

    scs_by_id = scenarios_by_id(load_scenarios())
    model = joblib.load(Path(args.models) / "model.joblib")
    vec = joblib.load(Path(args.models) / "vectorizer.joblib")

    results = {"test": compare(_read_jsonl(args.data), model, vec, scs_by_id)}
    ood = _read_jsonl(args.ood)
    if ood:
        results["ood"] = compare(ood, model, vec, scs_by_id)
    real = _read_jsonl(args.real)
    if real:
        results["real"] = compare_real(real, model, vec, scs_by_id)
    results["leakage_auc_dumb_model"] = leakage_auc(_read_jsonl(args.data))

    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))

    # MLflow (best-effort; never fail the run on tracking issues)
    try:
        import os
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # recent MLflow gates file store
        import mlflow
        mlflow.set_tracking_uri(f"file://{Path('mlruns').resolve()}")
        with mlflow.start_run():
            for split, blk in results.items():
                if isinstance(blk, dict):
                    for sub in ("model", "baseline", "baseline_verdict"):
                        if isinstance(blk.get(sub), dict):
                            for k, v in blk[sub].items():
                                mlflow.log_metric(f"{split}_{sub}_{k}", v)
            mlflow.log_metric("leakage_auc", results["leakage_auc_dumb_model"])
            mlflow.log_artifact(str(REPORT / "metrics.json"))
    except Exception as e:  # noqa: BLE001
        print(f"[mlflow skipped] {e}")

if __name__ == "__main__":
    main()
