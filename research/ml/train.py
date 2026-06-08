import argparse
import json
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression

from .config import MODELS, RAW
from .features import featurize, labels, make_vectorizer

def train_model(records):
    vectorizer = make_vectorizer()
    X = featurize(records, vectorizer, fit=True)
    y = labels(records)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X, y)
    return model, vectorizer

def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(RAW / "train.jsonl"))
    ap.add_argument("--out", default=str(MODELS))
    args = ap.parse_args()
    records = _read_jsonl(args.data)
    model, vectorizer = train_model(records)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    joblib.dump(vectorizer, out / "vectorizer.joblib")
    print(f"trained on {len(records)} -> {out}")

if __name__ == "__main__":
    main()
