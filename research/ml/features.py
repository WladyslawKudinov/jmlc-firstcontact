import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

def structured_features(record):
    transcript = record.get("transcript", []) or []
    user_turns = [t for t in transcript if t.get("role") == "user"]
    fields = record.get("fields", {}) or {}
    n_fields = len(fields)
    filled = sum(1 for v in fields.values() if v not in (None, ""))
    completeness = (filled / n_fields) if n_fields else 0.0
    text = record.get("transcript_text", "")
    avg_user_len = float(np.mean([len(t["text"]) for t in user_turns])) if user_turns else 0.0
    return [completeness, float(len(transcript)), float(len(user_turns)),
            float(len(text)), avg_user_len]

def make_vectorizer():
    # char n-grams handle Russian morphology without a tokenizer
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=20000)

def _texts(records):
    return [r.get("transcript_text", "") for r in records]

def featurize(records, vectorizer, fit=False):
    X_text = vectorizer.fit_transform(_texts(records)) if fit else vectorizer.transform(_texts(records))
    X_struct = csr_matrix(np.array([structured_features(r) for r in records], dtype=float))
    return hstack([X_text, X_struct]).tocsr()

def labels(records):
    return np.array([int(r["label"]) for r in records])
