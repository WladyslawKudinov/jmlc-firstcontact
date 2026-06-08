import numpy as np
from ml.train import train_model

def _toy(n=40, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for _ in range(n):
        pos = rng.random() < 0.5
        text = ("срочно работа на этой неделе готов начать" if pos
                else "просто смотрю подумаю потом не уверен")
        recs.append({
            "transcript": [{"role": "user", "text": text}],
            "transcript_text": "Лид: " + text,
            "fields": {"situation": "работа" if pos else None, "start_date": None},
            "label": int(pos),
        })
    return recs

def test_train_model_returns_fitted_estimator_and_vectorizer():
    model, vec = train_model(_toy())
    from ml.features import featurize
    X = featurize(_toy(n=8, seed=9), vec, fit=False)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (8,)
    assert ((proba >= 0) & (proba <= 1)).all()
