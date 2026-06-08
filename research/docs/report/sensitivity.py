"""Signal-strength sensitivity analysis — the transparency backbone of the
generator recalibration. Re-labels the stored theta across a range of trait->outcome
signal strengths (no regeneration) and plots learned-model vs rule-baseline AUC as a
function of how predictable lead quality is (the oracle AUC). Shows the learned model
beats the rules across the *entire* realistic range, not at a cherry-picked point.

    python docs/report/sensitivity.py     # -> docs/report/sensitivity.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ml.metrics import evaluate_scores
from ml.personas import LATENT_TRAITS, TRAIT_WEIGHTS, BIAS
from ml.baseline import score_lead
from ml.config import load_scenarios, scenarios_by_id

SCS = scenarios_by_id(load_scenarios())
RAW = ROOT / "data" / "raw"
read = lambda p: [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
data = read(RAW / "train.jsonl") + read(RAW / "test.jsonl")
N = len(data)

# centered logit per record under the CURRENT (recalibrated) weights; sweep a multiplier
# m so m=1.0 is the shipped recalibrated generator. (weights already 2x the original.)
L = np.array([sum(TRAIT_WEIGHTS[t] * r["theta"][t] for t in LATENT_TRAITS) - BIAS for r in data])
texts = [r["transcript_text"] for r in data]
Xcont = np.array([[r["theta"][t] for t in LATENT_TRAITS] for r in data], float)
baseline = np.array([score_lead(SCS[r["scenario_id"]], r.get("fields", {}))["fit"] for r in data], float)
rng = np.random.default_rng(0)
u, z = rng.random(N), rng.standard_normal(N)
SD = 0.4
sig = lambda x: 1.0 / (1.0 + np.exp(-x))
cv = StratifiedKFold(5, shuffle=True, random_state=42)

def auc_of(estimator, X, y):
    return evaluate_scores(y, cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")[:, 1])["auc"]

mults = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
rows = []
for m in mults:
    y = (u < sig(m * L + SD * z)).astype(int)
    if y.sum() in (0, N):
        continue
    oracle = auc_of(LogisticRegression(max_iter=2000), Xcont, y)
    model = auc_of(Pipeline([("tf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)),
                             ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))]), texts, y)
    base = evaluate_scores(y, baseline)["auc"]
    rows.append((m, oracle, model, base))
    print(f"m={m:<4} oracle={oracle:.3f} model={model:.3f} baseline={base:.3f}  gap={model-base:+.3f}")

m, oracle, model, base = map(np.array, zip(*rows))
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(oracle, model, "o-", color="#2e7d32", lw=2.2, label="learned model (TF-IDF + LR)")
ax.plot(oracle, base, "s--", color="#c62828", lw=2.0, label="rule baseline (prod lib/scoring.js)")
ax.plot(oracle, oracle, ":", color="#888", lw=1.3, label="oracle ceiling (exact θ)")
# mark the shipped recalibrated operating point (m=1.0)
i = list(m).index(1.0)
ax.axvline(oracle[i], color="#1565c0", ls=":", lw=1.2)
ax.annotate("shipped generator\n(realistic predictability)",
            xy=(oracle[i], base[i]), xytext=(oracle[i] - 0.02, 0.52),
            fontsize=9, color="#1565c0", ha="right",
            arrowprops=dict(arrowstyle="->", color="#1565c0"))
ax.set_xlabel("lead-quality predictability  (oracle AUC, exact latent traits)")
ax.set_ylabel("ranking quality  (ROC-AUC, 5-fold CV)")
ax.set_title("Learned model beats the rules across the whole realistic difficulty range")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
out = Path(__file__).resolve().parent / "sensitivity.png"
fig.savefig(out, dpi=130)
print(f"wrote {out}")
