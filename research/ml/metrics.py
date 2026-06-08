import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def _topk_idx(scores, k_frac):
    n = len(scores)
    k = max(1, int(round(n * k_frac)))
    order = np.argsort(-np.asarray(scores, dtype=float))
    return order[:k], k

def precision_at_k(y_true, scores, k_frac=0.2):
    idx, k = _topk_idx(scores, k_frac)
    return float(np.asarray(y_true)[idx].sum() / k)

def ndcg_at_k(y_true, scores, k_frac=0.2):
    idx, k = _topk_idx(scores, k_frac)
    gains = np.asarray(y_true, dtype=float)[idx]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((gains * discounts).sum())
    ideal = np.sort(np.asarray(y_true, dtype=float))[::-1][:k]
    idcg = float((ideal * discounts).sum())
    return (dcg / idcg) if idcg > 0 else 0.0

def bootstrap_auc_ci(y_true, scores, n=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for AUC — essential honesty on tiny sets (real N=31)."""
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    aucs = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if y[b].min() == y[b].max():  # need both classes to define AUC
            continue
        aucs.append(roc_auc_score(y[b], s[b]))
    if not aucs:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))

def evaluate_scores(y_true, scores, k_frac=0.2):
    y_true = list(y_true)
    multiclass = len(set(y_true)) > 1
    return {
        "auc": float(roc_auc_score(y_true, scores)) if multiclass else float("nan"),
        "ap": float(average_precision_score(y_true, scores)) if multiclass else float("nan"),
        "precision_at_20pct": precision_at_k(y_true, scores, k_frac),
        "ndcg_at_20pct": ndcg_at_k(y_true, scores, k_frac),
    }
