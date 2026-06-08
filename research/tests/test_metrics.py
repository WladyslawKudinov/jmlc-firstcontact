from ml.metrics import precision_at_k, ndcg_at_k, evaluate_scores

def test_precision_at_k_perfect_and_worst():
    assert precision_at_k([1, 1, 0, 0], [0.9, 0.8, 0.1, 0.2], 0.5) == 1.0
    assert precision_at_k([0, 0, 1, 1], [0.9, 0.8, 0.1, 0.2], 0.5) == 0.0

def test_ndcg_perfect_is_one():
    assert abs(ndcg_at_k([1, 1, 0, 0], [0.9, 0.8, 0.1, 0.2], 1.0) - 1.0) < 1e-9

def test_evaluate_scores_keys():
    out = evaluate_scores([0, 1, 0, 1], [0.2, 0.9, 0.3, 0.8])
    assert {"auc", "ap", "precision_at_20pct", "ndcg_at_20pct"} <= set(out)
    assert out["auc"] == 1.0

def test_bootstrap_auc_ci_brackets_and_orders():
    from ml.metrics import bootstrap_auc_ci
    y = [0, 0, 0, 1, 1, 1] * 4          # perfectly separable -> CI pinned near 1.0
    s = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9] * 4
    lo, hi = bootstrap_auc_ci(y, s, n=500, seed=0)
    assert 0.5 < lo <= hi <= 1.0

def test_bootstrap_auc_ci_is_deterministic_with_seed():
    from ml.metrics import bootstrap_auc_ci
    y = [0, 1, 0, 1, 1, 0, 0, 1]
    s = [0.2, 0.7, 0.4, 0.6, 0.9, 0.1, 0.5, 0.8]
    assert bootstrap_auc_ci(y, s, n=300, seed=1) == bootstrap_auc_ci(y, s, n=300, seed=1)
