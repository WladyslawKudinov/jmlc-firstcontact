import numpy as np
from ml.relabel import relabel_records
from ml.personas import LATENT_TRAITS

def test_relabel_records_derives_from_theta():
    hi = {"theta": {t: 0.9 for t in LATENT_TRAITS}, "label": 0, "keep": "x"}
    lo = {"theta": {t: 0.1 for t in LATENT_TRAITS}, "label": 1, "keep": "y"}
    out = relabel_records([dict(hi) for _ in range(200)] + [dict(lo) for _ in range(200)],
                          np.random.default_rng(0))
    labels = [r["label"] for r in out]
    assert set(labels) <= {0, 1}
    assert out[0]["keep"] == "x"                  # non-label fields preserved
    assert np.mean(labels[:200]) > 0.8            # high-trait leads -> mostly enrolled
    assert np.mean(labels[200:]) < 0.2            # low-trait leads  -> mostly not

def test_relabel_records_deterministic():
    recs = [{"theta": {t: 0.5 for t in LATENT_TRAITS}, "label": 0} for _ in range(50)]
    a = relabel_records([dict(r) for r in recs], np.random.default_rng(7))
    b = relabel_records([dict(r) for r in recs], np.random.default_rng(7))
    assert [r["label"] for r in a] == [r["label"] for r in b]
