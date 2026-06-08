import numpy as np
from ml.personas import (
    LATENT_TRAITS, sample_latent, label_probability, sample_label, behavior_brief,
)

def test_latent_in_unit_range_and_seeded():
    a = sample_latent(np.random.default_rng(0))
    b = sample_latent(np.random.default_rng(0))
    assert set(a) == set(LATENT_TRAITS)
    assert a == b
    assert all(0.0 <= v <= 1.0 for v in a.values())

def test_label_probability_monotonic_in_readiness():
    base = {t: 0.5 for t in LATENT_TRAITS}
    hi = dict(base, readiness=0.95)
    lo = dict(base, readiness=0.05)
    assert label_probability(hi) > label_probability(lo)

def test_higher_traits_more_positive_labels():
    rng = np.random.default_rng(1)
    hi = {t: 0.85 for t in LATENT_TRAITS}
    lo = {t: 0.15 for t in LATENT_TRAITS}
    mean_hi = np.mean([sample_label(hi, rng) for _ in range(400)])
    mean_lo = np.mean([sample_label(lo, rng) for _ in range(400)])
    assert mean_hi > mean_lo + 0.2

def test_brief_leaks_no_numbers_or_label():
    brief = behavior_brief({t: 0.9 for t in LATENT_TRAITS})
    assert not any(ch.isdigit() for ch in brief)
    low = behavior_brief({t: 0.1 for t in LATENT_TRAITS})
    assert brief != low                       # behavior differs by traits
    assert "решил" in brief                    # high readiness phrase present

def test_label_probability_strongly_separates_extremes():
    # Recalibrated generator: lead quality must be *realistically predictable*.
    # An extreme-high-trait lead should have a near-certain positive outcome and
    # an extreme-low one a near-certain negative (oracle AUC ~0.78, matching real
    # lead-scoring). The as-shipped weak weights produced near-coin-flips and fail this.
    hi = label_probability({t: 0.85 for t in LATENT_TRAITS})
    lo = label_probability({t: 0.15 for t in LATENT_TRAITS})
    assert hi > 0.95 and lo < 0.05
