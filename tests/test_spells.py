import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import spells as SP  # noqa: E402
from src.bench import A, D, H, X  # noqa: E402


def test_runs_stay_inside_halves():
    pred = np.array([X, H, H, D, A, X, X, A, A, H, X], np.int8)
    period = np.array([0, 1, 1, 1, 1, 0, 0, 2, 2, 2, 0], np.int8)
    assert SP.runs(pred, period).tolist() == [[1, 3], [3, 4], [4, 5], [7, 9], [9, 10]]


def test_run_classifier_is_team_symmetric():
    rng = np.random.default_rng(0)
    n = 4000
    period = np.repeat(np.array([0, 1, 2, 0], np.int8), [100, 1900, 1900, 100])
    truth = np.repeat(rng.choice([H, A, D], 80), 50).astype(np.int8)
    pred = np.where(rng.random(n) < 0.02, D, truth).astype(np.int8)
    q, r = rng.random(n), np.clip((truth == H) * 0.6 + rng.random(n) * 0.4, 0, 1)
    sym, anti = rng.normal(size=(n, 3)), rng.normal(size=(n, 2))
    rr, s, a = SP.run_features(pred, period, q, r, sym, anti)
    clf = SP.fit([(rr, s, a, truth)])
    _, s2, a2 = SP.run_features(np.where(pred == D, D, H + A - pred).astype(np.int8), period, q, 1 - r,
                                sym, -anti)
    np.testing.assert_allclose(s2, s, atol=1e-12)        # features: symmetric up to rounding
    np.testing.assert_allclose(a2, -a, atol=1e-12)
    np.testing.assert_allclose(SP.proba(clf, s, a), SP.proba(clf, s, -a)[:, SP.SWAP], atol=1e-12)
