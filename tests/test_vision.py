import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import vision as V  # noqa: E402

PITCH = np.array((105.0, 68.0))


def test_to_pitch_inverts_the_homography():
    H = np.array([[1500.0, -400.0, 100.0], [2.0, 200.0, 250.0], [-0.02, -0.5, 1.0]])
    u, v = np.array([0.3, 0.8]), np.array([0.6, 0.1])
    den = H[2, 0] * u + H[2, 1] * v + H[2, 2]
    px, py = (H[0, 0] * u + H[0, 1] * v + H[0, 2]) / den, (H[1, 0] * u + H[1, 1] * v + H[1, 2]) / den
    xy = V.to_pitch(np.array([0, 0]), px, py, H[None], PITCH)
    np.testing.assert_allclose(xy, np.column_stack([u, v]) * PITCH, atol=1e-9)


def test_accept_keeps_supported_best_candidates_and_drops_paint_and_outsiders():
    frames = np.array([0, 0, 5, 10, 40, 60, 65])
    xy = np.array([[30, 30], [80, 10], [31, 30], [32, 30],   # a moving ball, a weaker rival at 0
                   [70, 30],                                 # isolated: no support
                   [52.5, 34.2], [52.5, 34.0]])              # centre spot, twice
    score = np.array([0.9, 0.4, 0.8, 0.7, 0.95, 0.9, 0.9])
    f, p = V.accept(frames, xy.astype(float), score, PITCH)
    assert f.tolist() == [0, 5, 10]
    np.testing.assert_allclose(p[0], [30, 30])
    f2, _ = V.accept(np.array([0, 5]), np.array([[30.0, 30], [200.0, 30]]), np.array([0.9, 0.9]), PITCH)
    assert f2.size == 0  # the second is off the pitch, so the first loses its support


def test_fill_only_gaps_and_only_short_plausible_joins_touching_a_detection():
    ball = np.full((40, 2), np.nan)
    ball[:3] = [[10, 10], [10.2, 10], [10.4, 10]]
    ball[30:] = [[50, 50]] * 10
    frames, xy = np.array([2, 7, 23]), np.array([[99.0, 99], [11.4, 10], [12.0, 10]])
    out, filled = V.fill(ball, frames, xy)
    np.testing.assert_array_equal(out[:3], ball[:3])           # ATD frames untouched
    np.testing.assert_allclose(out[5], [11.0, 10])             # line from ATD frame 2 to detection 7
    assert filled[3:8].all() and not filled[8:23].any()        # 7 -> 23 is longer than 0.6 s
    assert filled[23] and not filled[24:30].any()              # 23 -> 30: 54 m in 0.28 s, too fast
    assert not filled[:3].any() and not filled[30:].any()


def test_agreement_reports_distance_where_both_exist():
    ball = np.array([[0.0, 0], [np.nan, np.nan], [10, 10]])
    n, med, share = V.agreement(np.array([0, 1, 2]), np.array([[1.0, 0], [5, 5], [10, 14]]), ball)
    assert n == 2 and np.isclose(med, 2.5) and share == 0.5


def test_seen_runs_keeps_atd_ball_stretches_inside_segments():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from vision_segments import seen_runs
    missing = np.ones(40, bool)
    missing[2:8] = missing[10:12] = missing[30:35] = False   # runs of 6, 2 (too short) and 5 frames
    assert seen_runs([(0, 20, 1), (28, 40, 2)], missing) == [(2, 8, 1), (30, 35, 2)]
