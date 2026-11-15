import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import tagging as T  # noqa: E402
from src.bench import A, D, H  # noqa: E402

EXACT = {"recall": 1.0, "false_per_10min": 0.0, "jitter_s": 0.0, "half_width_s": 2.0,
         "team_na": 0.0, "team_wrong": 0.0}


def test_pseudo_tags_mark_restarts_with_the_restarting_team():
    label = np.array([D] * 100 + [H] * 200 + [D] * 100 + [A] * 200, np.int8)
    period = np.ones(len(label), np.int8)
    tags = T.pseudo_tags(label, period, np.random.default_rng(0), EXACT)
    assert tags == [(50, 150, H), (350, 450, A)]


def test_tag_features():
    sym, anti = T.tag_features([(50, 150, 0), (300, 400, -1)], 500)
    assert sym[100, 0] == 1 and sym[200, 0] == 0
    assert sym[100, 1] == 0.0 and sym[125, 1] == 1.0          # seconds from the nearest centre
    assert np.isnan(T.tag_features([(0, 10, 0)], 1000)[0][900, 1])  # beyond CAP_S
    assert anti[100, 0] == 1.0 and anti[350, 0] == 0.0 and anti[200, 0] == 0.0
