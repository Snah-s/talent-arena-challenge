import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.diagnose_bench import boundary_kind, merged_across  # noqa: E402
from src.bench import A, H  # noqa: E402


def test_boundary_kind():
    assert boundary_kind(100, 100, H, A) == (0, 0)
    assert boundary_kind(150, 100, H, H) == (1, 0)
    assert boundary_kind(200, 100, H, A) == (2, 1)


def test_merged_across():
    assert merged_across(np.array([60]), np.array([30]), 100, 50)
    assert not merged_across(np.array([60, 0]), np.array([0, 30]), 100, 50)


def test_vision_segments_pad_merge_and_stay_in_half():
    from scripts.vision_segments import segments
    period = np.array([1] * 20 + [2] * 20, np.int8)
    missing = np.zeros(40, bool)
    missing[[2, 3, 4, 8, 9, 10, 15]] = True
    missing[18:23] = True
    assert segments(missing, period, 3, 2) == [(0, 13, 1), (20, 25, 2)]
