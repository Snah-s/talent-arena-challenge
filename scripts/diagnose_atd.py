"""Aggregate facts about the challenge ATD, to understand the problem only (PLAN.md F0.3).

Prints summary numbers, never rows. Nothing here feeds the bench: its degradation ranges are
fixed in src/bench.py and are deliberately not calibrated to these numbers.

    python scripts/diagnose_atd.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.atd import load_atd  # noqa: E402


def runs(mask):
    """Lengths of the True runs of a boolean array."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def main():
    m = load_atd()
    graded = m["period"] > 0
    ball_seen = np.isfinite(m["ball"][:, 0])
    seen = np.isfinite(m["players"][..., 0]).sum(1)
    print("graded frames {}  ball seen {:.3f}  tracked objects/frame mean {:.1f}  K {}".format(
        graded.sum(), ball_seen[graded].mean(), seen[graded].mean(), len(m["team"])))
    for half in (1, 2):
        sel = m["period"] == half
        gaps = runs(~ball_seen[sel]) / 25.0
        print("half {}  ball seen {:.3f}  gaps {}  length s p50 {:.2f} p90 {:.2f} p99 {:.2f} "
              "max {:.1f}".format(half, ball_seen[sel].mean(), len(gaps),
                                  *np.percentile(gaps, (50, 90, 99)), gaps.max()))
    inside = m["ball"][graded & ball_seen, :2]
    off = ((inside < 0) | (inside > m["pitch"])).any(1).mean()
    print("ball positions outside the pitch rectangle: {:.4f}".format(off))


if __name__ == "__main__":
    main()
