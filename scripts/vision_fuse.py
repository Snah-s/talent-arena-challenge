"""Colab detections + local homography -> ball positions for the ATD's gaps.

    python scripts/vision_fuse.py [--det external/vision/out] [--check-only]

Reads OUT/detections.csv and OUT/status.json from src/ball_detect.py, refuses them if the video
decode did not match the reference frame count, projects every candidate to metres with that
frame's homography, keeps the plausible ones (src/vision.py, fixed rules) and writes
external/vision/ball_vision.npz for `make_submission.py --ball-fill`. Prints aggregates only.

The agreement with the ATD ball where both exist is a pipeline check (frame alignment, homography
convention); it stops the run if it fails, and it is never used to choose anything.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import vision as V  # noqa: E402
from src.atd import load_atd  # noqa: E402
from load_homography import load as load_homography  # noqa: E402  (starter/ on the path via src.atd)

CHECK_MEDIAN_M = 5.0
CHECK_MIN_N = 30
OUT = os.path.join(ROOT, "external", "vision", "ball_vision.npz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--det", default=os.path.join(ROOT, "external", "vision", "out"))
    parser.add_argument("--check-only", action="store_true",
                        help="run the pipeline check and stop (first pass over segments_pads.csv)")
    args = parser.parse_args()
    with open(os.path.join(args.det, "status.json")) as handle:
        status = json.load(handle)
    if not status.get("frames_match"):
        sys.exit("refused: the Colab decode gave {} frames, the reference has {}".format(
            status.get("decoded_frames"), status.get("expected_frames")))

    m = load_atd()
    ball = m["ball"][:, :2].astype(np.float64)
    H, _, valid = load_homography(source="tactical")
    det = pd.read_csv(os.path.join(args.det, "detections.csv")).drop_duplicates()
    det = det[valid[det["frame"].values]]
    frames = det["frame"].values.astype(np.int64)
    xy = V.to_pitch(frames, det["cx"].values, det["cy"].values, H, m["pitch"])
    f, p = V.accept(frames, xy, det["score"].values, m["pitch"])

    n_chk, med, share = V.agreement(f, p, ball)
    print("check vs ATD ball where both exist: n {}  median {:.2f} m  within 3 m {:.2f}".format(
        n_chk, med, share))
    if n_chk < CHECK_MIN_N:
        sys.exit("refused: only {} accepted detections overlap the ATD ball (< {})".format(
            n_chk, CHECK_MIN_N))
    if med > CHECK_MEDIAN_M:
        sys.exit("refused: detections sit {:.1f} m from the ATD ball; alignment or homography "
                 "convention is wrong, or the detector does not see the ball".format(med))
    if args.check_only:
        print("check passed ({} detection rows, {} accepted); run the full segments next".format(
            len(det), len(f)))
        return

    filled_ball, filled = V.fill(ball, f, p)
    graded = m["period"] > 0
    gap = graded & ~np.isfinite(ball[:, 0])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, ball=filled_ball.astype(np.float32), filled=filled)
    print("candidates {}  frames with a candidate {}  accepted {}".format(
        len(det), len(np.unique(frames)), len(f)))
    print("graded frames without ATD ball {}  filled {} ({:.1%})  ball seen on graded frames "
          "{:.3f} -> {:.3f}".format(gap.sum(), (filled & gap).sum(), (filled & gap).sum() / gap.sum(),
                                    np.isfinite(ball[graded, 0]).mean(),
                                    np.isfinite(filled_ball[graded, 0]).mean()))
    print("wrote {}".format(os.path.relpath(OUT, ROOT)))


if __name__ == "__main__":
    main()
