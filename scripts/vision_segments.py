"""Video segments where a ball detector should run: the ATD's ball gaps inside the halves.

    python scripts/vision_segments.py

Rule fixed a priori (PLAN.md Phase 2-3), not tuned on the challenge: gaps of at least MIN_GAP_S
(shorter ones are left to interpolation), padded by PAD_S on each side so detections overlap the
ATD ball for an automatic consistency check, clipped to the half and merged where they touch.
Writes external/vision/segments.csv (video frames, half-open [start, stop)),
external/vision/segments_pads.csv (only the stretches of those segments where the ATD has the
ball: the frames of vision_fuse's pipeline check, run first so a failing detector costs little)
and external/vision/vision_bundle.zip (both + src/ball_detect.py) for the Colab notebook
notebooks/vision_ball.ipynb. Prints aggregates only.
"""

import os
import sys
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.atd import load_atd  # noqa: E402

FPS = 25
MIN_GAP_S = 1.0
PAD_S = 1.0
OUT = os.path.join(ROOT, "external", "vision", "segments.csv")
PADS = os.path.join(ROOT, "external", "vision", "segments_pads.csv")


def segments(missing, period, min_gap, pad):
    """[(start, stop, half)] padded gaps of missing >= min_gap frames, inside each half, merged."""
    out = []
    for half in (1, 2):
        idx = np.flatnonzero(period == half)
        if not len(idx):
            continue
        lo, hi = idx[0], idx[-1] + 1
        edges = np.diff(np.concatenate(([0], missing[lo:hi].astype(np.int8), [0])))
        starts, stops = np.flatnonzero(edges == 1) + lo, np.flatnonzero(edges == -1) + lo
        keep = stops - starts >= min_gap
        for a, b in zip(np.maximum(starts[keep] - pad, lo), np.minimum(stops[keep] + pad, hi)):
            if out and out[-1][2] == half and a <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], int(b)), half)
            else:
                out.append((int(a), int(b), half))
    return out


def seen_runs(segs, missing, min_len=3):
    """[(start, stop, half)] runs inside the segments where the ATD has the ball, at least min_len
    frames (one detector triplet)."""
    out = []
    for a, b, half in segs:
        edges = np.diff(np.concatenate(([0], (~missing[a:b]).astype(np.int8), [0])))
        out += [(a + int(s), a + int(t), half) for s, t in
                zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)) if t - s >= min_len]
    return out


def write(path, segs):
    with open(path, "w") as handle:
        handle.write("start_frame,stop_frame,half,start_s,stop_s\n")
        handle.writelines("{},{},{},{:.2f},{:.2f}\n".format(a, b, h, a / FPS, b / FPS) for a, b, h in segs)


def main():
    m = load_atd()
    missing = ~np.isfinite(m["ball"][:, 0])
    segs = segments(missing, m["period"], int(MIN_GAP_S * FPS), int(PAD_S * FPS))
    pads = seen_runs(segs, missing)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    write(OUT, segs)
    write(PADS, pads)
    bundle = os.path.join(os.path.dirname(OUT), "vision_bundle.zip")
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(OUT, "segments.csv")
        z.write(PADS, "segments_pads.csv")
        z.write(os.path.join(ROOT, "src", "ball_detect.py"), "src/ball_detect.py")
    frames = sum(b - a for a, b, _ in segs)
    graded = int((m["period"] > 0).sum())
    print("wrote {}: {} segments, {} frames ({:.1f} min of video, {:.0%} of graded frames)".format(
        os.path.relpath(OUT, ROOT), len(segs), frames, frames / FPS / 60, frames / graded))
    print("wrote {}: {} runs, {} frames with the ATD ball".format(
        os.path.relpath(PADS, ROOT), len(pads), sum(b - a for a, b, _ in pads)))


if __name__ == "__main__":
    main()
