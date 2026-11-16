"""From ball candidates in pixels (src/ball_detect.py) to ball positions that fill the ATD's gaps.

numpy only, runs locally next to the homography. Every rule and constant is fixed a priori from
the laws of the game and generic detector behaviour; none is tuned on the challenge video, ATD or
ground truth. The ATD ball is never overwritten: only frames where it is missing are filled.
"""

import numpy as np

FPS = 25
# No score cut here: src/ball_detect.py already keeps only WASB blobs above its own 0.5 threshold;
# the score (the blob's heatmap sum, WASB's confidence) only ranks candidates within a frame.
PITCH_MARGIN_M = 2.0   # a ball just over a line is still on camera
SPOT_R = 1.0           # m: a candidate this close to the centre or a penalty spot is the paint
PENALTY_SPOT_M = 11.0  # IFAB Law 1
MAX_SPEED = 40.0       # m/s, above the hardest shots
SUPPORT_S = 0.4        # a candidate needs another one within this time, at a reachable distance
MAX_INTERP_S = 0.6     # longest stretch filled by joining two anchors with a line


def to_pitch(frames, cx, cy, H, pitch):
    """Pixels -> metres with each frame's homography (normalised pitch -> pixels, DATA.md)."""
    inv = np.linalg.inv(H[frames])
    q = np.einsum("nij,nj->ni", inv, np.column_stack([cx, cy, np.ones(len(cx))]))
    return q[:, :2] / q[:, 2:3] * np.asarray(pitch, np.float64)


def _spots(pitch):
    L, W = (float(v) for v in pitch)
    return np.array([(L / 2, W / 2), (PENALTY_SPOT_M, W / 2), (L - PENALTY_SPOT_M, W / 2)])


def accept(frames, xy, score, pitch):
    """(frames, xy) of accepted candidates, at most one per frame: the best-scoring plausible one,
    kept only if another accepted-plausible candidate is reachable within SUPPORT_S."""
    L, W = (float(v) for v in pitch)
    near_spot = (np.linalg.norm(xy[:, None] - _spots(pitch), axis=2) < SPOT_R).any(1)
    inside = ((xy[:, 0] > -PITCH_MARGIN_M) & (xy[:, 0] < L + PITCH_MARGIN_M)
              & (xy[:, 1] > -PITCH_MARGIN_M) & (xy[:, 1] < W + PITCH_MARGIN_M))
    ok = inside & ~near_spot & np.isfinite(xy).all(1)
    f, p, s = frames[ok], xy[ok], score[ok]
    order = np.lexsort((-s, f))                 # by frame, best score first
    f, p = f[order], p[order]
    first = np.r_[True, f[1:] != f[:-1]]
    f, p = f[first], p[first]
    keep = np.zeros(len(f), bool)
    win = int(round(SUPPORT_S * FPS))
    for i in range(len(f)):
        lo, hi = np.searchsorted(f, f[i] - win), np.searchsorted(f, f[i] + win, "right")
        for j in range(lo, hi):
            dt = abs(f[j] - f[i]) / FPS
            if j != i and np.linalg.norm(p[j] - p[i]) <= MAX_SPEED * dt + 1.0:
                keep[i] = True
                break
    return f[keep], p[keep]


def fill(ball, frames, xy):
    """(ball copy with gaps filled, mask of filled frames). Accepted candidates fill their own
    frame where the ATD has no ball; a stretch between two anchors (ATD ball or candidate) is
    joined linearly if at least one end is a candidate, it is at most MAX_INTERP_S long and the
    implied speed is at most MAX_SPEED."""
    out = ball.copy()
    missing = ~np.isfinite(ball[:, 0])
    put = missing[frames]
    out[frames[put]] = xy[put]
    is_det = np.zeros(len(ball), bool)
    is_det[frames[put]] = True
    anchors = np.flatnonzero(np.isfinite(out[:, 0]))
    for a, b in zip(anchors[:-1], anchors[1:]):
        gap = b - a
        if gap < 2 or gap > MAX_INTERP_S * FPS or not (is_det[a] or is_det[b]):
            continue
        if np.linalg.norm(out[b] - out[a]) > MAX_SPEED * gap / FPS:
            continue
        w = (np.arange(1, gap) / gap)[:, None]
        out[a + 1:b] = out[a] + w * (out[b] - out[a])
    return out, missing & np.isfinite(out[:, 0])


def agreement(frames, xy, ball):
    """Where the ATD also has the ball: (n, median distance m, share within 3 m). A pipeline
    check (frame alignment, homography convention), never used to choose anything."""
    seen = np.isfinite(ball[frames, 0])
    if not seen.any():
        return 0, float("nan"), float("nan")
    d = np.linalg.norm(xy[seen] - ball[frames[seen]], axis=1)
    return int(seen.sum()), float(np.median(d)), float((d < 3.0).mean())
