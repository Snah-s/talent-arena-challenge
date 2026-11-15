"""Where the tactical camera looks, as model input.

The homography maps pitch to pixels for every frame; the image centre mapped back onto the pitch
is the camera's focus. A camera that follows play keeps pointing near the ball while the tracking
has lost it, so the focus is evidence for both the ball's position and who is around it.

The bench has no camera, so one is simulated from the true DFL ball: an operator following the
ball with a lag, plus a slowly drifting framing offset. Ranges fixed a priori, before any look at
the challenge homography, and never tuned on it.
"""

import os
import sys

import numpy as np
from scipy.signal import lfilter

from . import features as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "starter"))

FPS = 25
IMAGE_CENTRE = (960.0, 540.0)  # tactical video is 1920x1080 (DATA.md)
CAM_SYM = ("cam_speed_r25", "cam_out", "cam_dmin", "cam_ball_last")
CAM_ANTI = ("dd_cam", "cam_terr")

CAMERA_RANGES = {
    "lag_s": (0.2, 2.0),        # time constant of the camera following the ball
    "noise_m": (1.0, 10.0),     # sd of the framing offset
    "noise_tau_s": (1.0, 5.0),  # how slowly that offset drifts
}


def sample_camera(rng):
    return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in CAMERA_RANGES.items()}


def _ema(x, tau_s):
    """Exponential smoothing along axis 0, started at the first value."""
    a = 1.0 - np.exp(-1.0 / (tau_s * FPS))
    return lfilter([a], [1.0, a - 1.0], x, axis=0, zi=((1.0 - a) * x[:1]))[0]


def simulate_camera(ball, period, pitch, rng, cfg):
    """(T, 2) camera focus in metres from the true ball, per half."""
    cam = np.full((len(ball), 2), np.nan)
    a = 1.0 - np.exp(-1.0 / (cfg["noise_tau_s"] * FPS))
    for half in (1, 2):
        idx = np.flatnonzero(period == half)
        if not len(idx):
            continue
        xy = F._bfill(F._ffill(ball[idx, :2].astype(np.float64)))
        white = rng.normal(0, cfg["noise_m"] * np.sqrt((2 - a) / a), (len(idx), 2))
        cam[idx] = _ema(xy, cfg["lag_s"]) + _ema(white, cfg["noise_tau_s"])
    return np.clip(cam, -5, np.asarray(pitch) + 5).astype(np.float32)


def challenge_camera(pitch, game=None, data_dir=None):
    """(T, 2) camera focus in metres from the challenge homography; NaN where it is invalid."""
    from load_homography import load
    matrices, _, valid = load(data_dir=data_dir) if game is None else load(game, data_dir)
    cam = np.full((len(valid), 2), np.nan, np.float32)
    inv = np.linalg.inv(matrices[valid])
    p = inv @ np.array([IMAGE_CENTRE[0], IMAGE_CENTRE[1], 1.0])
    cam[valid] = (p[:, :2] / p[:, 2:3]) * np.asarray(pitch)
    return cam


def camera_features(cam, m):
    """(sym (T, 4), anti (T, 2)) from the camera focus and the (degraded) match."""
    with np.errstate(invalid="ignore"), F.warnings.catch_warnings():
        F.warnings.simplefilter("ignore", RuntimeWarning)
        P, home = m["players"], m["team"] == 0
        length, width = (float(v) for v in m["pitch"])
        dist = np.linalg.norm(P - cam[:, None, :], axis=2)
        dh = np.nanmin(np.where(home, dist, np.nan), 1)
        da = np.nanmin(np.where(~home, dist, np.nan), 1)
        last = F._ffill(m["ball"][:, :2])
        sign = F.attack_sign(m)
        sym = np.column_stack([
            F._roll(F._speed(cam), 25),
            F._out(cam, length, width),
            np.fmin(dh, da),
            np.linalg.norm(cam - last, axis=1),
        ])
        anti = np.column_stack([np.clip(dh - da, -30, 30), (cam[:, 0] - length / 2) / (length / 2) * sign])
    return sym.astype(np.float32), anti.astype(np.float32)
