"""Inputs of the neural owner head (plan v3, E4), numpy only so they build without torch.

Per step at 5 fps: up to M players per team as tokens (nearest first to the ball, or to where it
was last seen), and one ball token. Only team membership is used, never track identity, so the
same code serves the DFL bench and the challenge ATD. Swapping the teams swaps the two token
groups and nothing else, which is what makes the head's team symmetry exact (owner_nn.predict).
"""

import numpy as np

STEP = 5             # 25 fps -> 5 fps
DT = STEP / 25.0     # seconds per step
M = 14               # players kept per team (duplicates and fragments push the ATD above 11)
CAP_S = 10.0
TOK_F, BALL_F = 7, 6
# feature columns negated by a mirror of the pitch along each axis (augmentation)
TOK_X, TOK_Y = (0, 2, 4), (1, 3, 5)
BALL_X, BALL_Y = (0, 4), (1, 5)


def _ffill(x):
    ok = np.isfinite(x[:, 0])
    return x[np.maximum.accumulate(np.where(ok, np.arange(len(ok)), 0))]


def _vel(p):
    """Central difference in m/s along axis 0; 0 where unknown."""
    v = np.zeros(p.shape, np.float32)
    v[1:-1] = (p[2:] - p[:-2]) / (2 * DT)
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


def downsample(m):
    """The 5 fps view of a match dict: (players, team, ball xy, period, pitch)."""
    return (m["players"][::STEP], m["team"], m["ball"][::STEP, :2], m["period"][::STEP],
            np.asarray(m["pitch"], np.float32))


def build_inputs(players, team, ball, pitch):
    """(tok (S, 2, M, TOK_F), mask (S, 2, M), ball (S, BALL_F)) float32 / bool, at 5 fps."""
    S = len(players)
    L, W = (float(v) for v in pitch)
    half = np.array((L / 2, W / 2), np.float32)
    seen = np.isfinite(ball[:, 0])
    ref = ball.copy()
    if seen.any():
        ref = _ffill(_ffill(ref[::-1])[::-1])  # before the first sighting: the first sighting
    ref[~np.isfinite(ref[:, 0])] = half
    last = np.maximum.accumulate(np.where(seen, np.arange(S), -10 ** 9))
    since = np.minimum((np.arange(S) - last) * DT, CAP_S) / CAP_S
    vel = _vel(players)

    tok = np.zeros((S, 2, M, TOK_F), np.float32)
    mask = np.zeros((S, 2, M), bool)
    for g in (0, 1):
        P, V = players[:, team == g], vel[:, team == g]
        k = min(M, P.shape[1])
        if not k:
            continue
        d = np.linalg.norm(P - ref[:, None], axis=2)
        order = np.argsort(np.where(np.isfinite(d), d, np.inf), axis=1, kind="stable")[:, :k]
        Pk = np.take_along_axis(P, order[..., None], 1)
        Vk = np.take_along_axis(V, order[..., None], 1)
        dk = np.take_along_axis(d, order, 1)
        ok = np.isfinite(dk)
        rel = (Pk - ref[:, None]) / 20.0
        f = np.concatenate([(Pk - half) / half, Vk / 10.0, rel, dk[..., None] / 20.0], axis=2)
        tok[:, g, :k] = np.where(ok[..., None], np.nan_to_num(f), 0.0)
        mask[:, g, :k] = ok
    bv = _vel(ball)
    ballf = np.column_stack([(ref - half) / half, seen, since, bv / 20.0]).astype(np.float32)
    return tok, mask, ballf
