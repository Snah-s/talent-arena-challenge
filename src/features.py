"""Per-frame features from a match dict (src/bench.py), identity-free: only team membership.

Two blocks, so the model can be made exactly symmetric in the two teams:
  sym   unchanged when home and away are swapped   -> alive/dead head, and context for the owner
  anti  negated when home and away are swapped     -> owner head
Missing values stay NaN; the gradient-boosted heads handle them natively.
"""

import warnings

import numpy as np

FPS = 25
CAP_S = 10.0  # cap for "seconds since/until the ball was seen"
SYM = ("seen", "since_s", "until_s", "bspeed", "bspeed_r12", "out_last", "out_next",
       "center_last", "corner_last", "dmin", "dmin_r25", "n10", "pspeed", "pspeed_r25",
       "pspeed_r100", "pmax_r25", "ntrack", "own", "own_r25", "abs_terr", "abs_dd_last",
       # Vidal-Codina 2022 set-piece configurations (IFAB Laws 8, 15, 16, 17), ball-free
       "touch_min", "touch_min_r25", "corner_min", "corner_min_r25", "center_min", "gk_cfg_r25",
       # Vidal-Codina duel zone; Link & Hoernig 2017 ball moving with the player / touches
       "duel3", "vrel", "vrel_r12", "btouch", "btouch_r12")
ANTI = ("dd", "dd_r12", "dd_r50", "dd_last", "dd_next", "n3", "terr", "cent", "cent_r50",
        "spread", "vdiff_r25",
        # who takes the restart; who is near where the ball vanished / reappears (pitch-control
        # flavour of Fernandez & Bornn 2018, distance only)
        "throw_team_r25", "corner_team_r25", "gk_team_r25", "ddl_now", "ddn_now",
        # Fernandez & Bornn 2018 pitch control at the ball / where it was last / next seen
        "pc_ball", "pc_ball_r12", "pc_last", "pc_next")
SYM = SYM + ("inf_last",)  # both teams' summed influence where the ball was last seen
# Longer context (bench possessions last ~11 s on average; the windows above stop at +/-4 s):
# centred +/-5 s and +/-10 s means, and past-only / future-only means whose gap marks a change.
SYM = SYM + ("pspeed_r250", "seen_r125")
ANTI = ANTI + ("dd_r125", "dd_r250", "dd_past125", "dd_fut125", "pc_last_r125", "pc_last_r250",
               "pc_past125", "pc_fut125", "cent_r250")
# Vidal-Codina 2022 touches credited to a team: the last / next touch, and net touches in context
SYM = SYM + ("ttouch_since", "ttouch_until")
ANTI = ANTI + ("tteam_last", "tteam_next", "tteam_past125", "tteam_fut125", "tteam_r250")
TOUCH_DV = 4.0  # m/s change of ball velocity that counts as a touch (fixed a priori)
TOUCH_R = 2.0   # m from the nearest player

# IFAB pitch markings, metres
GOAL_AREA = (5.5, 9.16)       # depth, half-width
PENALTY_AREA = (16.5, 20.16)
MARK_TOL = 1.0                # tracking tolerance around a marking (Vidal-Codina's epsilons)


def _roll(x, w, before=None, after=None):
    """Mean over [t - before, t + after] (default +/- w) ignoring NaN; NaN where the window holds
    no value."""
    before = w if before is None else before
    after = w if after is None else after
    ok = np.isfinite(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    n = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo, hi = np.clip(i - before, 0, len(x)), np.clip(i + after + 1, 0, len(x))
    cnt = n[hi] - n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (c[hi] - c[lo]) / cnt, np.nan)


def _ffill(x):
    ok = np.isfinite(x if x.ndim == 1 else x[:, 0])
    return x[np.maximum.accumulate(np.where(ok, np.arange(len(ok)), 0))]


def _bfill(x):
    return _ffill(x[::-1])[::-1]


def _since(ok):
    """Frames since ok was last True (0 on a True frame), capped."""
    last = np.maximum.accumulate(np.where(ok, np.arange(len(ok)), -10 ** 9))
    return np.minimum(np.arange(len(ok)) - last, CAP_S * FPS) / FPS


def _speed(p, s=3):
    """Speed in m/s from positions s frames either side; p is (T, ..., 2)."""
    v = np.full(p.shape[:-1], np.nan, np.float32)
    v[s:-s] = np.linalg.norm(p[2 * s:] - p[:-2 * s], axis=-1) / (2 * s / FPS)
    return v


def _out(b, length, width):
    """Signed distance outside the pitch rectangle: > 0 outside, -(distance to a line) inside."""
    return np.max(np.stack([-b[:, 0], b[:, 0] - length, -b[:, 1], b[:, 1] - width]), axis=0)


def pitch_control(P, vp, home, z):
    """Fernandez & Bornn 2018 (SSAC), appendix A.1, at one target point per frame.

    Each player's influence is a bivariate normal centred at position + velocity / 2, rotated to
    the direction of movement and stretched by (speed / 13 m/s)^2, with radius 4-10 m growing
    with the distance to the target point. Returns (2 * PC_home - 1, summed influence of both
    teams): the first is antisymmetric in the teams, NaN where z or every player is missing.
    """
    v = np.nan_to_num(vp)                                       # unknown velocity -> standing
    rel = z[:, None, :] - P                                     # (T, K, 2)
    dist = np.linalg.norm(rel, axis=2)
    radius = np.minimum(4 + dist ** 3 / (18 ** 3 / 6), 10)
    speed = np.linalg.norm(v, axis=2)
    ratio = np.minimum(speed ** 2 / 13 ** 2, 0.9)
    sx, sy = (radius + radius * ratio) / 2, (radius - radius * ratio) / 2
    theta = np.arctan2(v[..., 1], v[..., 0])
    c, s = np.cos(theta), np.sin(theta)

    def quad(delta):                                            # Mahalanobis^2 in player frame
        u = c * delta[..., 0] + s * delta[..., 1]
        w = -s * delta[..., 0] + c * delta[..., 1]
        return (u / sx) ** 2 + (w / sy) ** 2

    mu = P + v / 2
    influence = np.exp(-0.5 * (quad(z[:, None, :] - mu) - quad(P - mu)))  # f(z) / f(player)
    influence = np.where(np.isfinite(influence), influence, 0.0)
    ih, ia = (influence * home).sum(1), (influence * ~home).sum(1)
    known = np.isfinite(z[:, 0]) & np.isfinite(P[..., 0]).any(1)
    pc = np.where(known, 2.0 / (1.0 + np.exp(-(ih - ia))) - 1.0, np.nan)
    return pc, np.where(known, ih + ia, np.nan)


def attack_sign(m):
    """+1 where home attacks towards +x, per half, from where each side's deepest player stands."""
    x, home = m["players"][..., 0], m["team"] == 0
    sign = np.ones(len(x), np.float32)
    for half in (1, 2):
        sel = m["period"] == half
        if sel.any():
            h = np.nanmedian(np.nanmin(x[sel][:, home], axis=1))
            a = np.nanmedian(np.nanmin(x[sel][:, ~home], axis=1))
            sign[sel] = 1.0 if h < a else -1.0
    return sign


def features(m):
    """(sym, anti) float32 arrays of shape (T, len(SYM)) and (T, len(ANTI))."""
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return _features(m)


def _features(m):
    P, B = m["players"], m["ball"][:, :2]
    length, width = (float(v) for v in m["pitch"])
    home = m["team"] == 0
    seen = np.isfinite(B[:, 0])
    sign = attack_sign(m)
    nan = np.float32(np.nan)

    d = np.linalg.norm(P - B[:, None, :], axis=2)                       # (T, K)
    dh = np.nanmin(np.where(home, d, nan), axis=1)
    da = np.nanmin(np.where(~home, d, nan), axis=1)
    dd = np.clip(dh - da, -30, 30)
    n3 = np.where(seen, (d[:, home] < 3).sum(1) - (d[:, ~home] < 3).sum(1), np.nan)
    n10 = np.where(seen, (d < 10).sum(1), np.nan)
    dmin = np.fmin(dh, da)

    last, nxt = _ffill(B), _bfill(B)
    centre = np.array((length / 2, width / 2), np.float32)
    corners = np.array([(0, 0), (0, width), (length, 0), (length, width)], np.float32)
    corner_last = np.min(np.linalg.norm(last[:, None, :] - corners, axis=2), axis=1)
    bspeed = _speed(B, 2)

    v = _speed(P)
    xs = (P[..., 0] - length / 2) * sign[:, None]                        # along home's attack
    tracked = np.isfinite(P[..., 0])
    # at a kick-off players stand on the halfway line and the taker at the ball: allow 3 m
    own_h = (xs[:, home] < 3).sum(1) / np.maximum(tracked[:, home].sum(1), 1)
    own_a = (xs[:, ~home] > -3).sum(1) / np.maximum(tracked[:, ~home].sum(1), 1)
    own = np.minimum(own_h, own_a)
    terr = (last[:, 0] - length / 2) / (length / 2) * sign
    cent = np.nanmean(xs, axis=1) / (length / 2)
    spread = (np.log(np.nanstd(P[:, home, 0], 1) * np.nanstd(P[:, home, 1], 1) + 1)
              - np.log(np.nanstd(P[:, ~home, 0], 1) * np.nanstd(P[:, ~home, 1], 1) + 1))
    vdiff = np.nanmean(v[:, home], 1) - np.nanmean(v[:, ~home], 1)
    pspeed = np.nanmean(v, 1)
    dd_last = _ffill(dd)

    # -- set-piece configurations, from players only --
    side = np.where(home, 1.0, -1.0)                                        # team sign per column
    x, y = P[..., 0], P[..., 1]
    inf = np.float32(np.inf)
    touch = np.minimum(y, width - y)                                      # < 0 means beyond the line
    touch_min = np.nanmin(touch, axis=1)
    near_touch = np.where(np.isnan(touch), inf, touch).argmin(1)
    throw_team = np.where(touch_min < MARK_TOL, side[near_touch], 0.0)
    corner_d = np.min(np.linalg.norm(P[:, :, None, :] - corners, axis=3), axis=2)
    corner_min = np.nanmin(corner_d, axis=1)
    near_corner = np.where(np.isnan(corner_d), inf, corner_d).argmin(1)
    corner_team = np.where(corner_min < 2 * MARK_TOL, side[near_corner], 0.0)
    center_min = np.nanmin(np.linalg.norm(P - centre, axis=2), axis=1)
    goal_x = np.where(sign > 0, 0.0, length)[:, None]                    # home defends this end
    gk = []
    for defending, goal in ((home, goal_x), (~home, length - goal_x)):
        dx, dy = np.abs(x - goal), np.abs(y - width / 2)
        keeper = ((dx <= GOAL_AREA[0] + MARK_TOL) & (dy <= GOAL_AREA[1] + MARK_TOL) & defending).any(1)
        intruders = ((dx <= PENALTY_AREA[0]) & (dy <= PENALTY_AREA[1]) & ~defending).sum(1)
        gk.append(keeper & (intruders == 0))                              # Law 16 goal-kick shape
    gk_cfg = (gk[0] | gk[1]).astype(np.float32)
    gk_team = gk[0].astype(np.float32) - gk[1].astype(np.float32)

    # -- duel and control near the ball --
    duel3 = np.where(seen, np.minimum((d[:, home] < 3).sum(1), (d[:, ~home] < 3).sum(1)), np.nan)
    vp = np.full(P.shape, np.nan, np.float32)
    vp[3:-3] = (P[6:] - P[:-6]) / (6 / FPS)
    vb = np.full(B.shape, np.nan, np.float32)
    vb[2:-2] = (B[4:] - B[:-4]) / (4 / FPS)
    nearest = np.where(np.isnan(d), inf, d).argmin(1)
    vrel = np.linalg.norm(vb - vp[np.arange(len(vp)), nearest], axis=1)
    vrel = np.where(np.isfinite(dmin), vrel, np.nan)
    btouch = np.full(len(B), np.nan, np.float32)
    btouch[3:-3] = np.linalg.norm(vb[6:] - vb[:-6], axis=1)               # change of ball velocity
    touched = (btouch > TOUCH_DV) & (dmin < TOUCH_R)
    tteam = np.where(touched, side[nearest], 0.0)                         # +1 home touch, -1 away
    tk = np.where(touched, tteam, np.nan)

    pc_ball, _ = pitch_control(P, vp, home, B)
    pc_last, inf_last = pitch_control(P, vp, home, last)

    # -- who is near where the ball was last seen / will be seen next, with current players --
    def team_gap(ref):
        dist = np.linalg.norm(P - ref[:, None, :], axis=2)
        return np.clip(np.nanmin(np.where(home, dist, nan), 1)
                       - np.nanmin(np.where(~home, dist, nan), 1), -30, 30)

    cols = {
        "seen": seen.astype(np.float32), "since_s": _since(seen), "until_s": _since(seen[::-1])[::-1],
        "bspeed": bspeed, "bspeed_r12": _roll(bspeed, 12),
        "out_last": _out(last, length, width), "out_next": _out(nxt, length, width),
        "center_last": np.linalg.norm(last - centre, axis=1), "corner_last": corner_last,
        "dmin": dmin, "dmin_r25": _roll(dmin, 25), "n10": n10,
        "pspeed": pspeed, "pspeed_r25": _roll(pspeed, 25), "pspeed_r100": _roll(pspeed, 100),
        "pmax_r25": _roll(np.nanmax(v, 1), 25), "ntrack": tracked.sum(1).astype(np.float32),
        "own": own, "own_r25": _roll(own, 25), "abs_terr": np.abs(terr),
        "abs_dd_last": np.abs(dd_last),
        "dd": dd, "dd_r12": _roll(dd, 12), "dd_r50": _roll(dd, 50), "dd_last": dd_last,
        "dd_next": _bfill(dd), "n3": n3, "terr": terr, "cent": cent, "cent_r50": _roll(cent, 50),
        "spread": _roll(spread, 25), "vdiff_r25": _roll(vdiff, 25),
        "touch_min": touch_min, "touch_min_r25": _roll(touch_min, 25),
        "corner_min": corner_min, "corner_min_r25": _roll(corner_min, 25),
        "center_min": center_min, "gk_cfg_r25": _roll(gk_cfg, 25),
        "duel3": duel3, "vrel": vrel, "vrel_r12": _roll(vrel, 12),
        "btouch": btouch, "btouch_r12": _roll(btouch, 12),
        "throw_team_r25": _roll(throw_team, 25), "corner_team_r25": _roll(corner_team, 25),
        "gk_team_r25": _roll(gk_team, 25), "ddl_now": team_gap(last), "ddn_now": team_gap(nxt),
        "pc_ball": pc_ball, "pc_ball_r12": _roll(pc_ball, 12), "pc_last": pc_last,
        "pc_next": pitch_control(P, vp, home, nxt)[0], "inf_last": inf_last,
        "pspeed_r250": _roll(pspeed, 250), "seen_r125": _roll(seen.astype(np.float32), 125),
        "dd_r125": _roll(dd, 125), "dd_r250": _roll(dd, 250),
        "dd_past125": _roll(dd, 0, 125, 0), "dd_fut125": _roll(dd, 0, 0, 125),
        "pc_last_r125": _roll(pc_last, 125), "pc_last_r250": _roll(pc_last, 250),
        "pc_past125": _roll(pc_last, 0, 125, 0), "pc_fut125": _roll(pc_last, 0, 0, 125),
        "cent_r250": _roll(cent, 250),
        "ttouch_since": _since(touched), "ttouch_until": _since(touched[::-1])[::-1],
        "tteam_last": _ffill(tk), "tteam_next": _bfill(tk),
        "tteam_past125": _roll(tteam, 0, 125, 0), "tteam_fut125": _roll(tteam, 0, 0, 125),
        "tteam_r250": _roll(tteam, 250),
    }
    sym = np.column_stack([cols[k] for k in SYM]).astype(np.float32)
    anti = np.column_stack([cols[k] for k in ANTI]).astype(np.float32)
    return sym, anti


def swap_teams(m):
    """The same match with home and away exchanged (for the symmetry test)."""
    out = dict(m)
    out["team"] = (1 - m["team"]).astype(np.int8)
    return out
