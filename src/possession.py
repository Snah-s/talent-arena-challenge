"""Possession model: two per-frame heads + a 4-state HMM with minimum durations, Viterbi decoded.

Heads (gradient-boosted trees, trained only on the external bench):
  q = P(ball alive | x)          from the symmetric features
  r = P(home owns  | x, alive)   trained on both team orientations and averaged over them, so
                                 swapping the teams swaps H and A exactly (no home bias)
States LH, LA, DH, DA: the dead states remember who had the ball but both emit D. Each state is
expanded into a chain of sub-states so a visit lasts at least its minimum duration.
"""

import numba
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import bench
from .bench import A, D, H, X

LH, LA, DH, DA = 0, 1, 2, 3
EMIT = np.array([H, A, D, D], np.int8)
EPS = 1e-4


# -- heads ---------------------------------------------------------------------------------------

def _hgb():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                          l2_regularization=1.0, early_stopping=False,
                                          random_state=0)


def fit_heads(sym, anti, label):
    alive = label != D
    q = _hgb().fit(np.hstack([sym, np.abs(anti)]), alive)
    s, n, y = sym[alive], anti[alive], label[alive] == H
    r = _hgb().fit(np.vstack([np.hstack([s, n]), np.hstack([s, -n])]), np.concatenate([y, ~y]))
    return {"q": q, "r": r, "p_alive": float(alive.mean())}


def train(mids, control_s=bench.CONTROL_S, stride=5, tags="none"):
    """Heads and transition statistics from bench matches: every degraded copy, every stride-th
    frame, labels under the challenge's control rule. The one training path, for the bench folds
    and for the submission alike."""
    from .tagging import with_tags
    sym, anti, lab, labels, periods = [], [], [], [], []
    for mid in mids:
        m = bench.load_match(mid)
        y = bench.controlled_labels(m["label"], m["period"], control_s)
        labels.append(y)
        periods.append(m["period"])
        for s, a in with_tags(mid, tags):
            sym.append(s[::stride])
            anti.append(a[::stride])
            lab.append(y[::stride])
    heads = fit_heads(np.vstack(sym), np.vstack(anti), np.concatenate(lab))
    return heads, transition_stats(labels, periods)


def predict_heads(heads, sym, anti):
    q = heads["q"].predict_proba(np.hstack([sym, np.abs(anti)]))[:, 1]
    r_fwd = heads["r"].predict_proba(np.hstack([sym, anti]))[:, 1]
    r_swp = heads["r"].predict_proba(np.hstack([sym, -anti]))[:, 1]
    return q, 0.5 * (r_fwd + 1.0 - r_swp)


# -- transition statistics from labelled matches -------------------------------------------------

def transition_stats(labels, periods):
    """Live/dead run lengths (frames), P(live run ends in a switch), P(restart by same team)."""
    live, dead, switch, ends, same, restarts = [], [], 0, 0, 0, 0
    for label, period in zip(labels, periods):
        for half in (1, 2):
            lab = label[period == half]
            if not len(lab):
                continue
            b = np.flatnonzero(lab[1:] != lab[:-1]) + 1
            starts, stops = np.concatenate(([0], b)), np.concatenate((b, [len(lab)]))
            ks = lab[starts]
            for i, k in enumerate(ks):
                (dead if k == D else live).append(stops[i] - starts[i])
                if k != D and i + 1 < len(ks):
                    ends += 1
                    switch += ks[i + 1] != D
                if k == D and 0 < i < len(ks) - 1 and ks[i - 1] != D:
                    restarts += 1
                    same += ks[i + 1] == ks[i - 1]
    return {"live_mean": float(np.mean(live)), "dead_mean": float(np.mean(dead)),
            "live_durs": np.array(live), "dead_durs": np.array(dead),
            "p_switch": switch / ends, "p_same": same / restarts}


# -- decoding ------------------------------------------------------------------------------------

def duration_shape(durs, min_len):
    """(forced, k, p): min_len - 1 forced sub-states, then one sub-state with self-loop p so the
    mean bench duration is kept -- minimum + geometric tail. (A negative-binomial tail, k > 1,
    was tried: fitted by moments on the bench it is always k = 1, because the bench durations are
    heavier-tailed than geometric, and an Erlang tail can only make them lighter.)"""
    forced = min_len - 1
    return forced, 1, 1.0 - 1.0 / max(durs.mean() - forced, 1.5)


def chain(stats, cfg):
    """Expanded state space as flat arrays for viterbi(): per sub-state its state, self-loop cost,
    predecessor (-1 = entry) and advance cost; per state its entry, last sub-state, exit cost;
    and log P(next state | exit)."""
    shapes = [duration_shape(stats["live_durs"], cfg["min_live"])] * 2 \
        + [duration_shape(stats["dead_durs"], cfg["min_dead"])] * 2
    state_of, self_cost, pred, adv = [], [], [], []
    entry, last, exit_cost = np.zeros(4, np.int64), np.zeros(4, np.int64), np.zeros(4)
    for s, (forced, k, p) in enumerate(shapes):
        entry[s] = len(state_of)
        for i in range(forced + k):
            erlang = i >= forced
            state_of.append(s)
            self_cost.append(np.log(p) if erlang else -np.inf)
            pred.append(-1 if i == 0 else len(state_of) - 2)
            adv.append(np.log1p(-p) if erlang and i > forced else 0.0)
        last[s] = len(state_of) - 1
        exit_cost[s] = np.log1p(-p)
    dest = np.zeros((4, 4))
    dest[LH, LA] = dest[LA, LH] = stats["p_switch"]
    dest[LH, DH] = dest[LA, DA] = 1 - stats["p_switch"]
    dest[DH, LH] = dest[DA, LA] = stats["p_same"]
    dest[DH, LA] = dest[DA, LH] = 1 - stats["p_same"]
    with np.errstate(divide="ignore"):
        log_dest = np.log(dest)
    return (np.array(state_of, np.int64), np.array(self_cost), np.array(pred, np.int64),
            np.array(adv), entry, last, exit_cost, log_dest)


def log_emissions(q, r, p_alive, alpha):
    """Scaled likelihoods log p(state|x) - log p(state), tempered by alpha for frame correlation.
    (A separate temperature for the owner head was tried: the bench picks it equal to alpha.)"""
    q, r = np.clip(q, EPS, 1 - EPS), np.clip(r, EPS, 1 - EPS)
    live = np.log(q) - np.log(p_alive)
    dead = np.log1p(-q) - np.log1p(-p_alive)
    e = np.column_stack([live + np.log(2 * r), live + np.log(2 * (1 - r)), dead, dead])
    return alpha * e


@numba.njit(cache=True)
def viterbi(logE, state_of, self_cost, pred, adv, entry, last, exit_cost, log_dest):
    """MAP state path over the expanded chain built by chain(). Every state starts in its last
    sub-state (the first visit counts as already long); the last visit may be cut by the end."""
    n_t, n = logE.shape[0], len(state_of)
    V = np.full(n, -np.inf)
    for s in range(4):
        V[last[s]] = logE[0, s]
    bp = np.empty((n_t, n), np.int16)
    W = np.empty(n)
    for t in range(1, n_t):
        for j in range(n):
            s = state_of[j]
            best, arg = -np.inf, -1
            if pred[j] >= 0:
                best, arg = V[pred[j]] + adv[j], pred[j]
            else:
                for s2 in range(4):
                    if log_dest[s2, s] > -np.inf:
                        v = V[last[s2]] + exit_cost[s2] + log_dest[s2, s]
                        if v > best:
                            best, arg = v, last[s2]
            if self_cost[j] > -np.inf:
                v = V[j] + self_cost[j]
                if v >= best:
                    best, arg = v, j
            W[j] = best + logE[t, s]
            bp[t, j] = arg
        V, W = W, V
    path = np.empty(n_t, np.int64)
    j = np.argmax(V)
    for t in range(n_t - 1, -1, -1):
        path[t] = state_of[j]
        if t:
            j = bp[t, j]
    return path


def decode(q, r, period, heads, stats, cfg, force_dead=None):
    """Labels for every frame: Viterbi inside each half, X outside the halves. force_dead is an
    optional boolean mask of frames that must be dead ball (before a half's kick-off)."""
    arrays = chain(stats, cfg)
    logE = log_emissions(q, r, heads["p_alive"], cfg["alpha"])
    if force_dead is not None:
        logE[force_dead, :2] = -np.inf
    out = np.full(len(q), X, np.int8)
    for half in (1, 2):
        idx = np.flatnonzero(period == half)
        if not len(idx):
            continue
        out[idx] = EMIT[viterbi(logE[idx], *arrays)]
    return out
