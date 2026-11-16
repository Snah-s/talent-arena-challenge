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


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def blend_owner(r, r_nn, w):
    """The HGB owner head and the neural one (src/owner_nn.py) averaged in log-odds, weight w on
    r_nn (plan v3 E4). Keeps the team symmetry: swapping the teams maps the result to 1 - p."""
    return 1.0 / (1.0 + np.exp(-((1 - w) * _logit(r) + w * _logit(r_nn))))


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


@numba.njit(cache=True)
def _lse(a, b):
    if a < b:
        a, b = b, a
    if b == -np.inf:
        return a
    return a + np.log1p(np.exp(b - a))


@numba.njit(cache=True)
def _draw(w, k):
    """Index among the first k log-weights of w, drawn proportionally (never a zero weight)."""
    m = w[0]
    for i in range(1, k):
        m = max(m, w[i])
    tot = 0.0
    for i in range(k):
        tot += np.exp(w[i] - m)
    u = np.random.random() * tot
    pick = -1
    for i in range(k):
        e = np.exp(w[i] - m)
        if e > 0:
            pick = i
            u -= e
            if u < 0:
                break
    return pick


@numba.njit(cache=True)
def sample_paths(logE, state_of, self_cost, pred, adv, entry, last, exit_cost, log_dest,
                 n_samples, seed):
    """Forward filtering, backward sampling over the same chain as viterbi() (same start and
    transition rules): n_samples state paths drawn from the posterior, seeded."""
    np.random.seed(seed)
    n_t, n = logE.shape[0], len(state_of)
    Fw = np.full((n_t, n), -np.inf)
    for s in range(4):
        Fw[0, last[s]] = logE[0, s]
    for t in range(1, n_t):
        top = -np.inf
        for j in range(n):
            s = state_of[j]
            acc = -np.inf
            if pred[j] >= 0:
                acc = Fw[t - 1, pred[j]] + adv[j]
            else:
                for s2 in range(4):
                    if log_dest[s2, s] > -np.inf:
                        acc = _lse(acc, Fw[t - 1, last[s2]] + exit_cost[s2] + log_dest[s2, s])
            if self_cost[j] > -np.inf:
                acc = _lse(acc, Fw[t - 1, j] + self_cost[j])
            Fw[t, j] = acc + logE[t, s]
            top = max(top, Fw[t, j])
        for j in range(n):
            Fw[t, j] -= top  # rescale; backward draws only compare weights within one frame
    paths = np.empty((n_samples, n_t), np.int8)
    w, idx = np.empty(6), np.empty(6, np.int64)
    for k in range(n_samples):
        j = _draw(Fw[n_t - 1], n)
        paths[k, n_t - 1] = state_of[j]
        for t in range(n_t - 1, 0, -1):
            s, m = state_of[j], 0
            if self_cost[j] > -np.inf:
                idx[m], w[m] = j, Fw[t - 1, j] + self_cost[j]
                m += 1
            if pred[j] >= 0:
                idx[m], w[m] = pred[j], Fw[t - 1, pred[j]] + adv[j]
                m += 1
            else:
                for s2 in range(4):
                    if log_dest[s2, s] > -np.inf:
                        idx[m] = last[s2]
                        w[m] = Fw[t - 1, last[s2]] + exit_cost[s2] + log_dest[s2, s]
                        m += 1
            j = idx[_draw(w, m)]
            paths[k, t - 1] = state_of[j]
    return paths


MBR_CAND, MBR_REF = 8, 32


def mbr(viterbi_codes, sample_codes, n_cand=MBR_CAND):
    """Of the Viterbi path and the first n_cand samples, the one with the highest mean spellF1
    against the other samples: minimum Bayes risk for the board metric (Goel & Byrne 2000).
    Ties keep Viterbi."""
    runs = [(0, len(viterbi_codes))]
    cands = [viterbi_codes] + list(sample_codes[:n_cand])
    gain = [np.mean([bench.pm.spell_scores(ref, c, runs)["mean_f1"] for ref in sample_codes[n_cand:]])
            for c in cands]
    return cands[int(np.argmax(gain))]


def decode(q, r, period, heads, stats, cfg, force_dead=None):
    """Labels for every frame: Viterbi inside each half, X outside the halves. force_dead is an
    optional boolean mask of frames that must be dead ball (before a half's kick-off).
    With cfg["mbr"], each half takes mbr() over the Viterbi path and posterior samples drawn with
    emissions tempered by cfg["mbr_alpha"] (default alpha); seeded per half, so deterministic."""
    arrays = chain(stats, cfg)

    def emissions(alpha):
        logE = log_emissions(q, r, heads["p_alive"], alpha)
        if force_dead is not None:
            logE[force_dead, :2] = -np.inf
        return logE

    logE = emissions(cfg["alpha"])
    logS = emissions(cfg.get("mbr_alpha", cfg["alpha"])) if cfg.get("mbr") else None
    out = np.full(len(q), X, np.int8)
    for half in (1, 2):
        idx = np.flatnonzero(period == half)
        if not len(idx):
            continue
        codes = EMIT[viterbi(logE[idx], *arrays)]
        if logS is not None:
            codes = mbr(codes, EMIT[sample_paths(logS[idx], *arrays, MBR_CAND + MBR_REF, half)])
        out[idx] = codes
    return out
