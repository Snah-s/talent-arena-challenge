"""Second stage: review each decoded run (spell) as a whole.

The Viterbi path cuts a half into runs of one label. Each run gets features aggregated over all its
frames (duration, head outputs, mean per-frame features, its neighbours), and a classifier trained
on out-of-fold runs of other bench matches says which label (H/A/D) the run really carries. The
classifier is team-symmetric like the owner head: trained on both orientations (anti features
negated, H and A swapped) and averaged over them.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .bench import A, D, H

SWAP = np.array([A, H, D])  # class index after exchanging the teams


def runs(pred, period):
    """(start, stop) of every run of one label inside each half."""
    out = []
    for half in (1, 2):
        idx = np.flatnonzero(period == half)
        if not len(idx):
            continue
        lab = pred[idx]
        b = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        out += [(idx[0] + s, idx[0] + e) for s, e in zip(np.concatenate(([0], b)),
                                                         np.concatenate((b, [len(lab)])))]
    return np.array(out, np.int64).reshape(-1, 2)


def _seg_mean(x, starts):
    ok = np.isfinite(x)
    s = np.add.reduceat(np.where(ok, x, 0.0), starts, axis=0)
    n = np.add.reduceat(ok, starts, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / n, np.nan)


def run_features(pred, period, q, r, sym, anti):
    """(runs, sym (R, ·), anti (R, ·)) -- anti is negated by a team swap, sym is not."""
    rr = runs(pred, period)
    st, en = rr[:, 0], rr[:, 1]
    # reduceat needs contiguous coverage: aggregate over the whole array, then keep run segments
    cuts = np.unique(np.concatenate((st, en[en < len(pred)])))
    pos = np.searchsorted(cuts, st)
    mean = lambda x: _seg_mean(np.asarray(x, np.float64), cuts)[pos]
    lab = pred[st]
    team = np.where(lab == H, 1.0, np.where(lab == A, -1.0, 0.0))
    live = (lab != D).astype(np.float64)
    dur = np.log((en - st) / 25.0)
    rc = np.clip(r, 1e-4, 1 - 1e-4)

    same_half = np.concatenate(([False], period[st[1:]] == period[st[:-1]]))
    def prev(v):
        return np.where(same_half, np.concatenate(([np.nan], v[:-1])), np.nan)
    def nxt(v):
        return np.where(np.concatenate((same_half[1:], [False])), np.concatenate((v[1:], [np.nan])), np.nan)

    s = np.column_stack([dur, live, mean(q), mean(q > 0.5), prev(dur), nxt(dur), prev(live), nxt(live),
                         mean(np.abs(2 * r - 1)), mean(sym)])
    a = np.column_stack([team, mean(2 * r - 1), mean(np.log(rc / (1 - rc))), mean(2.0 * (r > 0.5) - 1),
                         prev(team), nxt(team), mean(anti)])
    return rr, s, a


def _clf():
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=20, l2_regularization=1.0,
                                          early_stopping=False, random_state=0)


def fit(tables):
    """tables: [(runs, sym, anti, truth per frame)] -> classifier. Target = majority true label."""
    S, N, y = [], [], []
    for rr, s, a, truth in tables:
        y.append(np.array([np.bincount(truth[b:e], minlength=3)[:3].argmax() for b, e in rr]))
        S.append(s)
        N.append(a)
    S, N, y = np.vstack(S), np.vstack(N), np.concatenate(y)
    return _clf().fit(np.vstack([np.hstack([S, N]), np.hstack([S, -N])]), np.concatenate([y, SWAP[y]]))


def proba(clf, s, a):
    """(R, 3) P(H, A, D) per run, exactly symmetric in the teams."""
    fwd = np.zeros((len(s), 3))
    swp = np.zeros((len(s), 3))
    fwd[:, clf.classes_] = clf.predict_proba(np.hstack([s, a]))
    swp[:, clf.classes_] = clf.predict_proba(np.hstack([s, -a]))
    return 0.5 * (fwd + swp[:, SWAP])


def relabel(pred, rr, p, owner_only=False):
    """New frame labels: each run takes its most probable label (owner_only: live runs may only
    switch H <-> A; dead runs are kept)."""
    out = pred.copy()
    for (b, e), pk in zip(rr, p):
        if owner_only:
            if pred[b] != D:
                out[b:e] = H if pk[H] >= pk[A] else A
        else:
            out[b:e] = int(pk.argmax())
    return out
