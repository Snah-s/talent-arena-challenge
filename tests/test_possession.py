import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import features as F  # noqa: E402
from src import possession as P  # noqa: E402


def _score(path, logE, mins, logT, logStay):
    """Score of one path under the chain's rules: a run may only be left once it has lasted its
    minimum (the first run counts as already long), and only a long run pays the stay cost."""
    score, k, first = logE[0, path[0]], 0, True  # k: position of frame t-1 within its run
    for t in range(1, len(path)):
        long_enough = first or k >= mins[path[t - 1]] - 1
        if path[t] == path[t - 1]:
            score += logStay[path[t]] if long_enough else 0.0
            k += 1
        else:
            if not long_enough:
                return -np.inf
            score += logT[path[t - 1], path[t]]
            k, first = 0, False
        score += logE[t, path[t]]
    return score


def _brute(logE, mins, logT, logStay):
    return max(_score(p, logE, mins, logT, logStay)
               for p in itertools.product(range(4), repeat=len(logE)))


def _rules(arrays):
    _, self_cost, _, _, entry, last, exit_cost, log_dest = arrays
    mins = np.array([last[s] - entry[s] + 1 for s in range(4)])
    return mins, exit_cost[:, None] + log_dest, self_cost[last]


def test_viterbi_matches_enumeration():
    """k = 1 (minimum + geometric tail) against enumeration of the run-length rules."""
    rng = np.random.default_rng(0)
    stats = _stats(rng)
    for trial in range(6):
        logE = rng.normal(0, 1.5, (7, 4))
        cfg = {"min_live": 1 + trial % 3, "min_dead": 1 + (trial // 2) % 3}
        arrays = P.chain(stats, cfg)
        mins, logT, logStay = _rules(arrays)
        path = P.viterbi(logE, *arrays)
        assert np.isclose(_score(path, logE, mins, logT, logStay), _brute(logE, mins, logT, logStay))


def test_sampled_paths_follow_the_exact_posterior():
    """FFBS draws against the posterior from enumerating every path under the same rules."""
    rng = np.random.default_rng(4)
    stats = _stats(rng)
    logE = rng.normal(0, 1.0, (5, 4))
    logE[2, :2] = -np.inf  # a forced-dead frame, as decode() makes them
    arrays = P.chain(stats, {"min_live": 2, "min_dead": 2})
    mins, logT, logStay = _rules(arrays)
    paths = list(itertools.product(range(4), repeat=5))
    w = np.exp([_score(p, logE, mins, logT, logStay) for p in paths])
    exact = np.einsum("p,pts->ts", w / w.sum(), np.eye(4)[np.array(paths)])
    draws = P.sample_paths(logE, *arrays, 20000, 0)
    freq = np.stack([(draws == s).mean(0) for s in range(4)], 1)
    np.testing.assert_allclose(freq, exact, atol=0.015)
    assert P.sample_paths(logE, *arrays, 50, 7).tolist() == P.sample_paths(logE, *arrays, 50, 7).tolist()


def test_mbr_prefers_the_consensus_and_keeps_viterbi_on_ties():
    H, A = 0, 1
    vit = np.array([H] * 40 + [A] * 40, np.int8)
    odd = np.array([H] * 80, np.int8)
    assert P.mbr(vit, np.stack([odd] + [vit] * 4), n_cand=1) is vit
    assert P.mbr(vit, np.stack([odd] * 5), n_cand=1) is not vit


def _stats(rng):
    return {"live_durs": rng.negative_binomial(3, 0.3, 500) + 1,
            "dead_durs": rng.negative_binomial(2, 0.4, 500) + 1,
            "p_switch": 0.6, "p_same": 0.5}


def test_viterbi_matches_dense_viterbi():
    """The chain Viterbi against a plain dense Viterbi over the expanded transition matrix."""
    rng = np.random.default_rng(3)
    stats = _stats(rng)
    for cfg in ({"min_live": 3, "min_dead": 2}, {"min_live": 1, "min_dead": 5}):
        arrays = P.chain(stats, cfg)
        state_of, self_cost, pred, adv, entry, last, exit_cost, log_dest = arrays
        assert max(last - entry) > 0
        n = len(state_of)
        A = np.full((n, n), -np.inf)
        for j in range(n):
            if pred[j] >= 0:
                A[pred[j], j] = adv[j]
            A[j, j] = max(A[j, j], self_cost[j])
        for s2 in range(4):
            for s in range(4):
                A[last[s2], entry[s]] = max(A[last[s2], entry[s]], exit_cost[s2] + log_dest[s2, s])
        logE = rng.normal(0, 1.5, (40, 4))
        V = np.full(n, -np.inf)
        V[last] = logE[0, state_of[last]]
        for t in range(1, 40):
            V = (V[:, None] + A).max(0) + logE[t, state_of]
        path = P.viterbi(logE, *arrays)
        # score the returned state path by re-running the dense recursion restricted to it
        Vp = np.where(state_of == path[0], np.where(np.isin(np.arange(n), last), logE[0, state_of], -np.inf), -np.inf)
        for t in range(1, 40):
            Vp = np.where(state_of == path[t], (Vp[:, None] + A).max(0) + logE[t, state_of], -np.inf)
        assert np.isclose(Vp.max(), V.max())


def _toy_match(rng, n=3000, k=22):
    team = np.repeat([0, 1], k // 2).astype(np.int8)
    players = rng.uniform(5, 100, (n, k, 2)).astype(np.float32)
    players[:, 0, 0], players[:, 11, 0] = 3, 102          # goalkeepers: home defends x = 0
    ball = np.column_stack([rng.uniform(0, 105, (n, 2)), np.full(n, np.nan)]).astype(np.float32)
    ball[rng.random(n) < 0.4] = np.nan
    return {"players": players, "team": team, "ball": ball,
            "period": np.repeat([1, 2], n // 2).astype(np.int8),
            "pitch": np.array((105, 68), np.float32)}


def test_feature_symmetry_under_team_swap():
    m = _toy_match(np.random.default_rng(1))
    s1, a1 = F.features(m)
    s2, a2 = F.features(F.swap_teams(m))
    np.testing.assert_allclose(s1, s2, rtol=1e-5, atol=1e-5, equal_nan=True)
    np.testing.assert_allclose(a1, -a2, rtol=1e-5, atol=1e-5, equal_nan=True)


def test_owner_head_is_exactly_symmetric():
    rng = np.random.default_rng(2)
    sym, anti = rng.normal(size=(2000, len(F.SYM))), rng.normal(size=(2000, len(F.ANTI)))
    label = np.where(anti[:, 0] < 0, 0, 1).astype(np.int8)
    label[rng.random(2000) < 0.3] = 2
    heads = P.fit_heads(sym, anti, label)
    _, r1 = P.predict_heads(heads, sym, anti)
    _, r2 = P.predict_heads(heads, sym, -anti)
    np.testing.assert_allclose(r1, 1 - r2, atol=1e-9)


def test_pitch_control_favours_the_team_at_the_point():
    P = np.array([[[10.0, 10.0], [30.0, 30.0]]], np.float32)       # home at (10,10), away far
    vp = np.zeros_like(P)
    home = np.array([True, False])
    pc, total = F.pitch_control(P, vp, home, np.array([[10.0, 10.0]]))
    assert np.isclose(total[0], 1.0, atol=1e-3)                     # own spot: influence 1 ...
    assert np.isclose(pc[0], 2 / (1 + np.exp(-1)) - 1, atol=1e-3)   # ... so logistic(1), as in the paper
    pc_far, _ = F.pitch_control(P, vp, home, np.array([[30.0, 30.0]]))
    assert np.isclose(pc_far[0], -pc[0], atol=1e-6)                  # mirror case
    assert np.isnan(F.pitch_control(P, vp, home, np.array([[np.nan, np.nan]]))[0][0])



def test_blend_owner_keeps_team_symmetry_and_endpoints():
    rng = np.random.default_rng(0)
    r, r_nn = rng.uniform(0.01, 0.99, 50), rng.uniform(0.01, 0.99, 50)
    np.testing.assert_allclose(P.blend_owner(r, r_nn, 0.0), r, atol=1e-9)
    np.testing.assert_allclose(P.blend_owner(r, r_nn, 1.0), r_nn, atol=1e-9)
    np.testing.assert_allclose(P.blend_owner(1 - r, 1 - r_nn, 0.75), 1 - P.blend_owner(r, r_nn, 0.75),
                               atol=1e-9)
