import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import nn_inputs as N  # noqa: E402


def _toy(rng, S=60, K=30):
    players = rng.uniform(0, 100, (S, K, 2)).astype(np.float32)
    players[rng.random((S, K)) < 0.3] = np.nan
    ball = rng.uniform(0, 100, (S, 2)).astype(np.float32)
    ball[:5] = np.nan
    ball[rng.random(S) < 0.4] = np.nan
    team = (np.arange(K) % 2).astype(np.int8)
    return players, team, ball, np.array((105.0, 68.0), np.float32)


def test_shapes_and_team_swap_exchanges_the_groups():
    players, team, ball, pitch = _toy(np.random.default_rng(0))
    tok, mask, bf = N.build_inputs(players, team, ball, pitch)
    assert tok.shape == (60, 2, N.M, N.TOK_F) and mask.shape == (60, 2, N.M) and bf.shape == (60, N.BALL_F)
    assert np.isfinite(tok).all() and np.isfinite(bf).all()
    assert not tok[~mask].any()
    tok2, mask2, bf2 = N.build_inputs(players, 1 - team, ball, pitch)
    np.testing.assert_array_equal(tok2, tok[:, ::-1])
    np.testing.assert_array_equal(mask2, mask[:, ::-1])
    np.testing.assert_array_equal(bf2, bf)


def test_tokens_are_nearest_first_to_the_ball_reference():
    players, team, ball, pitch = _toy(np.random.default_rng(1))
    tok, mask, _ = N.build_inputs(players, team, ball, pitch)
    dist = tok[..., 6]
    for t in range(len(tok)):
        for g in (0, 1):
            d = dist[t, g][mask[t, g]]
            assert (np.diff(d) >= 0).all()


def test_owner_net_is_exactly_team_antisymmetric():
    torch = pytest.importorskip("torch")
    from src import owner_nn as O
    players, team, ball, pitch = _toy(np.random.default_rng(2))
    tok, mask, bf = (torch.from_numpy(a)[None] for a in N.build_inputs(players, team, ball, pitch))
    model = O.OwnerNet().eval()
    with torch.no_grad():
        a = O.symmetric_logits(model, tok, mask, bf)
        b = O.symmetric_logits(model, *O._swap(tok, mask), bf)
    torch.testing.assert_close(a, -b)
