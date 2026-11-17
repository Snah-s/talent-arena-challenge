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
    assert not tok[~mask].any()  # padding is zero
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


@pytest.mark.parametrize("over", [{}, {"layers": 2, "aux": 0.5, "d": 32, "hidden": 16}])
def test_owner_net_is_exactly_team_antisymmetric(over):
    torch = pytest.importorskip("torch")
    from src import owner_nn as O
    players, team, ball, pitch = _toy(np.random.default_rng(2))
    tok, mask, bf = (torch.from_numpy(a)[None] for a in N.build_inputs(players, team, ball, pitch))
    model = O.build(dict(O.CFG, **over)).eval()
    with torch.no_grad():
        a = O.symmetric_logits(model, tok, mask, bf)
        b = O.symmetric_logits(model, *O._swap(tok, mask), bf)
    torch.testing.assert_close(a, -b)


def test_change_target_marks_only_direct_team_changes():
    pytest.importorskip("torch")
    from src import owner_nn as O
    y = np.array([1, 1, 0, 0, -1, 1, 1, 0, 0, 0])
    period = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2])
    # 1->0 at step 2 is direct; 0->D->1 is not; the 1->0 at step 7 crosses the half
    np.testing.assert_array_equal(O.change_target(y, period, 1), [0, 1, 1, 1, 0, 0, 0, 0, 0, 0])


def test_v7_weights_still_load():
    torch = pytest.importorskip("torch")
    from src import owner_nn as O
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "owner_nn_s0.pt")
    if not os.path.exists(path):
        pytest.skip("no v7 weights")
    O.build(O.CFG).load_state_dict(torch.load(path, map_location="cpu", weights_only=True))  # strict
