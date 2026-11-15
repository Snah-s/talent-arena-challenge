import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import bench  # noqa: E402
from src.bench import A, D, H  # noqa: E402

INFO = """<?xml version="1.0"?><PutDataRequest><MatchInformation><Teams>
<Team TeamId="T1" Role="home"/><Team TeamId="T2" Role="guest"/></Teams></MatchInformation>
</PutDataRequest>"""

POS = """<?xml version="1.0"?><PutDataRequest><Positions>
<MetaData><PitchSize X="105.00" Y="68.00"/></MetaData>
<FrameSet GameSection="firstHalf" TeamId="referee" PersonId="R"><Frame N="10" X="0" Y="0"/></FrameSet>
<FrameSet GameSection="firstHalf" TeamId="T2" PersonId="P2"><Frame N="11" X="1" Y="2"/></FrameSet>
<FrameSet GameSection="firstHalf" TeamId="T1" PersonId="P1"><Frame N="10" X="-52.5" Y="-34"/></FrameSet>
<FrameSet GameSection="firstHalf" TeamId="BALL" PersonId="B">
<Frame N="10" X="0" Y="0" Z="0.1" BallStatus="0" BallPossession="1"/>
<Frame N="11" X="1" Y="1" Z="2.0" BallStatus="1" BallPossession="2"/>
<Frame N="12" X="2" Y="1" Z="0.1" BallStatus="1" BallPossession="0"/></FrameSet>
<FrameSet GameSection="secondHalf" TeamId="BALL" PersonId="B">
<Frame N="50" X="0" Y="0" Z="0" BallStatus="1" BallPossession="1"/></FrameSet>
</Positions></PutDataRequest>"""


def test_parse_dfl_tiny(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "DFL_02_01_matchinformation_X_DFL-MAT-TEST.xml").write_text(INFO)
    (raw / "DFL_04_03_positions_raw_observed_X_DFL-MAT-TEST.xml").write_text(POS)
    monkeypatch.setattr(bench, "DFL_DIR", str(tmp_path))

    m = bench.parse_dfl("TEST")
    assert m["team"].tolist() == [0, 1]                      # P1 home, P2 away, referee dropped
    assert m["period"].tolist() == [1, 1, 1, 2]
    assert m["label"].tolist() == [D, A, A, H]               # owner 0 on a live frame carries A
    assert int(m["unknown_owner"]) == 1
    np.testing.assert_allclose(m["players"][0, 0], (0.0, 0.0))   # corner origin
    np.testing.assert_allclose(m["players"][1, 1], (53.5, 36.0))
    assert np.isnan(m["players"][0, 1]).all() and np.isnan(m["players"][3]).all()
    np.testing.assert_allclose(m["ball"][1], (53.5, 35.0, 2.0))


def test_gaps_hit_target_and_prefer_weighted_frames():
    rng = np.random.default_rng(0)
    w = np.ones(50000)
    w[:25000] = np.e ** 2
    mask = bench._gaps(w, 0.4, 25, rng)
    assert abs(mask.mean() - 0.4) < 0.02
    assert mask[:25000].mean() > mask[25000:].mean()


def test_degrade_keeps_labels_and_drops_height():
    rng = np.random.default_rng(1)
    n, k = 5000, 22
    m = {"players": rng.uniform(0, 60, (n, k, 2)).astype(np.float32),
         "ball": np.column_stack([rng.uniform(0, 60, (n, 2)), np.zeros(n)]).astype(np.float32),
         "team": np.repeat([0, 1], 11).astype(np.int8), "label": rng.integers(0, 3, n).astype(np.int8),
         "period": np.repeat([1, 2], n // 2).astype(np.int8),
         "pitch": np.array((105, 68), np.float32)}
    cfg = dict(bench.sample_degradation(rng), dup_frac=0.05, phantom_frac=0.03)
    out = bench.degrade(m, cfg, rng)
    assert out["label"] is m["label"]
    assert np.isnan(out["ball"][:, 2]).all()
    assert abs(np.isnan(out["ball"][:, 0]).mean() - cfg["ball_missing"]) < 0.05
    assert np.isfinite(m["ball"]).all()  # the input is not modified
    assert out["players"].shape == (n, k + bench.N_DUP, 2)
    assert out["team"].tolist()[k:] == [0, 1]
    dup_on = np.isfinite(out["players"][:, k:, 0]).mean()
    assert 0.01 < dup_on < 0.04  # 5 % of frames shared by the two columns, minus missing sources


def test_score_splits_halves():
    period = np.array([1] * 100 + [2] * 100, np.int8)
    gt = np.array([H] * 200, np.int8)
    assert bench.score(gt, gt, period)["n_gt_spells"] == 2
    assert bench.score(gt, gt, period)["spell_f1_mean"] == 1.0


def test_controlled_labels_apply_the_challenge_rule():
    s = lambda k, n: [k] * n  # noqa: E731
    lab = np.array(s(H, 100) + s(A, 10) + s(H, 100)       # touch without control -> H
                   + s(A, 60)                              # controlled switch -> A
                   + s(H, 5) + s(D, 50)                    # deflection into touch -> A, then D
                   + s(H, 5) + s(A, 100), np.int8)         # restart by H counts even if short
    period = np.ones(len(lab), np.int8)
    want = np.array(s(H, 210) + s(A, 65) + s(D, 50) + s(H, 5) + s(A, 100), np.int8)
    np.testing.assert_array_equal(bench.controlled_labels(lab, period, min_s=2.0), want)
