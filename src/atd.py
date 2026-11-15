"""Challenge ATD -> the match dict of src/bench.py (players, team, ball, period, pitch).

Row i is video frame i (= submission row i). Period is 1/2 inside the graded halves, 0 elsewhere.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import team_of_each_track  # noqa: E402


def load_atd(game=GAME, data_dir=None):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    team = team_of_each_track(meta_path)
    n, k = source_frames("tactical", game), len(team)
    length, width = GAMES[game]["pitch"]

    players = np.full((n, k, 2), np.nan, np.float32)
    ball = np.full((n, 3), np.nan, np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1  # ATD frame_id counts from 1
            p = np.array(parts[1].replace(";", ",").split(","), dtype=np.float32).reshape(k, -1)
            players[row] = p[:, :2]
            ball[row, :2] = np.array(parts[2].split(",")[:2], dtype=np.float32)

    players *= (length, width)
    ball[:, :2] *= (length, width)
    period = np.zeros(n, np.int8)
    for half, first, last in half_video_blocks(game):
        period[first:last + 1] = half
    return {"players": players, "team": team, "ball": ball, "period": period,
            "pitch": np.array((length, width), np.float32)}
