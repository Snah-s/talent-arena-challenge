"""Leave-one-match-out tuning on the external DFL bench. The only place parameters are chosen.

For each held-out match: heads and transition statistics are fitted on the other six, then every
grid configuration is decoded on two degraded copies of the held-out match and scored against its
labels under each control threshold (bench.CONTROL_SENSITIVITY_S). The configuration with the best
mean spellF1 over all of that is written to external/model_<tags>.json, and to
configs/model.json with --write.

    python scripts/bench_eval.py --tags none|pseudo|shifted [--write]

--tags pseudo adds noisy set-piece pseudo-tags (src/tagging.py); --tags shifted is its control.
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import bench  # noqa: E402
from src import features as F  # noqa: E402
from src import possession as P  # noqa: E402
from src import tagging  # noqa: E402

COPIES = bench.COPIES
STRIDE = 5          # training subsample; consecutive frames are near duplicates
TRAIN_CONTROL_S = bench.CONTROL_S
GRID = {"alpha": (0.2, 0.35, 0.5, 0.75),
        "min_live": (50, 75, 100, 125, 150),  # frames: 2, 3, 4, 5, 6 s (grid 1 peaked at its 3 s edge)
        "min_dead": (25, 50, 100)}         # frames: 1, 2, 4 s
CONFIG_PATH = os.path.join(ROOT, "configs", "model.json")


def baseline_smooth50(anti, period):
    """The published floor on the bench: nearest team where the ball is seen, majority +/- 50."""
    dd = anti[:, F.ANTI.index("dd")]
    votes = np.stack([dd < 0, dd > 0, np.isnan(dd)]).astype(np.float64)
    out = np.empty(len(dd), np.int8)
    for half in (1, 2):
        idx = np.flatnonzero(period == half)
        out[idx] = np.argmax(np.stack([F._roll(v[idx], 50) for v in votes]), axis=0)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", default="none", choices=("none", "pseudo", "shifted"))
    parser.add_argument("--write", action="store_true", help="also write configs/model.json")
    args = parser.parse_args()

    mids = bench.match_ids()
    matches = {mid: bench.load_match(mid) for mid in mids}
    labels = {mid: {s: bench.controlled_labels(m["label"], m["period"], s)
                    for s in bench.CONTROL_SENSITIVITY_S} for mid, m in matches.items()}
    feats = {mid: tagging.with_tags(mid, args.tags) for mid in mids}
    grid = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    scores = np.zeros((len(grid), len(mids), COPIES, len(bench.CONTROL_SENSITIVITY_S), 2))
    base = np.zeros((len(mids), COPIES, len(bench.CONTROL_SENSITIVITY_S), 2))

    for j, held in enumerate(mids):
        train = [mid for mid in mids if mid != held]
        heads, stats = P.train(train, TRAIN_CONTROL_S, STRIDE, args.tags)
        period = matches[held]["period"]
        for c, (s, a) in enumerate(feats[held]):
            q, r = P.predict_heads(heads, s, a)
            b = baseline_smooth50(a, period)
            for n, ctl in enumerate(bench.CONTROL_SENSITIVITY_S):
                base[j, c, n] = bench.quick_score(labels[held][ctl], b, period)
            for g, cfg in enumerate(grid):
                pred = P.decode(q, r, period, heads, stats, cfg)
                for n, ctl in enumerate(bench.CONTROL_SENSITIVITY_S):
                    scores[g, j, c, n] = bench.quick_score(labels[held][ctl], pred, period)
        best = scores[:, j, :, :, 0].mean(axis=(1, 2)).max()
        print("fold {} held out {}  best spellF1 {:.4f}  baseline {:.4f}".format(
            j, held, best, base[j, :, :, 0].mean()), flush=True)

    mean = scores[..., 0].mean(axis=(1, 2, 3))
    order = np.argsort(-mean)
    print("\nrank  alpha  min_live  min_dead  spellF1  [per N=1/2/3 s]      worst match  macroF1")
    for rank, g in enumerate(order[:10]):
        per_n = scores[g, ..., 0].mean(axis=(0, 1))
        worst = scores[g, ..., 0].mean(axis=(1, 2)).min()
        print("{:>4}  {:>5}  {:>8}  {:>8}  {:.4f}   {}   {:.4f}       {:.4f}".format(
            rank + 1, grid[g]["alpha"], grid[g]["min_live"], grid[g]["min_dead"], mean[g],
            " ".join("{:.4f}".format(v) for v in per_n), worst, scores[g, ..., 1].mean()))
    print("baseline_smooth50 on the bench: spellF1 {:.4f} [{}]  macroF1 {:.4f}".format(
        base[..., 0].mean(), " ".join("{:.4f}".format(v) for v in base[..., 0].mean(axis=(0, 1))),
        base[..., 1].mean()))

    chosen = dict(grid[order[0]], tags=args.tags, train_control_s=TRAIN_CONTROL_S, stride=STRIDE,
                  bench_spell_f1=round(float(mean[order[0]]), 4),
                  bench_baseline_spell_f1=round(float(base[..., 0].mean()), 4))
    paths = [os.path.join(ROOT, "external", "model_{}.json".format(args.tags))]
    paths += [CONFIG_PATH] if args.write else []
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            json.dump(chosen, handle, indent=2)
        print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
