"""Switch bonus for the HSMM decoder (v9), on the cached out-of-fold head outputs: no head retrained.

    python scripts/switch_eval.py

The decoder merges short possessions (fewer predicted spells than true ones). src.possession's
switch_bonus adds to each direct LH<->LA change a constant gamma plus b times the jump of the
owner logit between the k frames before and after. Base: v7 heads with the re-tuned decoder of
scripts/decoder_eval.py. Nested estimate as in decoder_eval.py; adoption rule: nested mean above
v7 and better on at least 5 of the 7 matches. (A learned change probability from an auxiliary
change head on the neural owner network, added as c * logit(p_change), was also tried: nested
0.6650, no gain.)
"""

import itertools
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.oof import oof  # noqa: E402
from src import bench  # noqa: E402
from src import possession as P  # noqa: E402

CTL = bench.CONTROL_SENSITIVITY_S
V7 = {"alpha": 0.75, "beta": 0.75, "min_live": 75, "min_dead": 100}
BASE = {"alpha": 1.0, "beta": 0.75, "min_live": 63, "min_dead": 100}
NN_OOF = os.path.join(ROOT, "external", "nn", "out", "oof_rnn.npz")


GRID = {"gamma": (0.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0), "b": (0.0, 0.5, 1.0, 2.0), "k": (25, 50)}
CONFIGS = [c for c in (dict(zip(GRID, v)) for v in itertools.product(*GRID.values()))
           if c["b"] or c["k"] == 25]  # k is unused when b = 0


def decode(q, r, period, p_alive, arrays, dec, c):
    logE = P.log_emissions(q, r, p_alive, dec["alpha"], dec["beta"])
    sw = P.switch_bonus(r, period, c if (c["gamma"] or c["b"]) else None)
    out = np.full(len(q), bench.X, np.int8)
    for half in (1, 2):
        idx = np.flatnonzero(period == half)
        out[idx] = P.EMIT[P.viterbi(logE[idx], sw[idx], *arrays)]
    return out


def fold(args):
    held, mids, cfg = args
    matches = {m: bench.load_match(m) for m in mids}
    lab = {m: bench.controlled_labels(matches[m]["label"], matches[m]["period"], cfg["train_control_s"])
           for m in mids}
    train = [m for m in mids if m != held]
    p_alive = float(np.mean(np.concatenate([lab[m][::cfg["stride"]] for m in train]) != bench.D))
    stats = P.transition_stats([lab[m] for m in train], [matches[m]["period"] for m in train])
    period = matches[held]["period"]
    truth = {n: bench.controlled_labels(matches[held]["label"], period, n) for n in CTL}
    nn_out = np.load(NN_OOF)
    res = np.zeros((len(CONFIGS) + 1, bench.COPIES, len(CTL), 2))  # last row: v7 as submitted
    for cp, (q, r, _) in enumerate(oof(mids, cfg)[held]):
        r = P.blend_owner(r, nn_out["{}_r{}".format(held, cp)], cfg["owner_nn"]["weight"])
        preds = [decode(q, r, period, p_alive, P.chain(stats, BASE), BASE, c) for c in CONFIGS]
        preds.append(decode(q, r, period, p_alive, P.chain(stats, V7), V7, CONFIGS[0]))
        for g, pred in enumerate(preds):
            for n, ctl in enumerate(CTL):
                res[g, cp, n] = bench.quick_score(truth[ctl], pred, period)
    print("done", held, flush=True)
    return res


def main():
    with open(os.path.join(ROOT, "configs", "model.json")) as handle:
        cfg = json.load(handle)
    mids = bench.match_ids()
    oof(mids, cfg)
    with Pool(len(mids)) as pool:
        res = np.stack(pool.map(fold, [(m, mids, cfg) for m in mids]), axis=1)
    np.save(os.path.join(ROOT, "external", "switch_eval.npy"), res)

    spell, macro = res[..., 0].mean(axis=(2, 3)), res[..., 1].mean(axis=(2, 3))
    v7, grid_sp = spell[-1], spell[:-1]
    picks = [int(np.argmax(np.delete(grid_sp, j, axis=1).mean(axis=1))) for j in range(len(mids))]
    nested = np.array([grid_sp[g, j] for j, g in enumerate(picks)])
    print("\nv7 decoder       spellF1 {:.4f}  macroF1 {:.4f}".format(v7.mean(), macro[-1].mean()))
    print("re-tuned, no bonus spellF1 {:.4f}".format(grid_sp[0].mean()))
    print("nested           spellF1 {:.4f}  wins vs v7 {}/{}".format(
        nested.mean(), int((nested > v7 + 1e-9).sum()), len(mids)))
    for mid, x, y, g in zip(mids, v7, nested, picks):
        print("  {}  v7 {:.4f}  nested {:.4f}  pick {}".format(mid, x, y, CONFIGS[g]))
    for g in np.argsort(-grid_sp.mean(axis=1))[:8]:
        print("  {:.4f}  macro {:.4f}  wins vs v7 {}/{}  {}".format(
            grid_sp[g].mean(), macro[g].mean(), int((grid_sp[g] > v7 + 1e-9).sum()), len(mids), CONFIGS[g]))


if __name__ == "__main__":
    main()
