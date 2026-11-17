"""Re-tune the HSMM decoder for v7's owner probabilities (HGB r blended with the neural head).

    python scripts/decoder_eval.py [--oof external/nn/out_e6a/oof_rnn.npz] [--name e6a]

v6's decoder grid was run before the neural head existed. This grid adds the blend weight and a
separate owner temperature beta (LH = alpha * live + beta * log 2r), on the cached out-of-fold head
outputs: no head is retrained. To avoid picking the best of many configurations on the same
matches, the estimate is nested: each held-out match is scored with the configuration that is best
on the other six. Adoption rule: nested mean above v7 and better on at least 5 of the 7 matches.
"""

import argparse
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
GRID = {"w": (0.5, 0.75, 1.0), "alpha": (0.5, 0.75, 1.0), "beta": (0.5, 0.75, 1.0, 1.25, 1.5),
        "min_live": (38, 50, 63, 75, 100), "min_dead": (75, 100, 125)}
CONFIGS = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]


def decode(q, r, period, p_alive, stats, c):
    arrays = P.chain(stats, c)
    qc, rc = np.clip(q, P.EPS, 1 - P.EPS), np.clip(r, P.EPS, 1 - P.EPS)
    live = c["alpha"] * (np.log(qc) - np.log(p_alive))
    dead = c["alpha"] * (np.log1p(-qc) - np.log1p(-p_alive))
    logE = np.column_stack([live + c["beta"] * np.log(2 * rc), live + c["beta"] * np.log(2 * (1 - rc)),
                            dead, dead])
    out = np.full(len(q), bench.X, np.int8)
    for half in (1, 2):
        idx = np.flatnonzero(period == half)
        out[idx] = P.EMIT[P.viterbi(logE[idx], np.zeros((len(idx), 2)), *arrays)]
    return out


def fold(args):
    held, mids, cfg, nn_path = args
    matches = {m: bench.load_match(m) for m in mids}
    train_lab = {m: bench.controlled_labels(matches[m]["label"], matches[m]["period"],
                                            cfg["train_control_s"]) for m in mids}
    train = [m for m in mids if m != held]
    p_alive = float(np.mean(np.concatenate([train_lab[m][::cfg["stride"]] for m in train]) != bench.D))
    stats = P.transition_stats([train_lab[m] for m in train], [matches[m]["period"] for m in train])
    period = matches[held]["period"]
    truth = {n: bench.controlled_labels(matches[held]["label"], period, n) for n in CTL}
    cached = oof(mids, cfg)[held]
    nn_out = np.load(nn_path)
    res = np.zeros((len(CONFIGS), bench.COPIES, len(CTL), 2))
    for c, (q, r, _) in enumerate(cached):
        r_nn = nn_out["{}_r{}".format(held, c)]
        blends = {w: P.blend_owner(r, r_nn, w) for w in GRID["w"]}
        for g, conf in enumerate(CONFIGS):
            pred = decode(q, blends[conf["w"]], period, p_alive, stats, conf)
            for n, ctl in enumerate(CTL):
                res[g, c, n] = bench.quick_score(truth[ctl], pred, period)
    print("done", held, flush=True)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", default=os.path.join(ROOT, "external", "nn", "out", "oof_rnn.npz"))
    parser.add_argument("--name", default="v7")
    args = parser.parse_args()
    with open(os.path.join(ROOT, "configs", "model.json")) as handle:
        cfg = json.load(handle)
    mids = bench.match_ids()
    oof(mids, cfg)
    with Pool(len(mids)) as pool:
        res = np.stack(pool.map(fold, [(m, mids, cfg, args.oof) for m in mids]), axis=1)
    np.save(os.path.join(ROOT, "external", "decoder_eval_{}.npy".format(args.name)), res)

    spell = res[..., 0].mean(axis=(2, 3))
    v7 = CONFIGS.index({"w": 0.75, "alpha": 0.75, "beta": 0.75, "min_live": 75, "min_dead": 100})
    nested = np.array([spell[np.argmax(np.delete(spell, j, axis=1).mean(axis=1)), j]
                       for j in range(len(mids))])
    picks = [CONFIGS[np.argmax(np.delete(spell, j, axis=1).mean(axis=1))] for j in range(len(mids))]
    best = int(np.argmax(spell.mean(axis=1)))
    print("\nv7 config      spellF1 {:.4f}  macroF1 {:.4f}".format(spell[v7].mean(), res[v7, ..., 1].mean()))
    print("nested         spellF1 {:.4f}  wins vs v7 {}/{}".format(
        nested.mean(), int((nested > spell[v7] + 1e-9).sum()), len(mids)))
    for mid, x, y, p in zip(mids, spell[v7], nested, picks):
        print("  {}  v7 {:.4f}  nested {:.4f}  pick {}".format(mid, x, y, p))
    print("best on all 7  spellF1 {:.4f}  macroF1 {:.4f}  wins vs v7 {}/{}  {}".format(
        spell[best].mean(), res[best, ..., 1].mean(), int((spell[best] > spell[v7] + 1e-9).sum()),
        len(mids), CONFIGS[best]))
    for g in np.argsort(-spell.mean(axis=1))[:8]:
        print("  {:.4f}  {}".format(spell[g].mean(), CONFIGS[g]))


if __name__ == "__main__":
    main()
