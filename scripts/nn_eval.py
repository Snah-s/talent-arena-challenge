"""Plan v3 E4 on the bench: does the neural owner head help the HGB r head?

    python scripts/nn_eval.py [--oof external/nn/out/oof_rnn.npz]

r_nn comes out of fold from Colab (src/owner_nn.py; same held-out matches and degraded copies as
the cached HGB heads of scripts/spell_eval.py). The two owner heads are combined in log-odds with
weight w on r_nn and decoded with the unchanged v6 HMM; w = 0 is v6. Adoption rule: higher mean
spellF1 and higher on at least 5 of the 7 matches.
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.spell_eval import oof  # noqa: E402
from src import bench  # noqa: E402
from src import possession as P  # noqa: E402

CTL = bench.CONTROL_SENSITIVITY_S
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", default=os.path.join(ROOT, "external", "nn", "out", "oof_rnn.npz"))
    args = parser.parse_args()
    with open(os.path.join(ROOT, "configs", "model.json")) as handle:
        cfg = json.load(handle)
    mids = bench.match_ids()
    matches = {mid: bench.load_match(mid) for mid in mids}
    periods = {mid: m["period"] for mid, m in matches.items()}
    train_lab = {mid: bench.controlled_labels(m["label"], m["period"], cfg["train_control_s"])
                 for mid, m in matches.items()}
    eval_lab = {mid: {n: bench.controlled_labels(m["label"], m["period"], n) for n in CTL}
                for mid, m in matches.items()}
    cached = oof(mids, cfg)
    nn_out = np.load(args.oof)

    res = np.zeros((len(WEIGHTS), len(mids), bench.COPIES, len(CTL), 2))
    acc = np.zeros((len(WEIGHTS), len(mids), bench.COPIES))
    for j, held in enumerate(mids):
        train = [m for m in mids if m != held]
        heads = {"p_alive": float(np.mean(np.concatenate(
            [train_lab[m][::cfg["stride"]] for m in train]) != bench.D))}
        stats = P.transition_stats([train_lab[m] for m in train], [periods[m] for m in train])
        truth = eval_lab[held][2.0]
        live = truth != bench.D
        for c, (q, r, vit) in enumerate(cached[held]):
            r_nn = nn_out["{}_r{}".format(held, c)]
            assert len(r_nn) == len(r), "r_nn does not line up with the cached heads"
            for i, w in enumerate(WEIGHTS):
                rc = P.blend_owner(r, r_nn, w)
                acc[i, j, c] = ((rc[live] > 0.5) == (truth[live] == bench.H)).mean()
                pred = vit if w == 0 else P.decode(q, rc, periods[held], heads, stats, cfg)
                for n, ctl in enumerate(CTL):
                    res[i, j, c, n] = bench.quick_score(eval_lab[held][ctl], pred, periods[held])
        print("{}  ".format(held) + "  ".join("w={} {:.4f}".format(w, res[i, j, ..., 0].mean())
                                              for i, w in enumerate(WEIGHTS)), flush=True)

    base = res[0, ..., 0].mean(axis=(1, 2))
    print("\nw      spellF1  [N=1/2/3 s]              macroF1  owner acc (live, N=2)  wins vs v6")
    for i, w in enumerate(WEIGHTS):
        per = res[i, ..., 0].mean(axis=(1, 2))
        print("{:<6} {:.4f}   {}   {:.4f}   {:.4f}                 {}/{}".format(
            w, res[i, ..., 0].mean(), " ".join("{:.4f}".format(x) for x in res[i, ..., 0].mean(axis=(0, 1))),
            res[i, ..., 1].mean(), acc[i].mean(), int((per > base + 1e-9).sum()), len(mids)))


if __name__ == "__main__":
    main()
