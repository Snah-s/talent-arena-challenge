"""Plan v3 E2 on the bench: MBR decoding for spellF1 against the Viterbi path.

    python scripts/mbr_eval.py

Reuses the out-of-fold head outputs and Viterbi paths cached by scripts/spell_eval.py (same
leave-one-match-out folds as scripts/bench_eval.py); p_alive and the transition statistics are
recomputed from the six training matches exactly as P.train() does. Only the sampling temperature
varies. Adoption rule: higher mean spellF1 and higher on at least 5 of the 7 matches.
"""

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
MBR_ALPHAS = (None, 0.5, 1.0)  # None = the model's alpha


def n_spells(codes, period):
    cut = int(np.argmax(period == 2))
    c = np.insert(np.asarray(codes, np.int8), cut, bench.X)
    return len(bench.pm.possession_spells(c, bench.pm.valid_runs(c)))


def main():
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
    names = ["viterbi"] + ["mbr a={}".format(a or cfg["alpha"]) for a in MBR_ALPHAS]
    res = np.zeros((len(names), len(mids), bench.COPIES, len(CTL), 2))
    spells = np.zeros((len(names), len(mids), bench.COPIES))

    for j, held in enumerate(mids):
        train = [m for m in mids if m != held]
        heads = {"p_alive": float(np.mean(np.concatenate(
            [train_lab[m][::cfg["stride"]] for m in train]) != bench.D))}
        stats = P.transition_stats([train_lab[m] for m in train], [periods[m] for m in train])
        for c, (q, r, vit) in enumerate(cached[held]):
            if c == 0:
                assert (P.decode(q, r, periods[held], heads, stats, cfg) == vit).all(), \
                    "recomputed p_alive/stats do not reproduce the cached Viterbi path"
            preds = [vit] + [P.decode(q, r, periods[held], heads, stats,
                                      dict(cfg, mbr=True, mbr_alpha=a or cfg["alpha"]))
                             for a in MBR_ALPHAS]
            for v, pred in enumerate(preds):
                spells[v, j, c] = n_spells(pred, periods[held])
                for n, ctl in enumerate(CTL):
                    res[v, j, c, n] = bench.quick_score(eval_lab[held][ctl], pred, periods[held])
        print("{}  ".format(held) + "  ".join("{} {:.4f}".format(v, res[i, j, ..., 0].mean())
                                              for i, v in enumerate(names)), flush=True)

    true_spells = np.mean([n_spells(eval_lab[m][2.0], periods[m]) for m in mids])
    base = res[0, ..., 0].mean(axis=(1, 2))
    print("\n{:<11} spellF1  [N=1/2/3 s]              macroF1  spells (true {:.1f})  wins vs viterbi".format(
        "", true_spells))
    for i, v in enumerate(names):
        per = res[i, ..., 0].mean(axis=(1, 2))
        print("{:<11} {:.4f}   {}   {:.4f}   {:>6.1f}               {}/{}".format(
            v, res[i, ..., 0].mean(), " ".join("{:.4f}".format(x) for x in res[i, ..., 0].mean(axis=(0, 1))),
            res[i, ..., 1].mean(), spells[i].mean(), int((per > base + 1e-9).sum()), len(mids)))


if __name__ == "__main__":
    main()
