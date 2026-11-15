"""Bench evaluation of the second stage (src/spells.py) at the configuration in configs/model.json.

    python scripts/spell_eval.py

Step 1 (cached in external/oof/): out-of-fold head outputs and Viterbi paths for every bench match,
from heads trained on the other six -- exactly the folds of scripts/bench_eval.py.
Step 2: for each held-out match, the run classifier is trained on the out-of-fold runs of the other
six matches and applied to the held-out one. (Those other matches' first-stage heads saw the
held-out match in training; its labels never reach its own predictions.)
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import bench  # noqa: E402
from src import possession as P  # noqa: E402
from src import spells as SP  # noqa: E402
from src import tagging as T  # noqa: E402

CTL = bench.CONTROL_SENSITIVITY_S
OOF_DIR = os.path.join(ROOT, "external", "oof")


def oof(mids, cfg):
    """{mid: [(q, r, pred) per copy]} from heads that never saw that match; cached."""
    tag = "a{alpha}_l{min_live}_d{min_dead}_{tags}".format(**cfg)
    out = {}
    for held in mids:
        path = os.path.join(OOF_DIR, "{}_{}.npz".format(held, tag))
        if not os.path.exists(path):
            heads, stats = P.train([m for m in mids if m != held], cfg["train_control_s"],
                                   cfg["stride"], cfg["tags"])
            period = bench.load_match(held)["period"]
            arrs = {}
            for c, (s, a) in enumerate(T.with_tags(held, cfg["tags"])):
                q, r = P.predict_heads(heads, s, a)
                arrs.update({"q{}".format(c): q, "r{}".format(c): r,
                             "pred{}".format(c): P.decode(q, r, period, heads, stats, cfg)})
            os.makedirs(OOF_DIR, exist_ok=True)
            np.savez_compressed(path, **arrs)
            print("oof", held, flush=True)
        with np.load(path) as z:
            out[held] = [(z["q{}".format(c)], z["r{}".format(c)], z["pred{}".format(c)])
                         for c in range(bench.COPIES)]
    return out


def main():
    with open(os.path.join(ROOT, "configs", "model.json")) as handle:
        cfg = json.load(handle)
    mids = bench.match_ids()
    matches = {mid: bench.load_match(mid) for mid in mids}
    train_lab = {mid: bench.controlled_labels(m["label"], m["period"], cfg["train_control_s"])
                 for mid, m in matches.items()}
    eval_lab = {mid: {n: bench.controlled_labels(m["label"], m["period"], n) for n in CTL}
                for mid, m in matches.items()}
    heads_out = oof(mids, cfg)
    tables = {}
    for mid in mids:
        period = matches[mid]["period"]
        tables[mid] = [SP.run_features(pred, period, q, r, s, a) + (pred,)
                       for (q, r, pred), (s, a) in zip(heads_out[mid], T.with_tags(mid, cfg["tags"]))]

    variants = ("model", "stage owner-only", "stage full")
    res = np.zeros((len(variants), len(mids), bench.COPIES, len(CTL), 2))
    for j, held in enumerate(mids):
        clf = SP.fit([(rr, s, a, train_lab[mid]) for mid in mids if mid != held
                      for rr, s, a, _ in tables[mid]])
        period = matches[held]["period"]
        for c, (rr, s, a, pred) in enumerate(tables[held]):
            p = SP.proba(clf, s, a)
            preds = (pred, SP.relabel(pred, rr, p, owner_only=True), SP.relabel(pred, rr, p))
            for v, pv in enumerate(preds):
                for n, ctl in enumerate(CTL):
                    res[v, j, c, n] = bench.quick_score(eval_lab[held][ctl], pv, period)
        print("{}  ".format(held) + "  ".join("{} {:.4f}".format(v, res[i, j, ..., 0].mean())
                                              for i, v in enumerate(variants)), flush=True)

    print("\n{:<18} spellF1  [N=1/2/3 s]              macroF1  wins vs model".format(""))
    for i, v in enumerate(variants):
        per = res[i, ..., 0].mean(axis=(1, 2))
        print("{:<18} {:.4f}   {}   {:.4f}   {}/{}".format(
            v, res[i, ..., 0].mean(), " ".join("{:.4f}".format(x) for x in res[i, ..., 0].mean(axis=(0, 1))),
            res[i, ..., 1].mean(), int((per > res[0, ..., 0].mean(axis=(1, 2)) + 1e-9).sum()), len(mids)))


if __name__ == "__main__":
    main()
