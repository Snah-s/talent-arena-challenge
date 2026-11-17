"""Out-of-fold head outputs and Viterbi paths for every bench match, from heads trained on the
other six -- exactly the folds of scripts/bench_eval.py. Cached in external/oof/; used by
scripts/nn_eval.py and scripts/diagnose_bench.py.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import bench  # noqa: E402
from src import possession as P  # noqa: E402
from src import tagging as T  # noqa: E402

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
