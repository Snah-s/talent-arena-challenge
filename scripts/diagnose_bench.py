"""Where is the remaining error? Two bench-only diagnostics at the chosen configuration.

    python scripts/diagnose_bench.py ceiling   # perfect ball (no gaps, no phantoms): vision's ceiling
    python scripts/diagnose_bench.py oracle    # one head replaced by the truth at a time

Same leave-one-match-out folds, pseudo-tags and scoring as scripts/bench_eval.py. Not used to
choose anything for the submission; it only says which lever has room.
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import bench  # noqa: E402
from src import features as F  # noqa: E402
from src import possession as P  # noqa: E402
from src import tagging as T  # noqa: E402

CTL = bench.CONTROL_SENSITIVITY_S


def perfect_ball_inputs(mid):
    """Like tagging.with_tags(mid, "pseudo") but the ball is never lost and never phantom."""
    m = bench.load_match(mid)
    i = bench.match_ids().index(mid)
    out = []
    for c in range(bench.COPIES):
        rng = np.random.default_rng(1000 * i + c)
        cfg = dict(bench.sample_degradation(rng), ball_missing=0.0, phantom_frac=0.0)
        s, a = F.features(bench.degrade(m, cfg, rng))
        trng = np.random.default_rng(500_000 + 1000 * i + c)
        ts, ta = T.tag_features(T.pseudo_tags(m["label"], m["period"], trng, T.sample_pseudo(trng)), len(s))
        out.append((np.hstack([s, ts]), np.hstack([a, ta])))
    return out


def train_on(inputs, mids, labels, periods, stride):
    sym = np.vstack([s[::stride] for mid in mids for s, _ in inputs[mid]])
    anti = np.vstack([a[::stride] for mid in mids for _, a in inputs[mid]])
    lab = np.concatenate([labels[mid][::stride] for mid in mids for _ in inputs[mid]])
    heads = P.fit_heads(sym, anti, lab)
    return heads, P.transition_stats([labels[mid] for mid in mids], [periods[mid] for mid in mids])


def main(mode):
    with open(os.path.join(ROOT, "configs", "model.json")) as handle:
        cfg = json.load(handle)
    mids = bench.match_ids()
    matches = {mid: bench.load_match(mid) for mid in mids}
    periods = {mid: m["period"] for mid, m in matches.items()}
    train_lab = {mid: bench.controlled_labels(m["label"], m["period"], cfg["train_control_s"])
                 for mid, m in matches.items()}
    eval_lab = {mid: {n: bench.controlled_labels(m["label"], m["period"], n) for n in CTL}
                for mid, m in matches.items()}
    if mode == "ceiling":
        inputs = {mid: perfect_ball_inputs(mid) for mid in mids}
        variants = ("model",)
    else:
        inputs = {mid: T.with_tags(mid, "pseudo") for mid in mids}
        variants = ("model", "oracle alive", "oracle owner", "oracle both")

    res = {v: np.zeros((len(mids), bench.COPIES, len(CTL), 2)) for v in variants}
    for j, held in enumerate(mids):
        train = [mid for mid in mids if mid != held]
        heads, stats = train_on(inputs, train, train_lab, periods, cfg["stride"])
        truth = train_lab[held]
        q_true = np.where(truth != bench.D, 1.0, 0.0)
        # owner truth on live frames; on dead frames the owner the dead states should remember
        live_owner = np.where(truth == bench.H, 1.0, np.where(truth == bench.A, 0.0, np.nan))
        r_true = F._bfill(F._ffill(live_owner))
        for c, (s, a) in enumerate(inputs[held]):
            q, r = P.predict_heads(heads, s, a)
            pairs = {"model": (q, r), "oracle alive": (q_true, r), "oracle owner": (q, r_true),
                     "oracle both": (q_true, r_true)}
            for v in variants:
                pred = P.decode(*pairs[v], periods[held], heads, stats, cfg)
                for n, ctl in enumerate(CTL):
                    res[v][j, c, n] = bench.quick_score(eval_lab[held][ctl], pred, periods[held])
        print("fold {} {}  ".format(j, held) + "  ".join(
            "{} {:.4f}".format(v, res[v][j, ..., 0].mean()) for v in variants), flush=True)

    print("\n{:<14} spellF1  [N=1/2/3 s]              macroF1  per-match spellF1".format(mode))
    for v in variants:
        print("{:<14} {:.4f}   {}   {:.4f}   {}".format(
            v, res[v][..., 0].mean(), " ".join("{:.4f}".format(x) for x in res[v][..., 0].mean(axis=(0, 1))),
            res[v][..., 1].mean(), " ".join("{:.3f}".format(x) for x in res[v][..., 0].mean(axis=(1, 2)))))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "oracle")
