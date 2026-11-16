"""Where is the remaining error? Two bench-only diagnostics at the chosen configuration.

    python scripts/diagnose_bench.py ceiling   # perfect ball (no gaps, no phantoms): vision's ceiling
    python scripts/diagnose_bench.py oracle    # one head replaced by the truth at a time
    python scripts/diagnose_bench.py cuts      # which true boundaries the decoder merges (OOF cache)

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


CUT_KINDS = ("H<->A direct", "D, same team", "D, other team")
GAP_BINS = ("<=2 s", "2-5 s", ">5 s")


def boundary_kind(a1, b0, k0, k1):
    """(kind, gap bin) of the boundary between true spells ending at b0 and starting at a1."""
    gap = (a1 - b0) / 25.0
    return (0 if a1 == b0 else 1 if k0 == k1 else 2), (0 if gap <= 2 else 1 if gap <= 5 else 2)


def merged_across(cov0, cov1, len0, len1):
    """True when one predicted spell covers at least half of both neighbouring true spells."""
    return bool(np.any((cov0 >= 0.5 * len0) & (cov1 >= 0.5 * len1)))


def cuts(cfg):
    """E0: classify the true possession boundaries the decoder does not cut, at N = 2 s."""
    from scripts.spell_eval import oof
    pm = bench.pm
    mids = bench.match_ids()
    out = oof(mids, cfg)
    tot, miss = np.zeros((3, 3)), np.zeros((3, 3))
    counts = np.zeros(5)  # n_gt, n_pred, unmatched@0.5, of which merged, of which fragmented
    n_runs = 0
    for mid in mids:
        m = bench.load_match(mid)
        cut = int(np.argmax(m["period"] == 2))
        gt = np.insert(bench.controlled_labels(m["label"], m["period"], 2.0).astype(np.int8), cut, bench.X)
        runs = pm.valid_runs(gt)
        gs = pm.possession_spells(gt, runs)
        for _, _, pred in out[mid]:
            n_runs += 1
            pr = pm.fill_abstentions(np.insert(pred.astype(np.int8), cut, bench.X), runs)
            ps = pm.possession_spells(pr, runs)
            pa, pb = np.array([s[0] for s in ps]), np.array([s[1] for s in ps])
            covs = [np.clip(np.minimum(pb, b) - np.maximum(pa, a), 0, None) for a, b, _ in gs]
            merged = np.zeros(len(gs), bool)
            for i in range(len(gs) - 1):
                (a0, b0, k0), (a1, b1, k1) = gs[i], gs[i + 1]
                if np.any(gt[b0:a1] == bench.X):  # halftime
                    continue
                kind, gbin = boundary_kind(a1, b0, k0, k1)
                hit = merged_across(covs[i], covs[i + 1], b0 - a0, b1 - a1)
                tot[kind, gbin] += 1
                miss[kind, gbin] += hit
                merged[i] |= hit
                merged[i + 1] |= hit
            unmatched = ~pm.matched_gt_flags(pm.spell_overlaps(gs, ps), len(gs), 0.5)
            frag = np.array([(c >= 25).sum() >= 2 for c in covs])
            counts += [len(gs), len(ps), unmatched.sum(), (unmatched & merged).sum(),
                       (unmatched & ~merged & frag).sum()]

    tot, miss, counts = tot / n_runs, miss / n_runs, counts / n_runs
    print("per match-copy (n = {}), labels N = 2 s".format(n_runs))
    print("\n{:<15} {:<7} {:>8} {:>8} {:>6} {:>12}".format("boundary", "D gap", "true", "merged", "rate", "share merged"))
    for k, name in enumerate(CUT_KINDS):
        for g, gname in enumerate(GAP_BINS):
            if tot[k, g] == 0:
                continue
            print("{:<15} {:<7} {:>8.1f} {:>8.1f} {:>6.2f} {:>12.2f}".format(
                name, gname if k else "-", tot[k, g], miss[k, g], miss[k, g] / tot[k, g], miss[k, g] / miss.sum()))
    print("{:<15} {:<7} {:>8.1f} {:>8.1f} {:>6.2f}".format("all", "", tot.sum(), miss.sum(), miss.sum() / tot.sum()))
    n_gt, n_pred, un, un_m, un_f = counts
    print("\nspells: true {:.1f}  predicted {:.1f}".format(n_gt, n_pred))
    print("unmatched true spells @IoU 0.5: {:.1f}  merged {:.1f}  fragmented {:.1f}  other {:.1f}".format(
        un, un_m, un_f, un - un_m - un_f))


def main(mode):
    with open(os.path.join(ROOT, "configs", "model.json")) as handle:
        cfg = json.load(handle)
    if mode == "cuts":
        return cuts(cfg)
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
