"""Challenge data -> submission CSV, in one command. Deterministic: run it twice, cmp the files.

    python scripts/make_submission.py --team <team_name>

Trains on the external DFL bench only (configs/model.json holds the parameters chosen there by
scripts/bench_eval.py), then runs the trained model on the challenge ATD. The neural owner head
(models/owner_nn_s*.pt, trained on the same DFL bench by src/owner_nn.py; blend weight chosen by
scripts/nn_eval.py) is blended into the HGB owner head when configs/model.json has "owner_nn".
Prints aggregates only.
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import bench  # noqa: E402
from src import features as F  # noqa: E402
from src import possession as P  # noqa: E402
from src.atd import load_atd  # noqa: E402
from games import require, source_frames  # noqa: E402
from src import tagging  # noqa: E402
from baseline import team_names  # noqa: E402  (starter/ on the path via src.atd)
import possession_metrics as pm  # noqa: E402  (path set by src.bench)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", default="submission", help="names the CSV, as SUBMISSION.md asks")
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results"))
    parser.add_argument("--ball-fill", help="ball_vision.npz from scripts/vision_fuse.py: fills "
                        "the ATD's ball gaps with video detections (never overwrites the ATD)")
    args = parser.parse_args()

    with open(os.path.join(ROOT, "configs", "model.json")) as handle:
        cfg = json.load(handle)
    tag_mode = cfg.get("tags", "none")
    heads, stats = P.train(bench.match_ids(), cfg["train_control_s"], cfg["stride"], tag_mode)

    m = load_atd()
    if args.ball_fill:
        with np.load(args.ball_fill) as z:
            vb, filled = z["ball"], z["filled"]
        assert not np.isfinite(m["ball"][filled, 0]).any(), "the fill must only touch ATD gaps"
        m["ball"][filled, :2] = vb[filled]
        print("ball fill: {} frames from video".format(int(filled.sum())))
    sym, anti = F.features(m)
    tags, kickoffs = tagging.challenge_tags(team_names(require(["atd_meta"])[0])[0])
    if tag_mode != "none":
        ts, ta = tagging.tag_features(tags, len(sym))
        sym, anti = np.hstack([sym, ts]), np.hstack([anti, ta])
    q, r = P.predict_heads(heads, sym, anti)
    nn_cfg = cfg.get("owner_nn")
    if nn_cfg:
        from src import owner_nn
        r = P.blend_owner(r, owner_nn.infer(m, [os.path.join(ROOT, p) for p in nn_cfg["models"]]),
                          nn_cfg["weight"])
        print("owner head: HGB + neural ({} seeds, weight {})".format(len(nn_cfg["models"]),
                                                                     nn_cfg["weight"]))

    force_dead = np.zeros(len(q), bool)
    for half, start in kickoffs.items():
        force_dead[(m["period"] == half) & (np.arange(len(q)) < start)] = True
    pred = P.decode(q, r, m["period"], heads, stats, cfg, force_dead)

    path = os.path.join(args.out, args.team + ".csv")
    os.makedirs(args.out, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, pm.LABELS[c]) for i, c in enumerate(pred))

    reason = pm.validate_submission(pm.read_label_csv(path), source_frames("tactical"))
    if reason:
        sys.exit("INVALID {}: {}".format(path, reason))
    runs = [(int(np.flatnonzero(m["period"] == h)[0]), int(np.flatnonzero(m["period"] == h)[-1]) + 1)
            for h in (1, 2)]
    spells = pm.possession_spells(pred, runs)
    graded = m["period"] > 0
    print("wrote {}  ({} rows, valid)".format(os.path.relpath(path, ROOT), len(pred)))
    print("graded frames: H {:.3f}  A {:.3f}  D {:.3f}".format(
        *[(pred[graded] == k).mean() for k in (bench.H, bench.A, bench.D)]))
    print("tags: {} set pieces ({} with a team), kick-off windows {}; forced dead {} frames".format(
        len(tags), sum(k >= 0 for _, _, k in tags), sorted(kickoffs), int(force_dead.sum())))
    print("possessions {}  mean {:.1f} s".format(
        len(spells), np.mean([b - a for a, b, _ in spells]) / 25 if spells else 0))


if __name__ == "__main__":
    main()
