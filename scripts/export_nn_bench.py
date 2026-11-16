"""Export the degraded DFL bench copies for training the neural owner head on Colab (plan v3, E4).

    python scripts/export_nn_bench.py

Rebuilds every degraded copy with the seeds of bench.degraded_features (checked against its
cache), keeps the 5 fps view plus the labels under the training control rule, and zips it with
the two source files Colab needs: external/nn/colab_bundle.zip. Only DFL data (CC BY 4.0), never
anything from the challenge.
"""

import json
import os
import sys
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import bench  # noqa: E402
from src import features as F  # noqa: E402
from src.nn_inputs import STEP, downsample  # noqa: E402

OUT = os.path.join(ROOT, "external", "nn")


def main():
    with open(os.path.join(ROOT, "configs", "model.json")) as handle:
        cfg = json.load(handle)
    os.makedirs(OUT, exist_ok=True)
    arrays = {}
    for i, mid in enumerate(bench.match_ids()):
        m = bench.load_match(mid)
        label = bench.controlled_labels(m["label"], m["period"], cfg["train_control_s"])
        cached = bench.degraded_features(mid)
        for c in range(bench.COPIES):
            rng = np.random.default_rng(1000 * i + c)
            d = bench.degrade(m, bench.sample_degradation(rng), rng)
            seen = cached[c][0][:, F.SYM.index("seen")] > 0
            assert (np.isfinite(d["ball"][:, 0]) == seen).all(), "copy differs from the cache"
            players, team, ball, period, pitch = downsample(d)
            key = "{}_{}".format(mid, c)
            arrays.update({key + "_players": players.astype(np.float32), key + "_team": team,
                           key + "_ball": ball.astype(np.float32), key + "_period": period,
                           key + "_pitch": pitch, key + "_label": label[::STEP],
                           key + "_n_frames": np.int64(len(label))})
        print(mid, flush=True)
    data = os.path.join(OUT, "nn_bench.npz")
    np.savez_compressed(data, **arrays)
    bundle = os.path.join(OUT, "colab_bundle.zip")
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(data, "nn_bench.npz")
        for name in ("src/nn_inputs.py", "src/owner_nn.py", "tests/test_nn_inputs.py"):
            z.write(os.path.join(ROOT, name), name)
    print("wrote {} ({:.0f} MB)".format(os.path.relpath(bundle, ROOT), os.path.getsize(bundle) / 1e6))


if __name__ == "__main__":
    main()
