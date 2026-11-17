# Build Up Viz — More than a Hack 2027, Phase 1: Ball Possession

Our solution to the Metrica Sports / FC Barcelona possession challenge: for every frame of the
tactical video, who has the ball (`H` / `A` / `D` / `X`). The challenge's own README, data
description and rules are kept unchanged in [README_CHALLENGE.md](README_CHALLENGE.md),
[DATA.md](DATA.md), [CHALLENGE.md](CHALLENGE.md), [RULES.md](RULES.md) and
[SUBMISSION.md](SUBMISSION.md).

The technical document is [report/Build_Up_Viz.pdf](report/Build_Up_Viz.pdf). Its LaTeX source and
the d3.js figure generator are in `report/`. To rebuild it, run `make -C report`, which needs
Node.js, `rsvg-convert` and `latexmk`.

## The approach in one screen

```
challenge ATD ─┐                          ┌─ q = P(ball alive)        (gradient boosting)
Smart Tagging ─┼─> per-frame features ────┤
ball fill ─────┘   (team-symmetric)       └─ r = P(home owns | alive) (gradient boosting
  ATD ball + WASB detections                                          + neural owner head)
  in the ATD's gaps                                   │
                                                      v
                              4-state HMM (live/dead × home/away) with minimum durations,
                              Viterbi per half  ──>  frame,label CSV
```

Everything that is trained or tuned is trained or tuned on an **external** bench: the seven
Sportec Open DFL Bundesliga matches (CC BY 4.0). On the challenge game the trained model is only
run, never fitted.

## Installation

Python 3.11 (tested with 3.11.15).

```bash
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-solution.txt
python -m pytest -q        # 24 tests
```

## Data

1. **Challenge data** (gated, not in this repository): unpack the bundle into `data/` as described
   in [data/README.md](data/README.md). Only `data/tactical/` is used.
2. **DFL bench** (public, CC BY 4.0, [figshare 28196177](https://doi.org/10.6084/m9.figshare.28196177)):
   21 files, ~2.5 GB, into `external/dfl/raw/`, each checked against its MD5:

   ```bash
   mkdir -p external/dfl/raw
   while read url name md5; do
     curl -sL -o "external/dfl/raw/$name" "$url"
     echo "$md5  external/dfl/raw/$name" | md5sum -c
   done < configs/dfl_files.txt
   ```

## Reproducing the submission

```bash
python scripts/vision_fuse.py
python scripts/make_submission.py --team Build_Up_Viz --ball-fill external/vision/ball_vision.npz
python evaluation/validate_submission.py evaluation/results/Build_Up_Viz.csv
```

`make_submission.py` trains the gradient-boosted heads on the DFL bench (cached under
`external/dfl/cache/` after the first run), loads the neural owner head from `models/`, runs on the
challenge ATD and writes `evaluation/results/Build_Up_Viz.csv`. It is deterministic: run it twice
and `cmp` the files. A run from a clean clone, with empty caches, reproduces the submitted file byte
for byte.

**Ball detections (intermediate file).** `vision_fuse.py` reads `external/vision/out/detections.csv`
and `status.json`: ball candidates detected by WASB in the tactical video. They are a derivative of
the challenge video, so they are not published here. We provide them to the organisers on request,
as an intermediate file (allowed by the rules). To regenerate them:

1. `python scripts/vision_segments.py` writes `external/vision/vision_bundle.zip`, which holds the
   video stretches where the ATD has no ball, plus `src/ball_detect.py`.
2. Run [notebooks/vision_ball.ipynb](notebooks/vision_ball.ipynb) on a GPU (we used Google Colab, T4).
   It runs inference only, with WASB (commit `923462ca`) and its public soccer weights
   (sha256 `d0369572…`). Both are recorded in `status.json`. The video decode is checked against the
   reference frame count, and `vision_fuse.py` refuses the detections otherwise.

GPU inference is not guaranteed to be bit-identical across GPU models, so regenerated detections
may differ slightly from ours. With the provided intermediate file, the CSV is identical.

**Without the detections**, `python scripts/make_submission.py --team Build_Up_Viz` (no
`--ball-fill`) needs only the challenge data and the DFL bench. It gives our previous version (v7),
fully reproducible.

## Retraining and evaluation on the bench (optional)

| Command | What |
|---|---|
| `python scripts/bench_eval.py` | leave-one-match-out grid for the decoder parameters → `configs/model.json` |
| `python scripts/export_nn_bench.py` + [notebooks/e4_owner_nn.ipynb](notebooks/e4_owner_nn.ipynb) | trains the neural owner head on the DFL bench (GPU) → `models/owner_nn_s*.pt` |
| `python scripts/nn_eval.py` | out-of-fold evaluation of the neural head blend |
| `python scripts/diagnose_bench.py` | oracle diagnostics (perfect ball, perfect alive/dead, perfect owner) |

## Layout

| Path | Contents |
|---|---|
| `src/bench.py` | DFL loader, label rule, bench degradation, scoring |
| `src/atd.py`, `src/tagging.py` | challenge ATD and Smart Tagging in the bench's format |
| `src/features.py`, `src/possession.py` | features, gradient-boosted heads, HMM + Viterbi |
| `src/nn_inputs.py`, `src/owner_nn.py` | neural owner head |
| `src/ball_detect.py`, `src/vision.py` | WASB inference (GPU) and the fusion rules (local) |
| `scripts/` | entry points listed above |
| `configs/model.json`, `models/` | chosen parameters and the trained neural head |
| `evaluation/`, `starter/`, `games.py` | the organisers' scorer and starter kit, unchanged |

## Licences and attribution

Code: MIT, as the organisers' kit ([LICENSE](LICENSE)). DFL data: Bassek et al. (2025), *An
integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data, CC BY 4.0.
The weights in `models/` are trained only on that data. WASB: Tarashima et al., BMVC 2023,
[nttcom/WASB-SBDT](https://github.com/nttcom/WASB-SBDT) (MIT code). The weights are not
redistributed here; see the technical document for their licence note.
