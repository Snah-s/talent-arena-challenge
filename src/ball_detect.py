"""Ball candidates from the tactical video inside the ATD's ball gaps (runs on Colab, GPU).

    python -m src.ball_detect --video TACTICAL.mp4 --segments segments.csv --wasb WASB-SBDT \
        --weights wasb_soccer_best.pth.tar --out OUT_DIR [--tiles 2]

Inference only, with WASB (Tarashima et al., BMVC 2023; github.com/nttcom/WASB-SBDT, code MIT) and
its public soccer weights, trained by its authors on ISSIA-CNR -- external data (team decision and
licence note in docs/PLAN.md s.6). Nothing is trained, tuned or reviewed on the challenge video:
preprocessing and postprocessing follow WASB's own evaluation settings (configs model/wasb,
detector/tracknetv2, transform/default), fixed before seeing any output:
  3 consecutive frames with step 1 (WASB's "oversampling", the setting of its headline results:
  every frame is seen in 3 different triplets and all their candidates are kept) -> centre/scale
  affine to 512 x 288, RGB, ImageNet normalisation -> HRNet -> sigmoid heatmaps -> threshold 0.5
  -> connected components -> heatmap-weighted centroid, score = heatmap sum of the blob.
WASB's online tracker is not used; candidate selection happens locally (src/vision.py).

--tiles 2 (second run, decided after the whole-frame run failed the pipeline check, before seeing
any tiled output): the tactical camera shows far more of the pitch than ISSIA-CNR's cameras, so at
512 x 288 the whole frame shrinks the ball to about a pixel. Each frame is instead cut into a 2 x 2
grid of crops 1/15 larger than a quarter (1024 x 576 on 1920 x 1080, so neighbours overlap), and
each crop goes through the same centre/scale affine to 512 x 288: the ball is ~1.9x larger. All
views' candidates of a frame are kept (TOP_K per view). Everything else is unchanged.

The video is decoded strictly in order, so frame i is the i-th decoded frame, and the total is
checked against the reference count (FAQ.md). Writes OUT_DIR/detections.csv (frame, cx, cy, w, h,
score, peak; original-video pixels; up to TOP_K per frame) and OUT_DIR/status.json. Resumable.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from collections import deque

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

N_FRAMES = 199319                  # tactical video, DATA.md
FRAMES_IN = 3                      # WASB model/wasb.yaml
STEP = 1                           # WASB oversampling (paper s.3.3)
INP_W, INP_H = 512, 288
SCORE_THRESHOLD = 0.5              # WASB detector/tracknetv2.yaml
TOP_K = 5
BATCH = 16                         # triplets per forward pass
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def affine(cx, cy, s):
    """WASB's get_transform: centre c = (cx, cy), scale s, no rotation. Returns the forward 2x3
    matrix and the scale factor k = INP_W / s; the inverse is x = (x' - INP_W/2)/k + cx."""
    k = INP_W / float(s)
    M = np.array([[k, 0, INP_W / 2.0 - k * cx], [0, k, INP_H / 2.0 - k * cy]], np.float32)
    return M, k, cx, cy


def views(width, height, tiles=1):
    """[(M, k, cx, cy)] per view. tiles=1: the whole frame, WASB's own c = (w/2, h/2), s = max(w, h).
    tiles>1: a tiles x tiles grid of crops 1/15 larger than a tile, so neighbours overlap."""
    grow = 1.0 if tiles == 1 else 16.0 / 15.0
    cw, ch = width / tiles * grow, height / tiles * grow
    return [affine(x0 + cw / 2.0, y0 + ch / 2.0, max(cw, ch))
            for y0 in np.linspace(0.0, height - ch, tiles) for x0 in np.linspace(0.0, width - cw, tiles)]


def check_views(vs, width, height):
    """Self-check: each view's forward matrix and the inverse used in blobs() agree, and together
    the views cover the whole frame."""
    covered = np.zeros((height // 8 + 1, width // 8 + 1), bool)
    for M, k, cx, cy in vs:
        p = np.array([cx + 37.0, cy - 21.0])
        q = M @ np.r_[p, 1.0]
        assert np.allclose((q - (INP_W / 2.0, INP_H / 2.0)) / k + (cx, cy), p, atol=1e-3), "inverse"
        x0, x1 = cx - INP_W / 2.0 / k, cx + INP_W / 2.0 / k
        y0, y1 = cy - INP_H / 2.0 / k, cy + INP_H / 2.0 / k
        covered[max(0, int(np.ceil(y0 / 8))):int(y1 // 8) + 1, max(0, int(np.ceil(x0 / 8))):int(x1 // 8) + 1] = True
    assert covered[:(height - 1) // 8 + 1, :(width - 1) // 8 + 1].all(), "views leave part of the frame out"


def triplets(segments_csv, step=STEP):
    """Start frames of the runs of FRAMES_IN consecutive frames inside each segment, every `step`
    frames; the last run is pulled back to end at the segment's end so every frame is covered."""
    starts = set()
    with open(segments_csv) as handle:
        for row in csv.DictReader(handle):
            a, b = int(row["start_frame"]), int(row["stop_frame"])
            if b - a < FRAMES_IN:
                continue
            starts.update(range(a, b - FRAMES_IN + 1, step))
            starts.add(b - FRAMES_IN)
    return sorted(starts)


def load_wasb(repo, weights, device):
    """HRNet from the WASB checkout (loaded by file path, so its package __init__ is not run)."""
    spec = importlib.util.spec_from_file_location("wasb_hrnet", os.path.join(repo, "src", "models", "hrnet.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = OmegaConf.load(os.path.join(repo, "src", "configs", "model", "wasb.yaml"))
    model = mod.HRNet(cfg)
    try:
        ckpt = torch.load(weights, map_location="cpu", weights_only=True)
    except Exception:  # older checkpoints carry non-tensor objects; the file is the authors' own
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def blobs(hm, k, cx, cy):
    """WASB postprocessing on one sigmoid heatmap: [(x, y, w, h, score, peak)] in video pixels."""
    _, binary = cv2.threshold(hm, SCORE_THRESHOLD, 1, cv2.THRESH_BINARY)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8))
    out = []
    for i in range(1, n):
        ys, xs = np.nonzero(labels == i)
        w = hm[ys, xs]
        x, y = (xs * w).sum() / w.sum(), (ys * w).sum() / w.sum()
        out.append(((x - INP_W / 2.0) / k + cx, (y - INP_H / 2.0) / k + cy,
                    stats[i, cv2.CC_STAT_WIDTH] / k, stats[i, cv2.CC_STAT_HEIGHT] / k,
                    float(w.sum()), float(w.max())))
    return sorted(out, key=lambda r: -r[4])[:TOP_K]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def run_batch(model, batch, device, vs):
    """batch: [(start, [3 frames, each a list of preprocessed views])] -> [(frame, rows)] for every
    frame in it, the rows of all views together."""
    x = torch.from_numpy(np.stack([np.concatenate([f[v] for f in fr], 0)
                                   for _, fr in batch for v in range(len(vs))])).to(device)
    hms = model(x)[0].sigmoid().float().cpu().numpy()     # (B * views, 3, H, W)
    hms = hms.reshape(len(batch), len(vs), FRAMES_IN, *hms.shape[2:])
    return [(s + j, [r for v, (_, k, cx, cy) in enumerate(vs) for r in blobs(hms[b, v, j], k, cx, cy)])
            for b, (s, _) in enumerate(batch) for j in range(FRAMES_IN)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--segments", required=True)
    parser.add_argument("--wasb", required=True, help="checkout of github.com/nttcom/WASB-SBDT")
    parser.add_argument("--weights", required=True, help="wasb_soccer_best.pth.tar")
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true", help="first 20 triplets only")
    parser.add_argument("--tiles", type=int, default=1, choices=(1, 2),
                        help="1: whole frame (first run); 2: 2 x 2 overlapping crops")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_wasb(args.wasb, args.weights, device)

    starts = triplets(args.segments)
    if args.smoke:
        starts = starts[:20]
    done_path = os.path.join(args.out, "done_triplets.txt")
    done = set()
    if os.path.exists(done_path):
        with open(done_path) as handle:
            done = {int(line) for line in handle if line.strip()}
    todo = [s for s in starts if s not in done]
    need = {s + j for s in todo for j in range(FRAMES_IN)}
    print("torch {} device {}  {} triplets (step {}), {} to do".format(
        torch.__version__, device, len(starts), STEP, len(todo)), flush=True)

    cap = cv2.VideoCapture(args.video)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vs = views(width, height, args.tiles)
    check_views(vs, width, height)
    per_batch = max(1, BATCH // len(vs))   # network inputs per forward pass stay at BATCH
    print("{} view(s) of {} x {} px per frame, {} triplets per forward pass".format(
        len(vs), round(INP_W / vs[0][1]), round(INP_H / vs[0][1]), per_batch), flush=True)
    det_path = os.path.join(args.out, "detections.csv")
    new = not os.path.exists(det_path)
    det_file, done_file = open(det_path, "a", newline=""), open(done_path, "a")
    writer = csv.writer(det_file)
    if new:
        writer.writerow(["frame", "cx", "cy", "w", "h", "score", "peak"])

    frames, batch, pending = {}, [], deque(todo)
    last = todo[-1] + FRAMES_IN - 1 if todo else -1
    idx, t0, n_done = 0, time.time(), 0
    while True:
        if not cap.grab():
            break
        if idx in need:
            ok, img = cap.retrieve()
            if not ok:
                raise RuntimeError("could not decode frame {}".format(idx))
            frames[idx] = [((cv2.warpAffine(img, M, (INP_W, INP_H), flags=cv2.INTER_LINEAR)[:, :, ::-1]
                             .astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
                           for M, *_ in vs]
        while pending and pending[0] + FRAMES_IN - 1 <= idx:
            s = pending.popleft()
            batch.append((s, [frames[s + j] for j in range(FRAMES_IN)]))
            for f in [f for f in frames if f <= s + FRAMES_IN - 1 and (not pending or f < pending[0])]:
                del frames[f]
        if batch and (len(batch) == per_batch or not pending):
            for f, rows in run_batch(model, batch, device, vs):
                writer.writerows([(f,) + tuple(round(v, 3) for v in r) for r in rows])
            done_file.writelines("{}\n".format(s) for s, _ in batch)
            det_file.flush()
            done_file.flush()
            n_done += len(batch)
            batch = []
            if n_done % 800 < per_batch:
                rate = n_done / (time.time() - t0)
                print("  frame {:>6}  {}/{} triplets  {:.1f}/s  ~{:.0f} min left".format(
                    idx, n_done, len(todo), rate, (len(todo) - n_done) / max(rate, 1e-9) / 60), flush=True)
        idx += 1
        if args.smoke and idx > last:
            break
    det_file.close()
    done_file.close()

    try:
        commit = subprocess.run(["git", "-C", args.wasb, "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
    except OSError:
        commit = None
    status = {"decoded_frames": idx, "expected_frames": N_FRAMES, "complete_decode": not args.smoke,
              "frames_match": (idx == N_FRAMES) if not args.smoke else None,
              "width": width, "height": height, "input": [INP_W, INP_H], "frames_in": FRAMES_IN, "step": STEP, "tiles": args.tiles,
              "score_threshold": SCORE_THRESHOLD, "top_k": TOP_K, "detector": "WASB (HRNet), soccer weights",
              "wasb_commit": commit, "weights_sha256": sha256(args.weights),
              "torch": torch.__version__, "opencv": cv2.__version__, "device": device}
    with open(os.path.join(args.out, "status.json"), "w") as handle:
        json.dump(status, handle, indent=1)
    if not args.smoke and idx != N_FRAMES:
        print("WARNING: decoded {} frames, the reference has {}. Frame indices may be shifted: "
              "do not use these detections.".format(idx, N_FRAMES), flush=True)
    print("done: {} frames decoded, {}".format(idx, det_path), flush=True)


if __name__ == "__main__":
    main()
