"""Neural owner head (plan v3, E4): P(home owns | ball alive) from raw trajectories.

Per 5 fps step, a transformer layer attends over the ball token and both teams' player tokens
(src/nn_inputs.py); the ball token and each team's pooled tokens feed a bidirectional LSTM over
10 s windows (Ball Radar / PathCRF style; partial ball as in TranSPORTmer). Trained on the
degraded DFL bench copies only, loss on live frames. Predictions average the model on the input
and on the team-swapped input, so swapping the teams maps p to 1 - p exactly, like the HGB r head.

Runs on Colab (GPU) with only the exported bench, never challenge data:

    python -m src.owner_nn --data nn_bench.npz --out OUT_DIR

Every fold and the final model are trained once per seed in CFG["seeds"] (plan v3 E5: a seed
ensemble, fixed a priori) and the seeds' logits are averaged. Writes, resumably, one file per fold
and seed (OUT_DIR/oof_<match>_s<seed>.npz, 5 fps logits per degraded copy) and per final seed
(OUT_DIR/owner_nn_s<seed>.pt, trained on all seven matches, for local CPU inference); then
OUT_DIR/oof_rnn.npz with the ensemble P(home | alive) at 25 fps, and OUT_DIR/owner_nn.json.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch import nn

from .nn_inputs import BALL_F, BALL_X, BALL_Y, STEP, TOK_F, TOK_X, TOK_Y, build_inputs, downsample

CFG = {"d": 64, "hidden": 64, "heads": 4, "win": 50, "hop": 25, "epochs": 15,
       "windows_per_epoch": 16384, "batch": 256, "lr": 2e-3, "weight_decay": 1e-4, "seed": 0,
       "seeds": (0, 1, 2)}


class OwnerNet(nn.Module):
    def __init__(self, d=CFG["d"], hidden=CFG["hidden"], heads=CFG["heads"]):
        super().__init__()
        self.tok = nn.Sequential(nn.Linear(TOK_F, d), nn.ReLU(), nn.Linear(d, d))
        self.ball = nn.Sequential(nn.Linear(BALL_F, d), nn.ReLU(), nn.Linear(d, d))
        self.team = nn.Parameter(torch.randn(2, d) * 0.1)
        self.attn = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True)
        self.lstm = nn.LSTM(3 * d, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, tok, mask, ball):
        """tok (B, S, 2, M, F), mask (B, S, 2, M), ball (B, S, Fb) -> logits (B, S)."""
        B, S, _, M, _ = tok.shape
        x = self.tok(tok) + self.team[None, None, :, None, :]
        seq = torch.cat([self.ball(ball).reshape(B * S, 1, -1), x.reshape(B * S, 2 * M, -1)], 1)
        valid = torch.cat([torch.ones(B * S, 1, dtype=torch.bool, device=tok.device),
                           mask.reshape(B * S, 2 * M)], 1)
        h = self.attn(seq, src_key_padding_mask=~valid)
        w = mask.reshape(B * S, 2, M, 1).float()
        pooled = (h[:, 1:].reshape(B * S, 2, M, -1) * w).sum(2) / w.sum(2).clamp(min=1.0)
        frame = torch.cat([h[:, 0], pooled[:, 0], pooled[:, 1]], 1).reshape(B, S, -1)
        out, _ = self.lstm(frame)
        return self.head(out).squeeze(-1)


def _swap(tok, mask):
    return tok.flip(2), mask.flip(2)


def _mirror(tok, ball, cols_tok, cols_ball):
    tok, ball = tok.clone(), ball.clone()
    tok[..., list(cols_tok)] *= -1
    ball[..., list(cols_ball)] *= -1
    return tok, ball


def symmetric_logits(model, tok, mask, ball):
    """Antisymmetric in the teams by construction: f(x) - f(swap x), halved."""
    return 0.5 * (model(tok, mask, ball) - model(*_swap(tok, mask), ball))


def _starts(period, win):
    """Window starts that keep a whole window inside one half."""
    out = []
    for half in (1, 2):
        idx = np.flatnonzero(period == half)
        if len(idx) >= win:
            out.extend(range(idx[0], idx[-1] + 2 - win))
    return np.array(out, np.int64)


def fit(items, device, cfg=CFG, log=print):
    """items: [(tok, mask, ball, y, period)] at 5 fps; y = 1 home, 0 away, -1 ignored."""
    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    win = cfg["win"]
    pool = [(i, s) for i, it in enumerate(items) for s in _starts(it[4], win)]
    model = OwnerNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * (cfg["windows_per_epoch"] // cfg["batch"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=steps)
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    for epoch in range(cfg["epochs"]):
        model.train()
        pick = rng.choice(len(pool), cfg["windows_per_epoch"], replace=False)
        total, count = 0.0, 0
        for b in range(0, len(pick) - cfg["batch"] + 1, cfg["batch"]):
            sel = [pool[k] for k in pick[b:b + cfg["batch"]]]
            tok, mask, ball, y = (torch.from_numpy(np.stack([items[i][j][s:s + win] for i, s in sel]))
                                  .to(device) for j in range(4))
            if rng.random() < 0.5:
                tok, mask = _swap(tok, mask)
                y = torch.where(y >= 0, 1 - y, y)
            if rng.random() < 0.5:
                tok, ball = _mirror(tok, ball, TOK_X, BALL_X)
            if rng.random() < 0.5:
                tok, ball = _mirror(tok, ball, TOK_Y, BALL_Y)
            live = y >= 0
            loss = loss_fn(model(tok, mask, ball)[live], y[live].float()) / live.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total, count = total + loss.item(), count + 1
        log("  epoch {:>2}  loss {:.4f}".format(epoch + 1, total / max(count, 1)))
    return model


@torch.no_grad()
def predict_logits(model, tok, mask, ball, period, device, cfg=CFG):
    """Team-antisymmetric logit of P(home | alive) per 5 fps step: overlapping windows (hop)
    averaged inside each half; 0 (p = 0.5) outside the halves."""
    model.eval()
    win, hop = cfg["win"], cfg["hop"]
    acc, cnt = np.zeros(len(tok)), np.zeros(len(tok))
    for half in (1, 2):
        idx = np.flatnonzero(period == half)
        if not len(idx):
            continue
        lo, hi = idx[0], idx[-1] + 1
        starts = list(range(lo, max(hi - win, lo) + 1, hop))
        if starts[-1] + win < hi:
            starts.append(hi - win)
        for b in range(0, len(starts), 256):
            chunk = starts[b:b + 256]
            n = min(win, hi - lo)
            t, m_, bl = (torch.from_numpy(np.stack([a[s:s + n] for s in chunk])).to(device)
                         for a in (tok, mask, ball))
            logit = symmetric_logits(model, t, m_, bl).cpu().numpy()
            for s, row in zip(chunk, logit):
                acc[s:s + n] += row
                cnt[s:s + n] += 1
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def upsample(p5, n_frames):
    """5 fps -> 25 fps: step i covers frames STEP*i .. STEP*i + STEP - 1."""
    return np.repeat(p5, STEP)[:n_frames]


def infer(m, weight_paths, device="cpu"):
    """P(home | alive) at 25 fps for a match dict (src/atd.py or src/bench.py) from the final
    seed models: their logits averaged, as for the bench ensemble in main."""
    players, team, ball, period, pitch = downsample(m)
    tok, mask, ballf = build_inputs(players, team, ball, pitch)
    zs = []
    for path in weight_paths:
        model = OwnerNet()
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        zs.append(predict_logits(model.to(device), tok, mask, ballf, period, device))
    return upsample(sigmoid(np.mean(zs, axis=0)), len(m["period"]))


def load_bench(path):
    """{(match, copy): (tok, mask, ball, y, period, n_frames)} from export_nn_bench.py's npz."""
    z = np.load(path)
    out = {}
    for key in sorted(k[:-len("_players")] for k in z.files if k.endswith("_players")):
        mid, c = key.rsplit("_", 1)
        tok, mask, ball = build_inputs(z[key + "_players"], z[key + "_team"], z[key + "_ball"],
                                       z[key + "_pitch"])
        y = z[key + "_label"].astype(np.int64)
        y = np.where(y == 0, 1, np.where(y == 1, 0, -1))
        out[(mid, int(c))] = (tok, mask, ball, y, z[key + "_period"], int(z[key + "_n_frames"]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="one tiny fold end to end, to check the pipeline before the real run")
    args = parser.parse_args()
    cfg = dict(CFG, epochs=1, windows_per_epoch=1024) if args.smoke else CFG
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    print("torch {}  device {}  cfg {}".format(torch.__version__, device, cfg), flush=True)

    data = load_bench(args.data)
    mids = sorted({mid for mid, _ in data})
    seeds = cfg["seeds"][:1] if args.smoke else cfg["seeds"]

    def live_acc(z, y):
        live = y >= 0
        return ((z[live] > 0) == (y[live] == 1)).mean()

    for held in (mids[:1] if args.smoke else mids + ["ALL"]):
        for seed in seeds:
            path = os.path.join(args.out, "owner_nn_s{}.pt".format(seed) if held == "ALL"
                                else "oof_{}_s{}.npz".format(held, seed))
            if os.path.exists(path):
                print("skip {} seed {} (done)".format(held, seed), flush=True)
                continue
            t0 = time.time()
            print("fold {} seed {}".format(held, seed), flush=True)
            model = fit([v[:5] for (mid, _), v in data.items() if mid != held], device,
                        dict(cfg, seed=seed), log=lambda s: print(s, flush=True))
            tmp = path + ".tmp"
            if held == "ALL":
                torch.save(model.state_dict(), tmp)
            else:
                logits, acc = {}, []
                for (mid, c), (tok, mask, ball, y, period, _) in data.items():
                    if mid == held:
                        z = predict_logits(model, tok, mask, ball, period, device, cfg)
                        acc.append(live_acc(z, y))
                        logits["z{}".format(c)] = z.astype(np.float32)
                with open(tmp, "wb") as handle:
                    np.savez_compressed(handle, **logits)
                print("  held-out live accuracy (5 fps) {:.4f}".format(np.mean(acc)), flush=True)
            os.replace(tmp, path)
            print("  {:.0f} s".format(time.time() - t0), flush=True)
    if args.smoke:
        print("smoke OK", flush=True)
        return

    merged, single, ens = {}, [], []
    for (mid, c), (_, _, _, y, _, n) in sorted(data.items()):
        zs = []
        for seed in seeds:
            with np.load(os.path.join(args.out, "oof_{}_s{}.npz".format(mid, seed))) as z:
                zs.append(z["z{}".format(c)])
        single.append(np.mean([live_acc(z, y) for z in zs]))
        ens.append(live_acc(np.mean(zs, 0), y))
        merged["{}_r{}".format(mid, c)] = upsample(sigmoid(np.mean(zs, 0)), n).astype(np.float32)
    np.savez_compressed(os.path.join(args.out, "oof_rnn.npz"), **merged)
    with open(os.path.join(args.out, "owner_nn.json"), "w") as handle:
        json.dump(dict(cfg, torch=torch.__version__, device=device), handle, indent=1)
    print("held-out live accuracy (5 fps): single seed {:.4f}  ensemble of {} {:.4f}".format(
        np.mean(single), len(seeds), np.mean(ens)), flush=True)
    print("done: oof_rnn.npz, owner_nn_s*.pt, owner_nn.json", flush=True)


if __name__ == "__main__":
    main()
