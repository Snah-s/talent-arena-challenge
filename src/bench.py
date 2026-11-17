"""External tuning bench: Sportec Open DFL matches (Bassek et al. 2025, CC BY 4.0).

Everything that is trained or tuned is trained or tuned here, never on the challenge game.

A match is a plain dict of numpy arrays, the same shape the ATD loader produces:

    players  (T, K, 2) float32  metres, origin at a pitch corner, NaN = not observed
    team     (K,)      int8     0 home, 1 away
    ball     (T, 3)    float32  x, y, z in metres (z NaN where unknown, always for ATD)
    period   (T,)      int8     1 or 2
    label    (T,)      int8     scorer codes H=0 A=1 D=2 X=3
    pitch    (2,)      float32  length, width

Labels follow the challenge ground truth's own derivation (CHALLENGE.md): D where the corrected
tracking says the ball is dead, otherwise the team it names as owner.
"""

import glob
import json
import os
import sys

import numpy as np
from lxml import etree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

DFL_DIR = os.path.join(ROOT, "external", "dfl")
H, A, D, X = 0, 1, 2, 3
SECTIONS = {"firstHalf": 1, "secondHalf": 2}

DEGRADE_RANGES = {
    "ball_missing": (0.20, 0.60),
    "ball_gap_s": (0.5, 4.0),
    "w_aerial": (0.5, 2.5),
    "w_crowd": (0.5, 2.5),
    "w_dead": (0.0, 2.0),
    "player_missing": (0.02, 0.15),
    "player_gap_s": (0.5, 10.0),
    "noise_player_m": (0.05, 0.5),
    "noise_ball_m": (0.05, 0.5),
    "dup_frac": (0.0, 0.05),
    "phantom_frac": (0.0, 0.03),
}
N_DUP = 2


def _match_files(match_id):
    raw = os.path.join(DFL_DIR, "raw")
    find = lambda kind: glob.glob(os.path.join(raw, "DFL_*_{}_*{}.xml".format(kind, match_id)))[0]
    return find("matchinformation"), find("positions_raw_observed")


def _team_ids(info_path):
    """{TeamId: 0 home | 1 away} from the match information file."""
    root = etree.parse(info_path).getroot()
    return {t.get("TeamId"): 0 if t.get("Role") == "home" else 1 for t in root.iter("Team")}


def parse_dfl(match_id):
    """Stream one positions file into a match dict. ~1 min and a few GB peak for 350 MB of XML."""
    info_path, pos_path = _match_files(match_id)
    team_of = _team_ids(info_path)
    if sorted(team_of.values()) != [0, 1]:
        raise ValueError("{}: expected one home and one guest team".format(info_path))

    pitch = None
    tracks = {}
    current = None
    for event, el in etree.iterparse(pos_path, events=("start", "end"),
                                     tag=("PitchSize", "FrameSet", "Frame")):
        if event == "start":
            if el.tag == "FrameSet":
                period = SECTIONS.get(el.get("GameSection"))
                team = el.get("TeamId")
                if period is None or (team != "BALL" and team not in team_of):
                    current = None
                else:
                    key = (period, "BALL" if team == "BALL" else el.get("PersonId"))
                    current = tracks.setdefault(key, [])
                    if team != "BALL":
                        tracks.setdefault(("team", key[1]), team_of[team])
            continue
        if el.tag == "PitchSize":
            pitch = (float(el.get("X")), float(el.get("Y")))
        elif el.tag == "Frame" and current is not None:
            g = el.get
            current.append((int(g("N")), float(g("X")), float(g("Y")), float(g("Z") or "nan"),
                            int(g("BallStatus") or -1), int(g("BallPossession") or -1)))
        el.clear(keep_tail=False)
        while el.getprevious() is not None:
            del el.getparent()[0]

    persons = sorted({k[1] for k in tracks if k[0] == "team"})
    team = np.array([tracks[("team", p)] for p in persons], dtype=np.int8)
    length, width = pitch

    blocks = []
    for period in (1, 2):
        ball = np.array(tracks[(period, "BALL")], dtype=np.float64)
        n0, n1 = int(ball[:, 0].min()), int(ball[:, 0].max())
        n = n1 - n0 + 1
        b = np.full((n, 3), np.nan, np.float32)
        idx = ball[:, 0].astype(int) - n0
        b[idx] = ball[:, 1:4] + (length / 2, width / 2, 0)
        status = np.full(n, -1, np.int8)
        owner = np.full(n, -1, np.int8)
        status[idx], owner[idx] = ball[:, 4], ball[:, 5]
        p = np.full((n, len(persons), 2), np.nan, np.float32)
        for k, person in enumerate(persons):
            rows = tracks.get((period, person))
            if not rows:
                continue
            r = np.array(rows, dtype=np.float64)
            keep = (r[:, 0] >= n0) & (r[:, 0] <= n1)
            p[r[keep, 0].astype(int) - n0, k] = r[keep, 1:3] + (length / 2, width / 2)
        blocks.append((b, p, status, owner, period))

    ball = np.concatenate([b for b, *_ in blocks])
    players = np.concatenate([p for _, p, *_ in blocks])
    status = np.concatenate([s for *_, s, _, _ in blocks])
    owner = np.concatenate([o for *_, o, _ in blocks])
    period = np.concatenate([np.full(len(b), per, np.int8) for b, *_, per in blocks])
    label = dfl_labels(status, owner)
    return {"players": players, "team": team, "ball": ball, "period": period, "label": label,
            "pitch": np.array(pitch, np.float32), "match_id": match_id,
            "unknown_owner": np.int64(((status != 0) & ~np.isin(owner, (1, 2))).sum())}


def dfl_labels(status, owner):
    """D where BallStatus is 0, else the BallPossession team. Unknown owners inherit the last."""
    label = np.where(status == 0, D, np.where(owner == 1, H, np.where(owner == 2, A, X)))
    label = label.astype(np.int8)
    unknown = label == X
    if unknown.any():
        carry = np.maximum.accumulate(np.where(~unknown, np.arange(len(label)), 0))
        label = label[carry]
    return label


CONTROL_S = 2.0
CONTROL_SENSITIVITY_S = (1.0, 2.0, 3.0)


def controlled_labels(label, period, min_s=CONTROL_S, fps=25):
    """DFL labels rewritten with the challenge's possession rule (CHALLENGE.md).

    DFL hands the ball to whoever touches it; the challenge counts a change only on *controlled*
    possession. Tracking-only proxy: a live switch counts if the new team keeps the ball for
    min_s, otherwise the previous team keeps possession (touch, deflection, lost aerial duel).
    After a dead ball the restarting team always has it ("there is always a clear possessor").
    The pass-count proxy is not used: DFL event times are not frame-synchronised.
    """
    out = label.copy()
    min_len = int(round(min_s * fps))
    for half in (1, 2):
        idx = np.flatnonzero(period == half)
        if not len(idx):
            continue
        lab = label[idx]
        breaks = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        current, prev_dead = None, True
        for a, b in zip(np.concatenate(([0], breaks)), np.concatenate((breaks, [len(lab)]))):
            k = lab[a]
            if k == D:
                prev_dead = True
                continue
            if not (prev_dead or k == current or b - a >= min_len):
                out[idx[a:b]] = current
            else:
                current = k
            prev_dead = False
    return out


def load_match(match_id):
    """Cached parse of one DFL match."""
    cache = os.path.join(DFL_DIR, "cache", match_id + ".npz")
    if not os.path.exists(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        m = parse_dfl(match_id)
        np.savez_compressed(cache, **{k: v for k, v in m.items() if k != "match_id"})
    with np.load(cache) as z:
        m = {k: z[k] for k in z.files}
    m["match_id"] = match_id
    return m


def match_ids():
    paths = glob.glob(os.path.join(DFL_DIR, "raw", "DFL_04_03_positions_raw_observed_*.xml"))
    return sorted(os.path.basename(p).rsplit("-", 1)[1][:-4] for p in paths)


def sample_degradation(rng):
    return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in DEGRADE_RANGES.items()}


def _gaps(weight, target, mean_len, rng):
    """Boolean mask of bursty gaps covering ~target of the frames, starting where weight is high.

    Gap starts are drawn with probability base * weight and lengths are geometric; base is
    bisected so the covered fraction hits the target. Common random numbers keep the coverage
    monotone in base, so the bisection is well posed.
    """
    n = len(weight)
    u = rng.random(n)
    lengths = rng.geometric(1.0 / max(mean_len, 1.0), n)
    t = np.arange(n)

    def mask(base):
        ends = np.where(u < base * weight, t + lengths, 0)
        reach = np.maximum.accumulate(ends)
        return reach > t

    lo, hi = 0.0, 1.0 / weight.min()
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mask(mid).mean() < target else (lo, mid)
    return mask(hi)


def degrade(m, cfg, rng, fps=25):
    """A copy of a match that looks like automatic tracking. Labels are never touched."""
    ball, players = m["ball"].copy(), m["players"].copy()
    n, k = players.shape[:2]

    d = np.linalg.norm(players - ball[:, None, :2], axis=2)
    crowd = (d < 3.0).sum(axis=1) >= 3
    aerial = np.nan_to_num(ball[:, 2]) > 1.5
    dead = m["label"] == D
    weight = np.exp(cfg["w_aerial"] * aerial + cfg["w_crowd"] * crowd + cfg["w_dead"] * dead)
    lost = _gaps(weight, cfg["ball_missing"], cfg["ball_gap_s"] * fps, rng)

    ball[:, :2] += rng.normal(0, cfg["noise_ball_m"], (n, 2))
    ball[lost, :2] = np.nan
    ball[:, 2] = np.nan

    players += rng.normal(0, cfg["noise_player_m"], players.shape).astype(np.float32)
    flat = np.ones(n)
    for j in range(k):
        gone = _gaps(flat, cfg["player_missing"], cfg["player_gap_s"] * fps, rng)
        players[gone, j] = np.nan

    team = np.concatenate([m["team"], np.arange(N_DUP) % 2]).astype(np.int8)
    dup = np.full((n, N_DUP, 2), np.nan, np.float32)
    for j in range(N_DUP):
        on = _gaps(flat, cfg["dup_frac"] / N_DUP, 2 * fps, rng) if cfg["dup_frac"] > 0 else flat < 0
        burst = np.cumsum(np.diff(np.concatenate(([0], on.astype(np.int8)))) == 1)
        pool = np.flatnonzero(m["team"] == j % 2)
        src = rng.choice(pool, burst.max() + 1)[burst]
        offset = rng.normal(0, 0.7, (burst.max() + 1, 2))[burst]
        dup[on, j] = players[np.flatnonzero(on), src[on]] + offset[on]
    players = np.concatenate([players, dup], axis=1)

    if cfg["phantom_frac"] > 0:
        ghost = _gaps(flat, cfg["phantom_frac"], 0.5 * fps, rng)
        burst = np.cumsum(np.diff(np.concatenate(([0], ghost.astype(np.int8)))) == 1)
        spot = rng.uniform((0, 0), m["pitch"], (burst.max() + 1, 2))[burst]
        ball[ghost, :2] = spot[ghost] + rng.normal(0, 0.3, (int(ghost.sum()), 2))

    out = dict(m)
    out.update(ball=ball, players=players, team=team)
    return out


def score(gt, pred, period):
    """The official evaluate() on a bench match, with an X frame between halves so no possession
    can straddle halftime (valid_runs splits on GT X, exactly as on the challenge game)."""
    cut = int(np.argmax(period == 2))
    gt = np.insert(np.asarray(gt, np.int8), cut, X)
    pred = np.insert(np.asarray(pred, np.int8), cut, X)
    return pm.evaluate(gt, pred)


COPIES = 2


def degraded_features(mid):
    """[(sym, anti, degradation cfg)] for each degraded copy of a bench match, cached on disk.
    Seeds depend only on the match's place in the sorted bench and the copy number."""
    from .features import features
    i = match_ids().index(mid)
    out = []
    for c in range(COPIES):
        path = os.path.join(DFL_DIR, "cache", "feat_{}_{}.npz".format(mid, c))
        if not os.path.exists(path):
            rng = np.random.default_rng(1000 * i + c)
            cfg = sample_degradation(rng)
            sym, anti = features(degrade(load_match(mid), cfg, rng))
            np.savez_compressed(path, sym=sym, anti=anti, cfg=json.dumps(cfg))
        with np.load(path) as z:
            out.append((z["sym"], z["anti"], json.loads(str(z["cfg"]))))
    return out


def quick_score(gt, pred, period):
    """(spellF1, macroF1) only -- the two board numbers, without evaluate()'s extras."""
    cut = int(np.argmax(period == 2))
    gt = np.insert(np.asarray(gt, np.int8), cut, X)
    pred = np.insert(np.asarray(pred, np.int8), cut, X)
    runs = pm.valid_runs(gt)
    spell = pm.spell_scores(gt, pm.fill_abstentions(pred, runs), runs)["mean_f1"]
    return spell, pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pred)))


if __name__ == "__main__":
    for mid in match_ids():
        m = load_match(mid)
        lab = m["label"]
        gt = np.insert(lab, int(np.argmax(m["period"] == 2)), X)
        n_spells = len(pm.possession_spells(gt, pm.valid_runs(gt)))
        seen = np.isfinite(m["players"][..., 0]).sum(1)
        print("{}  frames {:>6}  H {:.2f} A {:.2f} D {:.2f}  possessions {:>4} (mean {:.1f} s)  "
              "ball seen {:.3f}  players/frame {:.1f}  K {}  unknown owner {}".format(
                  mid, len(lab), (lab == H).mean(), (lab == A).mean(), (lab == D).mean(),
                  n_spells, (lab != D).sum() / max(n_spells, 1) / 25,
                  np.isfinite(m["ball"][:, 0]).mean(), seen.mean(), len(m["team"]),
                  int(m["unknown_owner"])))
