"""Set-piece tags as model input: the challenge's Smart Tagging, and pseudo-tags for the bench.

A tag is (start, stop, team) in frames: a window around a restart, team 0 home / 1 away / -1 N/A.
Only SET PIECES are used: a restart ends a dead-ball stretch, which is the evidence the tracking
lacks most. Phase codes (Build Up, Progression...) are not possessions and have no bench analogue,
so their weight could not be tuned honestly; they are left out.

Bench pseudo-tags come from the dead->live transitions of the DFL labels, then degraded with noise
ranges fixed a priori (never measured on the challenge tags): omissions, false alarms, time jitter,
window width, unknown or wrong team.
"""

import difflib
import os
import sys

import numpy as np

from .bench import D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "starter"))

FPS = 25
CAP_S = 15.0
TAG_SYM = ("in_tag", "tag_dt")
TAG_ANTI = ("tag_team",)

PSEUDO_RANGES = {
    "recall": (0.2, 0.9),
    "false_per_10min": (0.0, 2.0),
    "jitter_s": (0.0, 2.0),
    "half_width_s": (1.0, 3.0),
    "team_na": (0.2, 0.6),
    "team_wrong": (0.0, 0.15),
}


def challenge_tags(home_name):
    """(tags, kickoff_starts): SET PIECES windows from the Smart Tagging, and per half the start
    frame of its first 'Kick Off (Start)' window ({1: frame, 2: frame}, missing if none)."""
    from load_smart_tagging import load
    tags, kickoffs = [], {}
    for event in load():
        if (event["code"] or "").strip().lower() != "set pieces":
            continue
        name = (event["tags"].get("Team") or "").strip()
        if not name or name.upper() == "N/A":
            team = -1
        else:
            team = 0 if difflib.SequenceMatcher(None, name.lower(), home_name.lower()).ratio() >= 0.6 else 1
        tags.append((event["start_frame"], event["end_frame"], team))
        half = {"1st Half": 1, "2nd Half": 2}.get(event["tags"].get("Half"))
        if (event["tags"].get("Type") or "").strip().lower() == "kick off (start)" and half:
            kickoffs.setdefault(half, event["start_frame"])
    return tags, kickoffs


def pseudo_tags(label, period, rng, cfg):
    """Noisy set-piece tags from a bench match's labels (raw DFL labels: restarts are D -> live)."""
    restart = np.flatnonzero((label[:-1] == D) & (label[1:] != D) & (period[:-1] == period[1:])) + 1
    keep = restart[rng.random(len(restart)) < cfg["recall"]]
    live = np.flatnonzero(label != D)
    n_false = rng.poisson(cfg["false_per_10min"] * len(live) / (10 * 60 * FPS))
    centres = np.concatenate([keep, rng.choice(live, n_false)]) if len(live) else keep
    owner = np.concatenate([label[keep], rng.integers(0, 2, n_false)])
    centres = centres + rng.normal(0, cfg["jitter_s"] * FPS, len(centres))
    half = cfg["half_width_s"] * FPS
    team = np.where(rng.random(len(owner)) < cfg["team_wrong"], 1 - owner, owner)
    team = np.where(rng.random(len(owner)) < cfg["team_na"], -1, team)
    order = np.argsort(centres)
    return [(int(round(c - half)), int(round(c + half)), int(t))
            for c, t in zip(centres[order], team[order])]


def sample_pseudo(rng):
    return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in PSEUDO_RANGES.items()}


def tag_features(tags, n):
    """(sym (n, 2), anti (n, 1)): inside a tag window; signed seconds to the nearest window centre
    (NaN if none within CAP_S); team of the covering tag along +1 home / -1 away, 0 otherwise."""
    in_tag = np.zeros(n, np.float32)
    team = np.zeros(n, np.float32)
    dt = np.full(n, np.nan, np.float32)
    best = np.full(n, np.inf)
    t = np.arange(n)
    for start, stop, k in tags:
        a, b = max(start, 0), min(stop, n)
        if a < b:
            in_tag[a:b] = 1
            if k >= 0:
                team[a:b] = 1.0 if k == 0 else -1.0
        centre = (start + stop) / 2
        lo, hi = max(int(centre - CAP_S * FPS), 0), min(int(centre + CAP_S * FPS) + 1, n)
        if lo < hi:
            d = (t[lo:hi] - centre) / FPS
            closer = np.abs(d) < best[lo:hi]
            best[lo:hi] = np.where(closer, np.abs(d), best[lo:hi])
            dt[lo:hi] = np.where(closer, d, dt[lo:hi])
    return np.column_stack([in_tag, dt]), team[:, None]


def with_tags(mid, mode):
    """[(sym, anti)] per degraded copy of a bench match, with tag features appended unless
    mode == "none". "pseudo" = noisy tags from the labels, "shifted" = the same tags moved
    (control). Seeds are fixed per match and copy, so every run sees the same tags."""
    from . import bench
    copies = bench.degraded_features(mid)
    if mode == "none":
        return [(s, a) for s, a, _ in copies]
    m = bench.load_match(mid)
    i = bench.match_ids().index(mid)
    out = []
    for c, (s, a, _) in enumerate(copies):
        rng = np.random.default_rng(500_000 + 1000 * i + c)
        tags = pseudo_tags(m["label"], m["period"], rng, sample_pseudo(rng))
        if mode == "shifted":
            tags = shift_tags(tags, len(s), rng)
        ts, ta = tag_features(tags, len(s))
        out.append((np.hstack([s, ts]), np.hstack([a, ta])))
    return out


def shift_tags(tags, n, rng):
    """Control: every tag moved by one random circular offset -- same count and team mix, no
    information about where the restarts are."""
    off = int(rng.integers(n // 10, n - n // 10))
    return sorted(((s + off) % n, (s + off) % n + (e - s), k) for s, e, k in tags)
