"""Paired comparison of two systems on the same items (Miller 2024, "Adding Error Bars to
Evals": compare models on question-level paired differences, with clustered errors).

Both runs are scored with identical, seeded cluster-bootstrap draws, so each bootstrap
replicate gives a paired difference. The result is a difference with a 95% interval and a
two-sided bootstrap p-value per subset, per track, for the OCR/HTR average and Overall.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import subsets as S
from .scoring import N_BOOT, SCORERS
from .taxonomy import Track


def _summary(diff_obs: float, reps: np.ndarray) -> dict:
    lo, hi = np.percentile(reps, [2.5, 97.5])
    p = min(1.0, 2 * min(float((reps <= 0).mean()), float((reps >= 0).mean())))
    if np.all(reps == 0):
        p = 1.0
    return {"diff": round(float(diff_obs), 2), "ci95": [round(float(lo), 2), round(float(hi), 2)],
            "p": round(p, 4), "separable": bool(lo > 0 or hi < 0)}


def compare(manifest: list[dict], preds_a: dict, preds_b: dict, *, supported_a: set[str] | None = None,
            supported_b: set[str] | None = None, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Score A − B. Subsets whose task either system does not support are left out, and
    track / average differences are reported only where both systems cover every subset."""
    by_subset: dict[str, list[dict]] = defaultdict(list)
    for r in manifest:
        by_subset[r["subset"]].append(r)
    subs, reps = {}, {}
    for sid, rows in by_subset.items():
        spec = S.get(sid)
        task = spec.task.value
        if (supported_a is not None and task not in supported_a) or (supported_b is not None and task not in supported_b):
            continue
        ra = SCORERS[spec.task](spec, rows, [preds_a.get(r["id"]) for r in rows], n_boot, seed, return_boot=True)
        rb = SCORERS[spec.task](spec, rows, [preds_b.get(r["id"]) for r in rows], n_boot, seed, return_boot=True)
        reps[sid] = ra["_boot"] - rb["_boot"]
        subs[sid] = {"a": round(ra["score"], 2), "b": round(rb["score"], 2),
                     **_summary(ra["score"] - rb["score"], reps[sid])}

    def combine(subset_ids):
        if not subset_ids or any(s not in subs for s in subset_ids):
            return None
        obs = float(np.mean([subs[s]["a"] - subs[s]["b"] for s in subset_ids]))
        rep = np.mean([reps[s] for s in subset_ids], axis=0)
        return obs, rep

    tracks, track_reps = {}, {}
    for t in Track:
        ids = [s for s in S.TRACK_SUBSETS[t] if s in by_subset]
        got = combine(ids)
        if got:
            tracks[t.value] = _summary(*got)
            track_reps[t.value] = got
        else:
            tracks[t.value] = None

    def mean_tracks(names):
        if any(track_reps.get(t.value) is None for t in names):
            return None
        obs = float(np.mean([track_reps[t.value][0] for t in names]))
        rep = np.mean([track_reps[t.value][1] for t in names], axis=0)
        return _summary(obs, rep)

    return {"subsets": subs, "tracks": tracks,
            "recognition_avg": mean_tracks(S.RECOGNITION_TRACKS), "overall": mean_tracks(list(Track)),
            "n_boot": n_boot, "seed": seed}
