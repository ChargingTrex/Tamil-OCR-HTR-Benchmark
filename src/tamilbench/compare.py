"""Paired comparison of two systems on the same items (Miller 2024, "Adding Error Bars to
Evals": compare models on question-level paired differences, with clustered errors).

Both runs are scored with identical, seeded cluster-bootstrap draws, so each bootstrap
replicate gives a paired difference. The result is a difference with a 95% interval and a
two-sided bootstrap p-value per subset, per track, per script stage, per era, for the
OCR/HTR score and for Overall — every aggregate computed with the scorer's own formulas.
"""

from __future__ import annotations

from collections import defaultdict

from . import subsets as S
from .scoring import N_BOOT, _script, aggregate, aggregation_input, paired_summary, scorer_for


def compare(manifest: list[dict], preds_a: dict, preds_b: dict, *, supported_a: set[str] | None = None,
            supported_b: set[str] | None = None, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Score A − B. Subsets whose task either system does not support are left out, and an
    aggregate difference is reported only where both systems cover all of its subsets."""
    by_subset: dict[str, list[dict]] = defaultdict(list)
    for r in manifest:
        by_subset[r["subset"]].append(r)
    layout = {sid: sorted({_script(r) for r in rows}) for sid, rows in by_subset.items()}
    subs = {}
    obs_a, obs_b, rep_a, rep_b = {}, {}, {}, {}
    for sid, rows in by_subset.items():
        spec = S.get(sid)
        task = spec.task.value
        if (supported_a is not None and task not in supported_a) or (supported_b is not None and task not in supported_b):
            obs_a[sid] = obs_b[sid] = rep_a[sid] = rep_b[sid] = None
            continue
        scorer = scorer_for(spec)
        ra = scorer(spec, rows, [preds_a.get(r["id"]) for r in rows], n_boot, seed, return_boot=True)
        rb = scorer(spec, rows, [preds_b.get(r["id"]) for r in rows], n_boot, seed, return_boot=True)
        subs[sid] = {"a": round(ra["score"], 2), "b": round(rb["score"], 2),
                     **paired_summary(ra["score"] - rb["score"], ra["_boot"] - rb["_boot"])}
        obs_a[sid], rep_a[sid] = aggregation_input(ra), aggregation_input(ra, boot=True)
        obs_b[sid], rep_b[sid] = aggregation_input(rb), aggregation_input(rb, boot=True)

    A, B = aggregate(obs_a, layout), aggregate(obs_b, layout)
    Ab, Bb = aggregate(rep_a, layout), aggregate(rep_b, layout)

    def diff(get):
        a, b = get(A), get(B)
        if a is None or b is None:
            return None
        return {"a": round(float(a), 2), "b": round(float(b), 2), **paired_summary(a - b, get(Ab) - get(Bb))}

    return {
        "subsets": subs,
        "tracks": {t: diff(lambda X, t=t: X["tracks"][t]) for t in A["tracks"]},
        "scripts": {sc: diff(lambda X, sc=sc: X["scripts"][sc]["score"]) for sc in A["scripts"]},
        "eras": {e: {k: diff(lambda X, e=e, k=k: X["eras"][e].get(k))
                     for k in ("reading", "identification", "translation", "composite") if k in A["eras"][e]}
                 for e in A["eras"]},
        "recognition_avg": diff(lambda X: X["recognition_avg"]),
        "overall": diff(lambda X: X["overall"]),
        "n_boot": n_boot, "seed": seed,
    }
