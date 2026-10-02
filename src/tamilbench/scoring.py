"""Turn a manifest + a model's predictions into scores.

Scores are reported per subset (with a 95% bootstrap confidence interval), per track
(unweighted mean of the track's subsets) and overall (unweighted mean of the eight tracks).
A model that does not support a task (e.g. a classical OCR engine asked to translate) gets
``None`` for that subset — never a zero — and therefore no Overall score; its Recognition
average is still reported. Missing predictions for supported tasks count as empty answers.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import subsets as subset_registry
from .metrics import classification as cls_m
from .metrics import recognition as rec_m
from .metrics import translation as tr_m
from .taxonomy import Task, Track

N_BOOT = 1000


def _ci(values: np.ndarray) -> list[float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [round(float(lo), 2), round(float(hi), 2)]


def _boot_idx(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def score_recognition(spec, rows, preds, n_boot, seed) -> dict:
    policy = rec_m.TextPolicy.from_dict(spec.policy)
    stats = [rec_m.sample_stats(r[spec.target], p or "", policy) for r, p in zip(rows, preds)]
    res = rec_m.aggregate(stats)
    e = np.array([s["char_edits"] for s in stats], dtype=float)
    c = np.array([s["chars"] for s in stats], dtype=float)
    idx = _boot_idx(len(stats), n_boot, seed)
    boot = 100.0 * np.clip(1.0 - e[idx].sum(1) / np.maximum(1, c[idx].sum(1)), 0, None)
    res["ci95"] = _ci(boot)
    # Prior-reliance slice: real text vs phonotactic nonce words.
    slices = defaultdict(list)
    for r, s in zip(rows, stats):
        slices[r.get("lexical", "corpus")].append(s)
    res["slices"] = {k: round(rec_m.score_value(v), 2) for k, v in slices.items()}
    res["per_sample"] = [{"id": r["id"], "cer": round(s["char_edits"] / max(1, s["chars"]), 4)}
                         for r, s in zip(rows, stats)]
    return res


def score_classification(spec, rows, preds, n_boot, seed) -> dict:
    labels = list(spec.labels)
    gold = [r["label"] for r in rows]
    pred = [cls_m.parse_label(p or "", labels) for p in preds]
    res = cls_m.scores(gold, pred, labels)
    idx = _boot_idx(len(gold), n_boot, seed)
    g, p = np.array(gold), np.array(pred)
    boot = [cls_m.scores(list(g[i]), list(p[i]), labels)["score"] for i in idx]
    res["ci95"] = _ci(np.array(boot))
    conf = defaultdict(lambda: defaultdict(int))
    for a, b in zip(gold, pred):
        conf[a][b] += 1
    res["confusion"] = {a: dict(b) for a, b in conf.items()}
    res["per_sample"] = [{"id": r["id"], "pred": q, "ok": q == r["label"]} for r, q in zip(rows, pred)]
    return res


def score_translation(spec, rows, preds, n_boot, seed) -> dict:
    refs = [r[spec.target] for r in rows]
    hyps = [p or "" for p in preds]
    st = np.array([tr_m.sentence_stats(h, r) for h, r in zip(hyps, refs)], dtype=float)
    chrf = tr_m.f_score(list(st.sum(0)))
    idx = _boot_idx(len(refs), n_boot, seed)
    boot = np.array([tr_m.f_score(list(st[i].sum(0))) for i in idx])
    per = [round(tr_m.f_score(list(s)), 2) for s in st]
    return {"n": len(refs), "chrf_pp": chrf, "score": chrf, "ci95": _ci(boot),
            "per_sample": [{"id": r["id"], "chrf": v} for r, v in zip(rows, per)]}


SCORERS = {Task.RECOGNITION: score_recognition, Task.SCRIPT_ID: score_classification,
           Task.MEDIUM_ID: score_classification, Task.TRANSLATION: score_translation}


def score_run(manifest: list[dict], predictions: dict[str, str | None], *,
              supported_tasks: set[str] | None = None, n_boot: int = N_BOOT, seed: int = 0,
              keep_per_sample: bool = True) -> dict:
    by_subset: dict[str, list[dict]] = defaultdict(list)
    for row in manifest:
        by_subset[row["subset"]].append(row)

    out_subsets: dict[str, dict | None] = {}
    for sid, rows in by_subset.items():
        spec = subset_registry.get(sid)
        if supported_tasks is not None and spec.task.value not in supported_tasks:
            out_subsets[sid] = None
            continue
        preds = [predictions.get(r["id"]) for r in rows]
        res = SCORERS[spec.task](spec, rows, preds, n_boot, seed)
        res["missing"] = sum(p is None for p in preds)
        if not keep_per_sample:
            res.pop("per_sample", None)
        out_subsets[sid] = res

    tracks: dict[str, float | None] = {}
    for track in Track:
        ids = [s for s in subset_registry.TRACK_SUBSETS[track] if s in by_subset]
        vals = [out_subsets[s]["score"] for s in ids if out_subsets.get(s) is not None]
        tracks[track.value] = round(float(np.mean(vals)), 2) if ids and len(vals) == len(ids) else None

    def mean_of(names):
        vals = [tracks[t.value] for t in names]
        return round(float(np.mean(vals)), 2) if all(v is not None for v in vals) else None

    nonce = [v["slices"].get("nonce") for v in out_subsets.values()
             if v and "slices" in v and "nonce" in v["slices"]]
    real = [v["slices"].get("corpus") for v in out_subsets.values()
            if v and "slices" in v and "corpus" in v["slices"] and "nonce" in v["slices"]]
    return {
        "overall": mean_of(list(Track)),
        "recognition_avg": mean_of(subset_registry.RECOGNITION_TRACKS),
        "tracks": tracks,
        "subsets": out_subsets,
        "prior_reliance": {
            "corpus_text": round(float(np.mean(real)), 2) if real else None,
            "nonce_words": round(float(np.mean(nonce)), 2) if nonce else None,
        },
    }
