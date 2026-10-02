"""Turn a manifest + a model's predictions into scores.

Scores are reported per subset (with a 95% bootstrap confidence interval), per track
(unweighted mean of the track's subsets) and overall (unweighted mean of the eight tracks).
A model that does not support a task (e.g. a classical OCR engine asked to translate) gets
``None`` for that subset — never a zero — and therefore no Overall score; its Recognition
average is still reported. Missing predictions for supported tasks count as empty answers.

Confidence intervals use a *cluster* bootstrap (Miller 2024, "Adding Error Bars to
Evals"): items rendered from the same source text are correlated, so the bootstrap
resamples source texts rather than items. Resampling is seeded and identical for every
model, which makes model-to-model differences paired (see ``compare.py``).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import subsets as subset_registry
from .metrics import classification as cls_m
from .metrics import diagnostics as diag
from .metrics import recognition as rec_m
from .metrics import translation as tr_m
from .taxonomy import Task, Track
from .text.tamil import letters

N_BOOT = 1000
CI_METHOD = "cluster bootstrap over source texts, 95% percentile interval"

# Text pools whose items are reused across images; items drawn from the same pool entry
# form one cluster. Everything else (nonce lines, random-word lines, numerals, screens,
# identification items) is treated as independent.
_CLUSTER_POOLS = {"modern", "classical", "historical", "signage", "plaques", "names", "ui",
                  "sanskrit", "manipravalam", "cict"}


def cluster_ids(rows: list[dict]) -> np.ndarray:
    keys = []
    for r in rows:
        first = (r.get("text_source") or "").split(",")[0]
        pool = first.split(":", 1)[0]
        keys.append(first if ":" in first and pool in _CLUSTER_POOLS else "item:" + r["id"])
    index: dict[str, int] = {}
    return np.array([index.setdefault(k, len(index)) for k in keys], dtype=int)


def boot_weights(rows: list[dict], n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_items) multiplicities from resampling clusters with replacement."""
    cids = cluster_ids(rows)
    if len(cids) == 0:
        return np.zeros((n_boot, 0))
    k = int(cids.max()) + 1
    counts = np.random.default_rng(seed).multinomial(k, np.full(k, 1.0 / k), size=n_boot)
    return counts[:, cids].astype(float)


def _ci(values: np.ndarray) -> list[float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [round(float(lo), 2), round(float(hi), 2)]


def _letters(text: str) -> list[str]:
    return [u for u in letters(text) if not u.isspace()]


def score_recognition(spec, rows, preds, n_boot, seed, *, return_boot=False) -> dict:
    policy = rec_m.TextPolicy.from_dict(spec.policy)
    refs = [policy.apply(r[spec.target]) for r in rows]
    hyps = [policy.apply(p or "") for p in preds]
    stats = [rec_m.sample_stats(r[spec.target], p or "", policy) for r, p in zip(rows, preds)]
    res = rec_m.aggregate(stats)
    e = np.array([s["char_edits"] for s in stats], dtype=float)
    c = np.array([s["chars"] for s in stats], dtype=float)
    w = boot_weights(rows, n_boot, seed)
    boot = 100.0 * np.clip(1.0 - (w @ e) / np.maximum(1, w @ c), 0, None)
    res["ci95"] = _ci(boot)
    # Prior-reliance slice: real text vs phonotactic nonce words.
    slices = defaultdict(list)
    for r, s in zip(rows, stats):
        slices[r.get("lexical", "corpus")].append(s)
    res["slices"] = {k: round(rec_m.score_value(v), 2) for k, v in slices.items()}
    # Failure modes, order-free error, sample-level NED and letter confusions.
    allowed = set().union(*(diag.scripts_in(t) for t in refs)) if refs else set()
    items = [diag.item_diagnostics(rt, ht, p, allowed) for rt, ht, p in zip(refs, hyps, preds)]
    res["diagnostics"] = diag.aggregate_diagnostics(items)
    res["confusions"] = diag.letter_confusions([(_letters(rt), _letters(ht)) for rt, ht in zip(refs, hyps)])
    res["_items"] = items
    # Reading vs reciting on real manuscripts: how far the answer moved from the scribe's
    # text towards the standard edition (-1 = reads the leaf, +1 = recites the edition).
    canon = [(rt, policy.apply(r["text_canonical"]), ht) for r, rt, ht in zip(rows, refs, hyps)
             if r.get("text_canonical")]
    if canon:
        from .metrics.edit import levenshtein
        num = sum(levenshtein(h, d) - levenshtein(h, k) for d, k, h in canon)
        den = sum(levenshtein(d, k) for d, k, _ in canon)
        res["recitation_index"] = round(num / den, 4) if den else None
        res["scribe_vs_edition_cer"] = round(den / max(1, sum(len(d) for d, _, _ in canon)), 4)
    res["per_sample"] = [{"id": r["id"], "cer": round(s["char_edits"] / max(1, s["chars"]), 4)}
                         for r, s in zip(rows, stats)]
    if return_boot:
        res["_boot"] = boot
    return res


def _weighted_macro_f1(w: np.ndarray, gold: np.ndarray, pred: np.ndarray, labels: list[str]) -> np.ndarray:
    """Macro-F1 for every bootstrap replicate, with the semantics of ``classification.scores``."""
    num = np.zeros(w.shape[0])
    den = np.zeros(w.shape[0])
    for lab in labels:
        g, p = gold == lab, pred == lab
        tp, fp, fn = w @ (g & p), w @ (~g & p), w @ (g & ~p)
        prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
        present = (tp + fp + fn) > 0
        num += np.where(present, f1, 0.0)
        den += present
    return 100.0 * num / np.maximum(1, den)


def score_classification(spec, rows, preds, n_boot, seed, *, return_boot=False) -> dict:
    labels = list(spec.labels)
    gold = [r["label"] for r in rows]
    pred = [cls_m.parse_label(p or "", labels) for p in preds]
    res = cls_m.scores(gold, pred, labels)
    boot = _weighted_macro_f1(boot_weights(rows, n_boot, seed), np.array(gold), np.array(pred), labels)
    res["ci95"] = _ci(boot)
    conf = defaultdict(lambda: defaultdict(int))
    for a, b in zip(gold, pred):
        conf[a][b] += 1
    res["confusion"] = {a: dict(b) for a, b in conf.items()}
    res["per_sample"] = [{"id": r["id"], "pred": q, "ok": q == r["label"]} for r, q in zip(rows, pred)]
    if return_boot:
        res["_boot"] = boot
    return res


def score_translation(spec, rows, preds, n_boot, seed, *, return_boot=False) -> dict:
    refs = [r[spec.target] for r in rows]
    hyps = [p or "" for p in preds]
    st = np.array([tr_m.sentence_stats(h, r) for h, r in zip(hyps, refs)], dtype=float)
    chrf = tr_m.f_score(list(st.sum(0)))
    sums = boot_weights(rows, n_boot, seed) @ st
    boot = np.array([tr_m.f_score(list(s)) for s in sums])
    per = [round(tr_m.f_score(list(s)), 2) for s in st]
    res = {"n": len(refs), "chrf_pp": chrf, "score": chrf, "ci95": _ci(boot),
           "per_sample": [{"id": r["id"], "chrf": v} for r, v in zip(rows, per)]}
    if return_boot:
        res["_boot"] = boot
    return res


SCORERS = {Task.RECOGNITION: score_recognition, Task.SCRIPT_ID: score_classification,
           Task.MEDIUM_ID: score_classification, Task.TRANSLATION: score_translation}


def _failure_modes(item_lists: list[list[dict]]) -> dict | None:
    items = [d for lst in item_lists for d in lst]
    if not items:
        return None
    agg = diag.aggregate_diagnostics(items)
    keys = ("empty_rate", "overlong_rate", "repetition_rate", "markup_rate", "offscript_item_rate",
            "offscript_letter_rate")
    return {"n": len(items), **{k: agg[k] for k in keys}}


def score_run(manifest: list[dict], predictions: dict[str, str | None], *,
              supported_tasks: set[str] | None = None, n_boot: int = N_BOOT, seed: int = 0,
              keep_per_sample: bool = True, return_boot: bool = False) -> dict:
    by_subset: dict[str, list[dict]] = defaultdict(list)
    for row in manifest:
        by_subset[row["subset"]].append(row)

    out_subsets: dict[str, dict | None] = {}
    recognition_items: list[list[dict]] = []
    for sid, rows in by_subset.items():
        spec = subset_registry.get(sid)
        if supported_tasks is not None and spec.task.value not in supported_tasks:
            out_subsets[sid] = None
            continue
        preds = [predictions.get(r["id"]) for r in rows]
        res = SCORERS[spec.task](spec, rows, preds, n_boot, seed, return_boot=return_boot)
        res["missing"] = sum(p is None for p in preds)
        if "_items" in res:
            recognition_items.append(res.pop("_items"))
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
    rec_idx = [(v["recitation_index"], sid) for sid, v in out_subsets.items()
               if v and v.get("recitation_index") is not None]
    return {
        "overall": mean_of(list(Track)),
        "recognition_avg": mean_of(subset_registry.RECOGNITION_TRACKS),
        "tracks": tracks,
        "subsets": out_subsets,
        "prior_reliance": {
            "corpus_text": round(float(np.mean(real)), 2) if real else None,
            "nonce_words": round(float(np.mean(nonce)), 2) if nonce else None,
            "recitation_index": {sid: v for v, sid in rec_idx} or None,
        },
        "failure_modes": _failure_modes(recognition_items),
        "ci_method": CI_METHOD,
    }
