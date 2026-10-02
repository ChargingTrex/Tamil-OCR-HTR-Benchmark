"""Turn a manifest + a model's predictions into scores.

Scores are reported per subset (with a 95% bootstrap confidence interval), per script
stage, per era, per track (the unweighted mean of a track's subsets), and as two headline
numbers that give **the modern script and the older scripts equal weight**:

* **OCR/HTR** = ½ · modern-script reading + ½ · older-script reading. The older half is the
  mean of the older script stages present (pre-reform Tamil, Grantha–Tamil, Grantha,
  Tamil-Brahmi), each counting the same. A script stage's reading score is the mean, over
  the reading tracks it appears in, of the mean of its subset slices in that track.
* **Overall** = ½ · modern composite + ½ · older composite. A composite is ¾ reading and ¼
  the other tasks: for the modern script ⅛ identification + ⅛ translation; for the older
  scripts ¼ identification (there is no translation task for them). Identification is
  split by era: script identification by the era of each script class, medium
  identification by the era of the script on each item.

A model that does not support a task (e.g. a classical OCR engine asked to translate)
gets ``None`` for that subset — never a zero — and therefore no Overall; its OCR/HTR score
is still reported. Missing predictions for supported tasks count as empty answers, and so
does an explicit "no text" answer (``prompts.NO_TEXT``).

Confidence intervals use a *cluster* bootstrap (Miller 2024, "Adding Error Bars to
Evals"): items rendered from the same source text are correlated, so the bootstrap
resamples source texts rather than items. Each subset is resampled with its own seed,
derived from the run seed and the subset id; the draws are identical for every model,
which makes model-to-model differences paired (see ``compare.py``). Intervals on script,
era and headline scores combine the subsets' bootstrap replicates with the same formulas
as the point estimates.

Control subsets (track ``diagnostics``) are scored and reported but never enter a track,
an era or a headline score.
"""

from __future__ import annotations

import zlib
from collections import defaultdict

import numpy as np

from . import subsets as S
from .metrics import classification as cls_m
from .metrics import diagnostics as diag
from .metrics import recognition as rec_m
from .metrics import translation as tr_m
from .metrics import unit_tests as ut
from .metrics.edit import levenshtein
from .taxonomy import ERAS, READING_TRACKS, SCORED_TRACKS, Era, Task, Track, era_of
from .text.tamil import letters

N_BOOT = 1000
CI_METHOD = "cluster bootstrap over source texts, 95% percentile interval"
AGGREGATION = ("script-balanced: OCR/HTR = ½ modern-script reading + ½ older-script reading (each older "
               "script stage weighted equally); Overall = ½ modern composite + ½ older composite")

# Weights inside each era's composite score (each era is half of Overall).
COMPOSITE_WEIGHTS: dict[Era, dict[str, float]] = {
    Era.MODERN: {"reading": 0.75, "identification": 0.125, "translation": 0.125},
    Era.OLDER: {"reading": 0.75, "identification": 0.25},
}

# Text pools whose items are reused across images; items drawn from the same pool entry
# form one cluster. Everything else (nonce lines, random-word lines, numerals, screens,
# identification items, blank controls) is treated as independent.
_CLUSTER_POOLS = {"modern", "classical", "historical", "signage", "plaques", "names", "ui",
                  "sanskrit", "manipravalam", "cict", "tirukkural"}


def subset_seed(seed: int, subset_id: str) -> int:
    """The resampling seed of one subset: independent draws per subset, identical per model."""
    return (seed * 1_000_003 + zlib.crc32(subset_id.encode("utf-8"))) % (2 ** 32)


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


def _ci(values) -> list[float] | None:
    if values is None:
        return None
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [round(float(lo), 2), round(float(hi), 2)]


def paired_summary(diff_obs: float, reps: np.ndarray) -> dict:
    """Difference with a 95% percentile interval and a two-sided bootstrap p-value."""
    lo, hi = np.percentile(reps, [2.5, 97.5])
    p = min(1.0, 2 * min(float((reps <= 0).mean()), float((reps >= 0).mean())))
    if np.all(reps == 0):
        p = 1.0
    return {"diff": round(float(diff_obs), 2), "ci95": [round(float(lo), 2), round(float(hi), 2)],
            "p": round(p, 4), "separable": bool(lo > 0 or hi < 0)}


def _letters(text: str) -> list[str]:
    return [u for u in letters(text) if not u.isspace()]


def _rec_boot(w: np.ndarray, e: np.ndarray, c: np.ndarray) -> np.ndarray:
    return 100.0 * np.clip(1.0 - (w @ e) / np.maximum(1, w @ c), 0, None)


def _rec_value(e: np.ndarray, c: np.ndarray) -> float:
    return 100.0 * max(0.0, 1.0 - float(e.sum()) / max(1.0, float(c.sum())))


def _script(row: dict) -> str:
    """The script stage of a manifest row (rows built by hand may omit it: the subset's first)."""
    return row.get("script") or S.get(row["subset"]).scripts[0].value


def _group(keys: list) -> dict:
    out: dict = defaultdict(list)
    for i, k in enumerate(keys):
        if k is not None:
            out[k].append(i)
    return {k: np.array(v, dtype=int) for k, v in out.items()}


# ----------------------------------------------------------------------------------- reading

def score_recognition(spec, rows, preds, n_boot, seed, *, return_boot=False) -> dict:
    pres = [S.presentation(r) for r in rows]
    pols = [rec_m.TextPolicy.from_dict(p.policy) for p in pres]
    raw_refs = [r.get(p.target) or "" for r, p in zip(rows, pres)]
    answers = [rec_m.answer_text(p) for p in preds]
    refs = [pol.apply(t) for pol, t in zip(pols, raw_refs)]
    hyps = [pol.apply(a) for pol, a in zip(pols, answers)]
    stats = [rec_m.sample_stats(t, a, pol) for t, a, pol in zip(raw_refs, answers, pols)]
    res = rec_m.aggregate(stats)
    e = np.array([s["char_edits"] for s in stats], dtype=float)
    c = np.array([s["chars"] for s in stats], dtype=float)
    w = boot_weights(rows, n_boot, subset_seed(seed, spec.id))
    boot = _rec_boot(w, e, c)
    res["ci95"] = _ci(boot)

    # Script stages inside the subset (stone and copper plates mix modern and older script).
    res["by_script"], boot_by_script = {}, {}
    for sc, idx in sorted(_group([_script(r) for r in rows]).items()):
        bs = _rec_boot(w[:, idx], e[idx], c[idx])
        res["by_script"][sc] = {"n": len(idx), "score": round(_rec_value(e[idx], c[idx]), 4), "ci95": _ci(bs)}
        boot_by_script[sc] = bs

    # Prior-reliance slices: real text vs random words vs phonotactic nonce words, also by era.
    eras = [era_of(_script(r)) for r in rows]
    lex = [r.get("lexical", "corpus") for r in rows]
    res["slices"] = {k: round(_rec_value(e[ix], c[ix]), 2) for k, ix in _group(lex).items()}
    res["slices_by_era"] = {}
    for era, ix in _group(eras).items():
        sub = _group([lex[i] for i in ix])
        res["slices_by_era"][era.value] = {k: round(_rec_value(e[ix[j]], c[ix[j]]), 2) for k, j in sub.items()}

    # Failure modes, order-free error, sample-level NED and letter confusions.
    allowed = set().union(*(diag.scripts_in(t) for t in refs)) if refs else set()
    items = [diag.item_diagnostics(rt, ht, p, allowed) for rt, ht, p in zip(refs, hyps, preds)]
    res["diagnostics"] = diag.aggregate_diagnostics(items)
    res["confusions"] = diag.letter_confusions([(_letters(rt), _letters(ht)) for rt, ht in zip(refs, hyps)])
    res["_items"] = items
    res["_item_eras"] = [x.value if x else None for x in eras]

    # olmOCR-style line tests for multi-line references.
    unit = [ut.run_tests(ut.make_tests([pol.apply(ln) for ln in t.split("\n")]), h)
            for t, h, pol in zip(raw_refs, hyps, pols)]
    res["unit_tests"] = ut.summarize(unit)
    res["_unit"] = unit

    # Reading vs reciting: how far the answer moved from the text in the image towards the
    # canonical text (the standard edition of a real manuscript, or the unperturbed original
    # of a perturbed famous text): −1 = reads the image, +1 = recites the canonical text.
    canon = {i: pols[i].apply(r["text_canonical"]) for i, r in enumerate(rows) if r.get("text_canonical")}
    if canon:
        def pull(ix):
            num = den = 0
            for i in ix:
                d, k, h = refs[i], canon[i], hyps[i]
                num += levenshtein(h, d) - levenshtein(h, k)
                den += levenshtein(d, k)
            return (round(num / den, 4) if den else None), den

        res["recitation_index"], den = pull(list(canon))
        res["scribe_vs_edition_cer"] = round(den / max(1, sum(len(refs[i]) for i in canon)), 4)
        res["recitation_by_script"] = {sc: pull([i for i in ix if i in canon])[0]
                                       for sc, ix in _group([_script(r) for r in rows]).items()
                                       if any(i in canon for i in ix)}
        res["recitation_by_era"] = {era.value: pull([i for i in ix if i in canon])[0]
                                    for era, ix in _group(eras).items() if any(i in canon for i in ix)}
    res["per_sample"] = [{"id": r["id"], "cer": round(s["char_edits"] / max(1, s["chars"]), 4)}
                         for r, s in zip(rows, stats)]
    if return_boot:
        res["_boot"] = boot
        res["_boot_by_script"] = boot_by_script
    return res


def score_abstention(spec, rows, preds, n_boot, seed, *, return_boot=False) -> dict:
    """Blank and effaced controls: the right answer is no text, so the score is the share of
    items answered without inventing letters (100 − hallucination rate)."""
    answers = [rec_m.answer_text(p) for p in preds]
    n_letters = [rec_m.letter_count(a) for a in answers]
    h = np.array([n > 0 for n in n_letters], dtype=float)
    w = boot_weights(rows, n_boot, subset_seed(seed, spec.id))
    boot = 100.0 * (1.0 - (w @ h) / np.maximum(1, w.sum(1)))
    rate = float(h.mean()) if len(h) else 0.0

    def rates(keys):
        return {k: {"n": len(ix), "hallucination_rate": round(float(h[ix].mean()), 4)}
                for k, ix in _group(keys).items()}

    res = {"n": len(rows), "score": 100.0 * (1.0 - rate), "hallucination_rate": round(rate, 4),
           "ci95": _ci(boot),
           "mean_letters_when_hallucinating": round(float(np.mean([n for n in n_letters if n])), 1)
           if any(n_letters) else 0.0,
           "abstained": sum(1 for p in preds if rec_m.is_abstention(p)),
           "by_era": {(k.value if k else "none"): v
                      for k, v in rates([era_of(_script(r)) for r in rows]).items()},
           "by_control": rates([r.get("control") for r in rows]),
           "by_medium": rates([r["medium"] for r in rows]),
           "per_sample": [{"id": r["id"], "letters": n} for r, n in zip(rows, n_letters)]}
    if return_boot:
        res["_boot"] = boot
        res["_boot_by_script"] = {}
    return res


# ------------------------------------------------------------------------------ identification

def _weighted_macro_f1(w: np.ndarray, gold: np.ndarray, pred: np.ndarray, labels: list[str], *,
                       support_only: bool = False) -> np.ndarray:
    """Macro-F1 for every bootstrap replicate, with the semantics of ``classification.scores``;
    ``support_only`` averages only over classes with gold items in the replicate."""
    num = np.zeros(w.shape[0])
    den = np.zeros(w.shape[0])
    for lab in labels:
        g, p = gold == lab, pred == lab
        tp, fp, fn = w @ (g & p), w @ (~g & p), w @ (g & ~p)
        prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
        present = (tp + fn) > 0 if support_only else (tp + fp + fn) > 0
        num += np.where(present, f1, 0.0)
        den += present
    return 100.0 * num / np.maximum(1, den)


def score_classification(spec, rows, preds, n_boot, seed, *, return_boot=False) -> dict:
    labels = list(spec.labels)
    gold = [r["label"] for r in rows]
    pred = [cls_m.parse_label(p or "", labels) for p in preds]
    res = cls_m.scores(gold, pred, labels)
    g, p = np.array(gold), np.array(pred)
    w = boot_weights(rows, n_boot, subset_seed(seed, spec.id))
    boot = _weighted_macro_f1(w, g, p, labels)
    res["ci95"] = _ci(boot)
    conf: dict = defaultdict(lambda: defaultdict(int))
    for a, b in zip(gold, pred):
        conf[a][b] += 1
    res["confusion"] = {a: dict(b) for a, b in conf.items()}
    res["per_class"] = {}
    for lab in labels:
        tp = int(((g == lab) & (p == lab)).sum())
        fp = int(((g != lab) & (p == lab)).sum())
        fn = int(((g == lab) & (p != lab)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        res["per_class"][lab] = {"n": tp + fn, "precision": round(prec, 4), "recall": round(rec, 4),
                                 "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0}
    # Accuracy on the items of each script stage (for medium identification: the script on
    # the item; for script identification: the correct label).
    res["by_script"] = {sc: {"n": len(ix), "accuracy": round(float((g[ix] == p[ix]).mean()), 4)}
                        for sc, ix in _group([_script(r) for r in rows]).items()}
    # By era. Script ID: mean F1 over the era's script classes, precision counted over every
    # item (so calling Malayalam "pre-reform Tamil" costs the older half). Medium ID: mean F1
    # over the media present among the era's items, counted on those items only.
    res["by_era"], boot_by_era = {}, {}
    for era in Era:
        if spec.task == Task.SCRIPT_ID:
            idx = np.arange(len(rows))
            classes = [lab for lab in labels if era_of(lab) == era]
        else:
            idx = np.array([i for i, r in enumerate(rows) if era_of(_script(r)) == era], dtype=int)
            classes = sorted({gold[i] for i in idx})
        if not len(idx) or not classes:
            continue
        obs = _weighted_macro_f1(np.ones((1, len(idx))), g[idx], p[idx], classes, support_only=True)[0]
        bs = _weighted_macro_f1(w[:, idx], g[idx], p[idx], classes, support_only=True)
        res["by_era"][era.value] = {"n": int(sum(1 for i in idx if gold[i] in classes)), "score": round(float(obs), 4),
                                    "ci95": _ci(bs), "classes": classes}
        boot_by_era[era.value] = bs
    res["per_sample"] = [{"id": r["id"], "pred": q, "ok": q == r["label"]} for r, q in zip(rows, pred)]
    if return_boot:
        res["_boot"] = boot
        res["_boot_by_era"] = boot_by_era
    return res


# --------------------------------------------------------------------------------- translation

def score_translation(spec, rows, preds, n_boot, seed, *, return_boot=False) -> dict:
    refs = [r[spec.target] for r in rows]
    hyps = [p or "" for p in preds]
    st = np.array([tr_m.sentence_stats(h, r) for h, r in zip(hyps, refs)], dtype=float)
    chrf = tr_m.f_score(list(st.sum(0)))
    w = boot_weights(rows, n_boot, subset_seed(seed, spec.id))
    boot = np.array([tr_m.f_score(list(s)) for s in w @ st])
    per = [round(tr_m.f_score(list(s)), 2) for s in st]
    res = {"n": len(refs), "chrf_pp": chrf, "score": chrf, "ci95": _ci(boot), "by_era": {},
           "per_sample": [{"id": r["id"], "chrf": v} for r, v in zip(rows, per)]}
    boot_by_era = {}
    for era, ix in _group([era_of(_script(r)) for r in rows]).items():
        bs = np.array([tr_m.f_score(list(s)) for s in w[:, ix] @ st[ix]])
        res["by_era"][era.value] = {"n": len(ix), "score": round(tr_m.f_score(list(st[ix].sum(0))), 4), "ci95": _ci(bs)}
        boot_by_era[era.value] = bs
    if return_boot:
        res["_boot"] = boot
        res["_boot_by_era"] = boot_by_era
    return res


def _presented_target(spec, row: dict) -> str:
    return row.get(S.presentation(row).target) or ""


def item_result(spec, row: dict, text: str | None) -> dict:
    """Official per-item score, computed with the same code as the subset score."""
    if spec.scoring == "abstention":
        n = rec_m.letter_count(rec_m.answer_text(text))
        return {"text": text, "hallucinated": n > 0, "letters": n}
    if spec.task == Task.RECOGNITION:
        pspec = S.presentation(row)
        st = rec_m.sample_stats(_presented_target(spec, row), rec_m.answer_text(text),
                                rec_m.TextPolicy.from_dict(pspec.policy))
        return {"text": text, "cer": round(st["char_edits"] / max(1, st["chars"]), 3),
                "char_edits": st["char_edits"], "chars": st["chars"]}
    ref = row.get(spec.target) or ""
    if spec.task in (Task.SCRIPT_ID, Task.MEDIUM_ID):
        lab = cls_m.parse_label(text or "", list(spec.labels))
        return {"text": text, "label": lab, "ok": lab == ref}
    return {"text": text, "chrf": round(tr_m.f_score(tr_m.sentence_stats(text or "", ref)), 1)}


SCORERS = {Task.RECOGNITION: score_recognition, Task.SCRIPT_ID: score_classification,
           Task.MEDIUM_ID: score_classification, Task.TRANSLATION: score_translation}


def scorer_for(spec):
    return score_abstention if spec.scoring == "abstention" else SCORERS[spec.task]


# --------------------------------------------------------------------------------- aggregation

def _mean(values):
    return sum(values) / len(values)


def aggregation_input(res: dict | None, boot: bool = False) -> dict | None:
    """The part of a subset result that ``aggregate`` needs: point values, or (``boot``)
    arrays of bootstrap replicates."""
    if res is None:
        return None
    if boot:
        return {"score": res["_boot"], "by_script": dict(res.get("_boot_by_script") or {}),
                "by_era": dict(res.get("_boot_by_era") or {})}
    return {"score": res["score"],
            "by_script": {k: v["score"] for k, v in (res.get("by_script") or {}).items() if "score" in v},
            "by_era": {k: v["score"] for k, v in (res.get("by_era") or {}).items() if "score" in v}}


def aggregate(subs: dict[str, dict | None], layout: dict[str, list[str]]) -> dict:
    """Combine subset values into tracks, script stages, eras and the headline scores.

    ``layout`` maps every subset in the manifest to the script stages of its items;
    ``subs[sid]`` is ``aggregation_input(...)`` — ``None`` where the system does not do the
    task. Values may be floats or arrays of bootstrap replicates; the formulas are the same."""
    tracks = {}
    for t in SCORED_TRACKS:
        ids = [s for s in S.TRACK_SUBSETS[t] if s in layout]
        vals = [subs[s]["score"] for s in ids if subs.get(s) is not None]
        tracks[t.value] = _mean(vals) if ids and len(vals) == len(ids) else None

    cells: dict[str, dict[str, list]] = {}
    slices: dict[str, dict[str, object]] = {}
    for t in READING_TRACKS:
        for sid in S.TRACK_SUBSETS[t]:
            for sc in layout.get(sid, ()):
                v = subs.get(sid)
                x = None if v is None else v["by_script"].get(sc)
                cells.setdefault(sc, {}).setdefault(t.value, []).append(x)
                slices.setdefault(sc, {})[sid] = x
    scripts = {}
    for sc, by_track in cells.items():
        tv = {t: (None if any(x is None for x in xs) else _mean(xs)) for t, xs in by_track.items()}
        score = None if any(x is None for x in tv.values()) else _mean(list(tv.values()))
        scripts[sc] = {"score": score, "tracks": tv, "subsets": slices[sc]}

    eras = {}
    for era in Era:
        members = [sc for sc in scripts if era_of(sc) == era]
        if not members:
            continue
        vals = [scripts[sc]["score"] for sc in members]
        parts = {"reading": None if any(v is None for v in vals) else _mean(vals)}
        for comp, track in (("identification", Track.CLASSIFICATION), ("translation", Track.TRANSLATION)):
            if comp not in COMPOSITE_WEIGHTS[era]:
                continue
            got, missing = [], False
            for sid in S.TRACK_SUBSETS[track]:
                if not any(era_of(sc) == era for sc in layout.get(sid, ())):
                    continue
                v = subs.get(sid)
                if v is None or era.value not in v["by_era"]:
                    missing = True
                else:
                    got.append(v["by_era"][era.value])
            if missing or got:
                parts[comp] = None if missing else _mean(got)
        weights = COMPOSITE_WEIGHTS[era]
        complete = set(parts) == set(weights) and all(parts[k] is not None for k in weights)
        composite = sum(weights[k] * parts[k] for k in weights) if complete else None
        eras[era.value] = {"scripts": members, **parts, "composite": composite}

    both = len(eras) == len(Era)
    reading = [eras[e]["reading"] for e in eras]
    composites = [eras[e]["composite"] for e in eras]
    return {
        "tracks": tracks, "scripts": scripts, "eras": eras,
        "recognition_avg": _mean(reading) if both and all(v is not None for v in reading) else None,
        "overall": _mean(composites) if both and all(v is not None for v in composites) else None,
    }


def _round(v, nd=2):
    return None if v is None else round(float(v), nd)


def _failure_modes(items: list[dict]) -> dict | None:
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
    layout = {sid: sorted({_script(r) for r in rows}) for sid, rows in by_subset.items()}

    out_subsets: dict[str, dict | None] = {}
    obs, reps = {}, {}
    pooled_items: dict[str | None, list[dict]] = defaultdict(list)
    pooled_unit: dict[str | None, list[dict]] = defaultdict(list)
    for sid, rows in by_subset.items():
        spec = S.get(sid)
        if supported_tasks is not None and spec.task.value not in supported_tasks:
            out_subsets[sid] = obs[sid] = reps[sid] = None
            continue
        preds = [predictions.get(r["id"]) for r in rows]
        res = scorer_for(spec)(spec, rows, preds, n_boot, seed, return_boot=True)
        res["missing"] = sum(p is None for p in preds)
        obs[sid], reps[sid] = aggregation_input(res), aggregation_input(res, boot=True)
        items, item_eras, unit = res.pop("_items", None), res.pop("_item_eras", None), res.pop("_unit", None)
        if spec.ranked and items is not None:
            for d, era, u in zip(items, item_eras, unit):
                for key in (None, era):
                    pooled_items[key].append(d)
                    pooled_unit[key].append(u)
        if not return_boot:
            for k in [k for k in res if k.startswith("_boot")]:
                res.pop(k)
        if not keep_per_sample:
            res.pop("per_sample", None)
        out_subsets[sid] = res

    agg, aggb = aggregate(obs, layout), aggregate(reps, layout)

    scripts = {}
    for sc, v in agg["scripts"].items():
        vb = aggb["scripts"][sc]
        entry = {"era": (era_of(sc) or Era.MODERN).value, "score": _round(v["score"]),
                 "ci95": _ci(vb["score"]) if v["score"] is not None else None,
                 "n": sum((out_subsets[s] or {}).get("by_script", {}).get(sc, {}).get("n", 0) for s in v["subsets"]),
                 "tracks": {t: _round(x) for t, x in v["tracks"].items()},
                 "subsets": {s: (out_subsets[s] or {}).get("by_script", {}).get(sc) for s in v["subsets"]}}
        ident = {}
        for sid in S.TRACK_SUBSETS[Track.CLASSIFICATION]:
            r = out_subsets.get(sid)
            if not r:
                continue
            if sid == "script-id" and sc in r.get("per_class", {}):
                ident["script_id_f1"] = round(100 * r["per_class"][sc]["f1"], 2)
                ident["script_id_recall"] = round(100 * r["per_class"][sc]["recall"], 2)
            elif sid != "script-id" and sc in r.get("by_script", {}):
                ident["medium_id_accuracy"] = round(100 * r["by_script"][sc]["accuracy"], 2)
        entry["identification"] = ident or None
        scripts[sc] = entry

    eras = {}
    for e, v in agg["eras"].items():
        vb = aggb["eras"][e]
        eras[e] = {"name": ERAS[Era(e)].name, "scripts": v["scripts"]}
        for k in ("reading", "identification", "translation", "composite"):
            if k in v:
                eras[e][k] = _round(v[k])
                eras[e][k + "_ci95"] = _ci(vb[k]) if v[k] is not None else None
    era_gap = None
    m, o = agg["eras"].get("modern", {}), agg["eras"].get("older", {})
    if m.get("reading") is not None and o.get("reading") is not None:
        era_gap = paired_summary(m["reading"] - o["reading"],
                                 aggb["eras"]["modern"]["reading"] - aggb["eras"]["older"]["reading"])

    def pooled(slice_key, era=None):
        vals = []
        for sid, v in out_subsets.items():
            if not v or not S.get(sid).ranked or "slices" not in v:
                continue
            sl = v["slices"] if era is None else v.get("slices_by_era", {}).get(era, {})
            if "corpus" in sl and "nonce" in sl:
                vals.append(sl[slice_key])
        return round(float(np.mean(vals)), 2) if vals else None

    rec_idx = {sid: v["recitation_index"] for sid, v in out_subsets.items()
               if v and v.get("recitation_index") is not None}
    pert = out_subsets.get("perturbed")
    prior_pull = None
    if pert and pert.get("recitation_index") is not None:
        prior_pull = {"index": pert["recitation_index"], "by_era": pert.get("recitation_by_era"),
                      "by_script": pert.get("recitation_by_script"), "n": pert["n"]}
    blank = out_subsets.get("blank-controls")
    hallucination = None
    if blank:
        lo, hi = blank["ci95"]
        hallucination = {"n": blank["n"], "rate": blank["hallucination_rate"],
                         "rate_ci95": [round(1 - hi / 100, 4), round(1 - lo / 100, 4)],
                         **{k: blank[k] for k in ("by_era", "by_control", "by_medium", "abstained",
                                                  "mean_letters_when_hallucinating")}}
    return {
        "overall": _round(agg["overall"]),
        "overall_ci95": _ci(aggb["overall"]) if agg["overall"] is not None else None,
        "recognition_avg": _round(agg["recognition_avg"]),
        "recognition_avg_ci95": _ci(aggb["recognition_avg"]) if agg["recognition_avg"] is not None else None,
        "eras": eras,
        "era_gap": era_gap,
        "scripts": scripts,
        "tracks": {t: _round(v) for t, v in agg["tracks"].items()},
        "subsets": out_subsets,
        "prior_reliance": {
            "corpus_text": pooled("corpus"), "nonce_words": pooled("nonce"),
            "by_era": {e.value: {"corpus_text": pooled("corpus", e.value), "nonce_words": pooled("nonce", e.value)}
                       for e in Era},
            "recitation_index": rec_idx or None,
            "prior_pull": prior_pull,
        },
        "hallucination": hallucination,
        "failure_modes": _failure_modes(pooled_items.get(None, [])),
        "failure_modes_by_era": {e.value: _failure_modes(pooled_items.get(e.value, [])) for e in Era},
        "unit_tests": {"all": ut.summarize(pooled_unit.get(None, [])),
                       **{e.value: ut.summarize(pooled_unit.get(e.value, [])) for e in Era}},
        "ci_method": CI_METHOD,
        "aggregation": AGGREGATION,
    }
