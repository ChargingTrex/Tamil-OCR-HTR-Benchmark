"""Closed-vocabulary classification (script ID, medium ID).

A free-text model answer is mapped onto the label set conservatively: exact match after
normalisation, else the single label (or alias) that occurs in the answer. Anything else
— including answers that name two labels — is ``invalid`` and counts as wrong.
"""

from __future__ import annotations

import re

INVALID = "invalid"

_ALIASES = {
    "tamil-modern": ["modern tamil", "tamil modern", "contemporary tamil", "reformed tamil"],
    "tamil-pre-reform": ["pre-reform tamil", "pre reform tamil", "old tamil script", "pre-1978 tamil",
                         "traditional tamil script", "tamil pre-reform", "tamil pre reform"],
    "tamil-brahmi": ["tamil brahmi", "tamili", "damili", "tamil-brāhmī", "tamil brāhmī"],
    "grantha": ["grantha script", "pallava grantha"],
    "printed-paper": ["printed paper", "print", "printed page", "printed text"],
    "handwritten-paper": ["handwritten paper", "handwriting", "handwritten"],
    "scene-signage": ["scene signage", "signboard", "signage", "scene text"],
    "palm-leaf": ["palm leaf", "palmleaf", "olai", "ola"],
    "copper-plate": ["copper plate", "copperplate", "metal plate", "metal plaque"],
    "estampage": ["ink rubbing", "rubbing"],
}


def _norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[`*_\"'“”‘’.:;!()\[\]{}]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_label(output: str, labels: list[str]) -> str:
    if not output:
        return INVALID
    o = _norm(output)
    norm_labels = {_norm(lab): lab for lab in labels}
    if o in norm_labels:
        return norm_labels[o]
    first = _norm(output.strip().splitlines()[0]) if output.strip() else ""
    if first in norm_labels:
        return norm_labels[first]
    hits = set()
    for lab in labels:
        cands = [_norm(lab), _norm(lab.replace("-", " "))] + [_norm(a) for a in _ALIASES.get(lab, [])]
        for c in cands:
            if re.search(rf"(?<![\w-]){re.escape(c)}(?![\w-])", o):
                hits.add(lab)
                break
    # "tamil-brahmi" also contains "tamil"-only aliases of nothing else; prefer the most specific hit
    if len(hits) > 1:
        specific = {h for h in hits if not any(h != g and h in g for g in hits)}
        hits = specific if len(specific) == 1 else hits
    return hits.pop() if len(hits) == 1 else INVALID


def scores(gold: list[str], pred: list[str], labels: list[str]) -> dict:
    n = len(gold)
    acc = sum(g == p for g, p in zip(gold, pred)) / max(1, n)
    f1s = []
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred))
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    macro_f1 = sum(f1s) / max(1, len(f1s))
    return {"n": n, "accuracy": acc, "macro_f1": macro_f1,
            "invalid_rate": sum(p == INVALID for p in pred) / max(1, n),
            "score": 100.0 * macro_f1}
