"""chrF++ (Popović 2017), implemented to match sacrebleu's CHRF(word_order=2) exactly:
character n-grams 1–6 (whitespace removed) plus word n-grams 1–2, β = 2, statistics
summed over the corpus before the F-score is computed.
"""

from __future__ import annotations

from collections import Counter

CHAR_ORDER, WORD_ORDER, BETA = 6, 2, 2
_PUNCTS = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')


def _char_ngrams(text: str, n: int) -> Counter:
    s = "".join(text.split())
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def _words(text: str) -> list[str]:
    out = []
    for w in text.split():
        if len(w) == 1:
            out.append(w)
        elif w[-1] in _PUNCTS:
            out += [w[:-1], w[-1]]
        elif w[0] in _PUNCTS:
            out += [w[0], w[1:]]
        else:
            out.append(w)
    return out


def _word_ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def sentence_stats(hyp: str, ref: str) -> list[int]:
    """[hyp, ref, match] counts for each of the 8 n-gram orders, concatenated."""
    stats: list[int] = []
    for n in range(1, CHAR_ORDER + 1):
        h, r = _char_ngrams(hyp, n), _char_ngrams(ref, n)
        stats += [sum(h.values()), sum(r.values()), sum((h & r).values())]
    hw, rw = _words(hyp), _words(ref)
    for n in range(1, WORD_ORDER + 1):
        h, r = _word_ngrams(hw, n), _word_ngrams(rw, n)
        stats += [sum(h.values()), sum(r.values()), sum((h & r).values())]
    return stats


def f_score(stats: list[int]) -> float:
    eps, factor = 1e-16, BETA ** 2
    order = CHAR_ORDER + WORD_ORDER
    eff, avg_p, avg_r = 0, 0.0, 0.0
    for i in range(order):
        n_hyp, n_ref, n_match = stats[3 * i: 3 * i + 3]
        p = n_match / n_hyp if n_hyp > 0 else eps
        r = n_match / n_ref if n_ref > 0 else eps
        eff += n_hyp > 0 and n_ref > 0
        avg_p += p
        avg_r += r
    if eff == 0:
        return 0.0
    avg_p /= eff
    avg_r /= eff
    if avg_p + avg_r == 0:
        return 0.0
    return 100.0 * (1 + factor) * avg_p * avg_r / (factor * avg_p + avg_r)


def corpus_chrf(hyps: list[str], refs: list[str]) -> float:
    total = [0] * (3 * (CHAR_ORDER + WORD_ORDER))
    for h, r in zip(hyps, refs):
        for i, v in enumerate(sentence_stats(h or "", r)):
            total[i] += v
    return f_score(total)
