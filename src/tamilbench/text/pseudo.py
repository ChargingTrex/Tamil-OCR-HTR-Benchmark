"""Phonotactically plausible Tamil *nonce* words.

A model that reads well but leans on its language model will do much better on real
words than on words it has never seen. Every recognition subset mixes in lines of
pseudo-words, and the leaderboard reports the gap ("prior reliance"). The generator
follows traditional Tamil phonotactics: permitted initial consonants, geminate and
homorganic nasal+stop clusters word-internally, and permitted final consonants.
"""

from __future__ import annotations

import random

from .tamil import INITIAL_CONSONANTS, VIRAMA, VOWEL_SIGNS, VOWELS

_MEDIAL = ["க", "ச", "ட", "த", "ப", "ற", "ண", "ந", "ம", "ன",
           "ய", "ர", "ல", "வ", "ழ", "ள"]  # ங/ஞ occur medially only in clusters
_CLUSTERS = [  # coda + onset, word-internal
    "க்க", "ச்ச", "ட்ட", "த்த", "ப்ப", "ற்ற",          # geminates
    "ங்க", "ஞ்ச", "ண்ட", "ந்த", "ம்ப", "ன்ற",          # homorganic nasal + stop
    "ல்ல", "ள்ள", "ம்ம", "ன்ன", "ண்ண", "ய்ய", "வ்வ",
    "ர்க", "ர்த", "ர்ப", "ய்த", "ழ்க", "ல்க", "ள்க", "ர்ம", "ய்ம",
]
_FINALS = ["ம்", "ன்", "ள்", "ல்", "ர்", "ய்", "ண்", "ழ்"]
_VOWEL_WEIGHTS = [8, 4, 6, 2, 8, 2, 4, 2, 3, 2, 2, 0.2]  # a ā i ī u ū e ē ai o ō au


def _vowel_sign(rng: random.Random) -> str:
    return rng.choices(VOWEL_SIGNS, weights=_VOWEL_WEIGHTS)[0]


def pseudo_word(rng: random.Random, syllables: tuple[int, int] = (2, 4)) -> str:
    n = rng.randint(*syllables)
    if rng.random() < 0.25:
        w = rng.choices(VOWELS, weights=_VOWEL_WEIGHTS)[0]
    else:
        w = rng.choice(INITIAL_CONSONANTS) + _vowel_sign(rng)
    for _ in range(n - 1):
        onset = rng.choice(_CLUSTERS) if rng.random() < 0.35 else rng.choice(_MEDIAL)
        w += onset + _vowel_sign(rng)
    if rng.random() < 0.5:
        w += rng.choice(_FINALS)
    return w.replace(VIRAMA + VIRAMA, VIRAMA)


def pseudo_line(rng: random.Random, n_words: tuple[int, int] = (2, 6),
                avoid: set[str] | None = None) -> str:
    words = []
    target = rng.randint(*n_words)
    while len(words) < target:
        w = pseudo_word(rng)
        if avoid and w in avoid:
            continue
        words.append(w)
    return " ".join(words)
