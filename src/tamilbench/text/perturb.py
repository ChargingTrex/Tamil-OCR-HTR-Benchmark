"""Controlled letter substitutions for the perturbed-text controls.

A perturbed item shows a famous text with a few letters changed. Substitutions stay
inside a class of similar letters (Tamil hard, soft and medial consonants; Sanskrit stops,
nasals, semivowels and sibilants), so the changed words still look and sound plausible: a
reader has to look at the letters to get them right, while a system that leans on memory
writes the familiar original instead (cf. Karamolegkou et al. 2026, "Reading or Guessing?").
"""

from __future__ import annotations

import random
import re

TAMIL_CLASSES = ("கசடதபற", "ஙஞணநமன", "யரலவழள")
IAST_CLASSES = ("kgcjṭḍtdpb", "nmṇ", "yrlv", "śṣs")
_TA = {c: g for g in TAMIL_CLASSES for c in g}
_SA = {c: g for g in IAST_CLASSES for c in g}
_SPACE = re.compile(r"(\s+)")


def _swap(rng: random.Random, ch: str, table: dict[str, str]) -> str:
    return rng.choice([c for c in table[ch] if c != ch])


def _perturb(text: str, rng: random.Random, rate: float, positions) -> str:
    tokens = _SPACE.split(text)
    cands = [i for i, t in enumerate(tokens) if positions(t)]
    if not cands:
        return text
    k = max(1, round(rate * len(cands)))
    for i in sorted(rng.sample(cands, min(k, len(cands)))):
        tok = tokens[i]
        j = rng.choice(positions(tok))
        table = _TA if tok[j] in _TA else _SA
        tokens[i] = tok[:j] + _swap(rng, tok[j], table) + tok[j + 1:]
    return "".join(tokens)


def _tamil_positions(tok: str) -> list[int]:
    return [j for j, ch in enumerate(tok) if ch in _TA] if len(tok) >= 3 else []


def _iast_positions(tok: str) -> list[int]:
    # unaspirated consonants only (never the first half of kh, gh, th, …), so the result is valid IAST
    return [j for j, ch in enumerate(tok) if ch in _SA and tok[j + 1:j + 2] != "h"] if len(tok) >= 3 else []


def perturb_tamil(text: str, rng: random.Random, rate: float = 0.3) -> str:
    """Change one consonant in about ``rate`` of the words (at least one word)."""
    return _perturb(text, rng, rate, _tamil_positions)


def perturb_iast(text: str, rng: random.Random, rate: float = 0.3) -> str:
    """The same for Sanskrit in IAST."""
    return _perturb(text, rng, rate, _iast_positions)
