"""Text-recognition metrics.

Primary: CER — character (Unicode code point) error rate on normalised text, aggregated
over a subset as total edits / total reference characters (micro-average).
Also reported: AER (Tamil-letter / akshara error rate), WER, exact-match rate.

The subset score shown on the leaderboard is ``100 × max(0, 1 − CER)``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..text.grantha import normalize_iast
from ..text.tamil import VIRAMA, letters, merge_long_e_o, normalize
from .edit import levenshtein


@dataclass(frozen=True)
class TextPolicy:
    """How a subset's references and hypotheses are normalised before comparison."""

    spaces: str = "collapse"           # "collapse" | "remove" (scriptio continua)
    script: str = "tamil"              # "tamil" | "iast"
    strip_tamil_numerals: bool = False  # e.g. ignore verse numbers on palm leaves
    fold_pulli: bool = False           # scribes often omit the puḷḷi; editions differ on restoring it
    fold_vowel_length: bool = False    # e/ē and o/ō were not distinguished before the 18th c.

    def apply(self, text: str) -> str:
        t = normalize(text or "", spaces=self.spaces, strip_tamil_numerals=self.strip_tamil_numerals)
        if self.fold_vowel_length:
            t = merge_long_e_o(t)
        if self.fold_pulli:
            t = t.replace(VIRAMA, "")
        if self.script == "iast":
            t = normalize_iast(t)
        return t

    @classmethod
    def from_dict(cls, d: dict | None) -> "TextPolicy":
        return cls(**(d or {}))


def sample_stats(ref: str, hyp: str, policy: TextPolicy = TextPolicy()) -> dict:
    """Per-sample edit statistics (sums are aggregated later)."""
    r, h = policy.apply(ref), policy.apply(hyp)
    rw, hw = r.split(), h.split()
    rl, hl = letters(r), letters(h)
    return {
        "char_edits": levenshtein(r, h), "chars": len(r),
        "letter_edits": levenshtein(rl, hl), "letters": len(rl),
        "word_edits": levenshtein(rw, hw), "words": max(1, len(rw)),
        "exact": int(r == h),
    }


def aggregate(stats: list[dict]) -> dict:
    """Micro-averaged rates over a list of per-sample stats."""
    if not stats:
        return {"n": 0}
    s = {k: sum(x[k] for x in stats) for k in stats[0]}
    cer = s["char_edits"] / max(1, s["chars"])
    return {
        "n": len(stats),
        "cer": cer,
        "aer": s["letter_edits"] / max(1, s["letters"]),
        "wer": s["word_edits"] / max(1, s["words"]),
        "exact_match": s["exact"] / len(stats),
        "score": 100.0 * max(0.0, 1.0 - cer),
    }


def score_value(stats: list[dict]) -> float:
    s_e = sum(x["char_edits"] for x in stats)
    s_n = sum(x["chars"] for x in stats)
    return 100.0 * max(0.0, 1.0 - s_e / max(1, s_n))
