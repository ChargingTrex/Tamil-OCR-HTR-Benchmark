"""Text-recognition metrics.

Primary: CER — character (Unicode code point) error rate on normalised text, aggregated
over a subset as total edits / total reference characters (micro-average).
Also reported: AER (Tamil-letter / akshara error rate), WER, exact-match rate.

The subset score shown on the leaderboard is ``100 × max(0, 1 − CER)``.
"""

from __future__ import annotations

import re
import unicodedata
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


# An explicit "nothing to read" answer. Prompts ask for "[no text]"; a few plain phrasings
# of the same are accepted. It scores as an empty answer: wrong on a test item, correct on
# a blank or effaced control.
_ABSTENTION = re.compile(
    r"^[\s\[\(<{\"'“”‘’*_`-]*(?:"
    r"no\s+(?:legible\s+|readable\s+|visible\s+)?text(?:\s+(?:found|visible|present|detected))?"
    r"|(?:the\s+)?(?:text\s+(?:is\s+)?)?(?:illegible|unreadable|not\s+legible|not\s+readable)"
    r"|blank(?:\s+image)?|empty)"
    r"[\s\]\)>}.!\"'“”‘’*_`-]*$", re.I)


def is_abstention(text: str | None) -> bool:
    return bool(text) and bool(_ABSTENTION.match(text.strip()))


def answer_text(text: str | None) -> str:
    """The answer as scored: missing answers and abstentions count as empty."""
    return "" if not text or is_abstention(text) else text


def letter_count(text: str) -> int:
    """Letters, marks and digits in an answer — what a blank control should not contain."""
    return sum(1 for ch in text if unicodedata.category(ch)[0] in "LMN")


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
