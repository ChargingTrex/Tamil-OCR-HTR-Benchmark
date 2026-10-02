"""Diagnostics that a character error rate alone hides. They are reported, never ranked.

* **Failure modes of generative readers**: empty answers, overlong answers, repetition
  loops, Markdown/HTML emission, and letters in a script that the subset's references
  never use ("off-script" generation, e.g. Devanagari for a Grantha image or Malayalam for
  Tamil-Brahmi). Vision-language models make these errors and traditional recognisers
  rarely do (Karamolegkou et al. 2026; Yang et al. 2024, CC-OCR; "When Low CER is Not
  Enough", 2026).
* **Order-free error**: a bag-of-characters error rate that ignores reading order. Its gap
  to the order-sensitive CER (both without spaces) shows how much of a block or page score
  is lost to serialisation rather than recognition (cf. Flexible Character Accuracy,
  Clausner et al. 2020; the Character Error Vector, Bourne et al. 2026).
* **Sample-level normalised edit distance**, the text metric of OmniDocBench, for
  comparison with document-parsing benchmarks.
* **Letter confusions**: the most frequent Tamil-letter substitutions (e.g. ல→ள, ண→ன).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from functools import lru_cache

from .edit import align, levenshtein

try:  # pragma: no cover - optional speed-up
    from rapidfuzz.distance import Levenshtein as _RF
except Exception:  # noqa: BLE001
    _RF = None

COMMON, INHERITED = "Common", "Inherited"

_MARKUP = [
    re.compile(r"(?m)^\s{0,3}#{1,6}\s"),                      # Markdown heading
    re.compile(r"\*\*[^*\n]+\*\*"),                            # bold
    re.compile(r"</?[A-Za-z][A-Za-z0-9]*(\s[^<>]*)?/?>"),       # HTML tag
    re.compile(r"(?m)^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$"),  # table rule
    re.compile(r"```"),                                        # code fence
    re.compile(r"\\(text|mathrm|textbf)\{|\$\$"),               # LaTeX
]
_REPEAT = re.compile(r"(.{3,}?)\1{4,}", re.S)                  # a unit of 3+ chars, 5+ times in a row


@lru_cache(maxsize=4096)
def script_of(ch: str) -> str:
    """The Unicode script of a letter or mark, from its character name; Common otherwise.
    Tamil-block digits and signs count as Tamil."""
    name = unicodedata.name(ch, "")
    if name.startswith("TAMIL "):
        return "TAMIL"
    cat = unicodedata.category(ch)
    if cat[0] not in "LM":
        return COMMON
    first = name.split(" ", 1)[0] if name else "UNKNOWN"
    if first == "COMBINING":
        return INHERITED
    if first in ("MODIFIER", "SUPERSCRIPT", "SUBSCRIPT"):
        return COMMON
    return first


def scripts_in(text: str) -> set[str]:
    return {script_of(c) for c in text} - {COMMON, INHERITED}


def offscript_letters(hyp: str, allowed: set[str]) -> tuple[int, int]:
    """(letters in a script outside ``allowed``, all letters) in ``hyp``."""
    n = bad = 0
    for c in hyp:
        s = script_of(c)
        if s in (COMMON, INHERITED):
            continue
        n += 1
        bad += s not in allowed
    return bad, n


def has_markup(text: str) -> bool:
    return any(p.search(text) for p in _MARKUP)


def has_repetition(text: str) -> bool:
    return bool(_REPEAT.search(text))


def bag_edits(ref: str, hyp: str) -> int:
    """Order-free edit count: the fewest substitutions, insertions and deletions that turn
    the multiset of ``ref``'s characters into ``hyp``'s, ignoring order and whitespace."""
    r = Counter(c for c in ref if not c.isspace())
    h = Counter(c for c in hyp if not c.isspace())
    missing = sum((r - h).values())
    extra = sum((h - r).values())
    return max(missing, extra)


def item_diagnostics(ref_norm: str, hyp_norm: str, hyp_raw: str | None, allowed: set[str]) -> dict:
    """Per-item diagnostic counts; ``ref_norm`` and ``hyp_norm`` are policy-normalised."""
    raw = hyp_raw or ""
    r_ns = "".join(ref_norm.split())
    h_ns = "".join(hyp_norm.split())
    bad, letters = offscript_letters(hyp_norm, allowed)
    lev = levenshtein(ref_norm, hyp_norm)
    return {
        "empty": int(not hyp_norm.strip()),
        "overlong": int(len(hyp_norm) > 1.5 * len(ref_norm) + 10),
        "repetition": int(has_repetition(hyp_norm) and not has_repetition(ref_norm)),
        "markup": int(has_markup(raw) and not has_markup(ref_norm)),
        "offscript": bad, "hyp_letters": letters,
        "offscript_item": int(bad > 0),
        "nospace_edits": levenshtein(r_ns, h_ns), "nospace_chars": len(r_ns),
        "bag_edits": bag_edits(r_ns, h_ns),
        "ned": lev / max(1, len(ref_norm), len(hyp_norm)),
    }


def aggregate_diagnostics(items: list[dict]) -> dict:
    n = len(items)
    if not n:
        return {}
    s = Counter()
    for d in items:
        s.update({k: v for k, v in d.items() if k != "ned"})
    chars = max(1, s["nospace_chars"])
    return {
        "empty_rate": round(s["empty"] / n, 4),
        "overlong_rate": round(s["overlong"] / n, 4),
        "repetition_rate": round(s["repetition"] / n, 4),
        "markup_rate": round(s["markup"] / n, 4),
        "offscript_item_rate": round(s["offscript_item"] / n, 4),
        "offscript_letter_rate": round(s["offscript"] / max(1, s["hyp_letters"]), 4),
        "cer_nospace": round(s["nospace_edits"] / chars, 4),
        "bag_cer": round(s["bag_edits"] / chars, 4),
        "order_gap": round((s["nospace_edits"] - s["bag_edits"]) / chars, 4),
        "ned": round(sum(d["ned"] for d in items) / n, 4),
    }


def _editops(a: list[str], b: list[str]):
    if _RF is not None:
        for op in _RF.editops(a, b):
            yield ({"replace": "S", "delete": "D", "insert": "I"}[op.tag],
                   a[op.src_pos] if op.tag != "insert" else None,
                   b[op.dest_pos] if op.tag != "delete" else None)
        return
    for op, x, y in align(a, b):
        if op != "=":
            yield op, x, y


def letter_confusions(pairs: list[tuple[list[str], list[str]]], top: int = 12) -> dict:
    """Most frequent letter substitutions, deletions and insertions over (ref, hyp) letter lists."""
    subs, dels, ins = Counter(), Counter(), Counter()
    for ref_letters, hyp_letters in pairs:
        for op, x, y in _editops(ref_letters, hyp_letters):
            if op == "S":
                subs[(x, y)] += 1
            elif op == "D":
                dels[x] += 1
            else:
                ins[y] += 1
    return {
        "substitutions": [[a, b, n] for (a, b), n in subs.most_common(top)],
        "deleted": [[a, n] for a, n in dels.most_common(top)],
        "inserted": [[a, n] for a, n in ins.most_common(top)],
    }
