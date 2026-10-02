"""Grantha–Tamil mixed writing (maṇipravāḷam).

For centuries Tamil manuscripts on religion, philosophy, astrology and medicine were
written in two scripts at once: Tamil words in the Tamil script and Sanskrit words in
Grantha, often inside a single word, where a Sanskrit stem in Grantha takes a Tamil
suffix in Tamil letters (kalyāṇaguṇa + ங்களை). Tamil inscriptions likewise open with
*svasti śrī* in Grantha before the Tamil text.

Corpus markup: Grantha segments are written in IAST inside braces and Tamil segments as
Tamil text, e.g. ``{kalyāṇaguṇa}ங்களை``. The *reference* keeps the IAST, so answers write
Tamil-script parts in Tamil Unicode and Grantha parts in IAST (``kalyāṇaguṇaங்களை``).
The *native* form converts the IAST into the Unicode Grantha block for rendering.
"""

from __future__ import annotations

import random
import re

from .grantha import iast_to_grantha
from .pseudo import pseudo_word

Segment = tuple[str, str]   # ("gr" | "ta", text)

_TAMIL = re.compile(r"^[஀-௿]+$")
_IAST = re.compile(r"^[a-zāīūṛṝḷḹṅñṭḍṇśṣṃḥ']+$")


def parse(markup: str) -> list[list[Segment]]:
    """Split a marked-up line into words, each a list of (script, text) segments."""
    words: list[list[Segment]] = []
    cur: list[Segment] = []
    grantha = False
    for ch in markup:
        if ch in "{}":
            if (ch == "{") == grantha:
                raise ValueError(f"unbalanced braces in {markup!r}")
            grantha = ch == "{"
            continue
        if ch.isspace():
            if cur:
                words.append(cur)
                cur = []
            continue
        script = "gr" if grantha else "ta"
        if cur and cur[-1][0] == script:
            cur[-1] = (script, cur[-1][1] + ch)
        else:
            cur.append((script, ch))
    if grantha:
        raise ValueError(f"unclosed brace in {markup!r}")
    if cur:
        words.append(cur)
    for word in words:
        for script, text in word:
            if not (_IAST if script == "gr" else _TAMIL).match(text):
                raise ValueError(f"{text!r} is not {'IAST' if script == 'gr' else 'Tamil'} in {markup!r}")
    return words


def native(word: list[Segment]) -> str:
    """The word as written: Tamil segments as-is, Grantha segments in the Grantha block."""
    return "".join(iast_to_grantha(text) if script == "gr" else text for script, text in word)


def reference(word: list[Segment]) -> str:
    """The word as transcribed: Tamil segments in Tamil, Grantha segments in IAST."""
    return "".join(text for _, text in word)


def has_grantha(words: list[list[Segment]]) -> bool:
    return any(script == "gr" for word in words for script, _ in word)


# ---- nonce lines (prior-free control) ------------------------------------------------------

_ONSETS = ["k", "t", "p", "n", "m", "r", "v", "s", "dh", "g", "y", "j", "ś", "h", "l", "d", "bh", "ṣ", "kṣ", "pr"]
_VOWELS = ["a", "a", "ā", "i", "ī", "u", "e", "o"]
_SUFFIXES = ["ங்களை", "த்தில்", "த்தை", "னுடைய", "த்துக்கு", "ம்", "த்தாலே", "ப்பித்து", "க்கிறான்",
             "ர்கள்", "னே", "மே", "ங்கள்", "த்தின்"]


def _stem(rng: random.Random) -> str:
    syl = [rng.choice(_ONSETS) + rng.choice(_VOWELS) for _ in range(rng.randint(1, 3))]
    return "".join(syl) + rng.choice(_ONSETS) + "a"   # ends in -a, so a Tamil suffix attaches


def nonce_markup(rng: random.Random, n_words: tuple[int, int] = (3, 6)) -> str:
    """A marked-up line of invented words with maṇipravāḷam structure: Grantha stems with
    Tamil suffixes, bare Grantha words and Tamil-like words."""
    words = []
    for _ in range(rng.randint(*n_words)):
        r = rng.random()
        if r < 0.55:
            words.append("{" + _stem(rng) + "}" + rng.choice(_SUFFIXES))
        elif r < 0.75:
            words.append("{" + _stem(rng) + "}")
        else:
            words.append(pseudo_word(rng))
    return " ".join(words)
