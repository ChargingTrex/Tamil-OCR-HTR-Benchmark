"""Tamil script fundamentals: the letter inventory, letter (எழுத்து) segmentation,
normalisation for scoring, and orthographic transforms used to simulate historical
writing conventions.

Terminology
-----------
uyir      உயிர்      the 12 vowels அ … ஔ
mei       மெய்       the 18 consonants with puḷḷi (க் … ன்)
uyirmei   உயிர்மெய்  consonant + vowel (க கா கி …)
āytam     ஆய்தம்     ஃ
puḷḷi     புள்ளி      the virama dot (U+0BCD) marking a vowel-less consonant
grantha   கிரந்தம்    ஜ ஷ ஸ ஹ ஶ (and the conjuncts க்ஷ, ஸ்ரீ)
"""

from __future__ import annotations

import re
import unicodedata

VIRAMA = "்"            # ் puḷḷi
AU_LENGTH_MARK = "ௗ"    # ௗ
AYTHAM = "ஃ"            # ஃ

VOWELS = ["அ", "ஆ", "இ", "ஈ", "உ", "ஊ", "எ", "ஏ", "ஐ", "ஒ", "ஓ", "ஔ"]
# Vowel signs aligned with VOWELS (அ has none: the inherent vowel).
VOWEL_SIGNS = ["", "ா", "ி", "ீ", "ு", "ூ",
               "ெ", "ே", "ை", "ொ", "ோ", "ௌ"]
SIGN_TO_VOWEL = {s: v for s, v in zip(VOWEL_SIGNS, VOWELS) if s}
VOWEL_TO_SIGN = {v: s for s, v in zip(VOWEL_SIGNS, VOWELS)}

NATIVE_CONSONANTS = ["க", "ங", "ச", "ஞ", "ட", "ண", "த", "ந", "ப",
                     "ம", "ய", "ர", "ல", "வ", "ழ", "ள", "ற", "ன"]
GRANTHA_CONSONANTS = ["ஜ", "ஶ", "ஷ", "ஸ", "ஹ"]
CONSONANTS = NATIVE_CONSONANTS + GRANTHA_CONSONANTS

# Word-initial letters that traditional Tamil phonotactics allows (Nannūl); used by the
# pseudo-word generator so nonce words look like Tamil words.
INITIAL_CONSONANTS = ["க", "ச", "ஞ", "த", "ந", "ப", "ம", "ய", "வ"]

TAMIL_DIGITS = "௦௧௨௩௪௫௬௭௮௯"
TAMIL_NUMBER_SIGNS = "௰௱௲"            # 10, 100, 1000
TAMIL_SYMBOLS = "௳௴௵௶௷௸௹௺"          # day, month, year, debit, credit, as-above, rupee, number
# The script-reform syllables: before 1978 these were written with special ligatures.
REFORM_AA = ["ணா", "றா", "னா", "ணொ", "ணோ", "றொ", "றோ", "னொ", "னோ"]
REFORM_AI = ["ணை", "லை", "ளை", "னை"]
REFORM_SYLLABLES = REFORM_AA + REFORM_AI

_TAMIL_BLOCK = re.compile(r"[஀-௿]")
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_PUNCT_MAP = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "«": '"', "»": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
    "−": "-", " ": " ", " ": " ", " ": " ", "　": " ",
    "…": "...", "।": ".", "॥": ".",
})
_WS = re.compile(r"\s+")


def is_tamil(ch: str) -> bool:
    return bool(_TAMIL_BLOCK.match(ch))


def letters(text: str) -> list[str]:
    """Split text into Tamil letters (எழுத்து): uyir, mei, uyirmei, āytam — plus every
    other character as its own unit. This is the unit Tamil readers count, and the unit
    of the akshara error rate (AER).

    >>> letters("தமிழ்")
    ['த', 'மி', 'ழ்']
    >>> letters("கௌரவம்")
    ['கௌ', 'ர', 'வ', 'ம்']
    """
    text = unicodedata.normalize("NFC", text)
    out: list[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        attach = out and (cat in ("Mn", "Mc") or ch in (VIRAMA, AU_LENGTH_MARK))
        if attach and out[-1] and not out[-1][-1].isspace():
            out[-1] += ch
        else:
            out.append(ch)
    return out


def tamil_letter_inventory(include_grantha: bool = True) -> list[str]:
    """The traditional 247 letters (12 + 1 + 18 + 216), optionally with the Grantha
    consonant series and ஸ்ரீ."""
    inv = list(VOWELS) + [AYTHAM]
    cons = NATIVE_CONSONANTS + (GRANTHA_CONSONANTS if include_grantha else [])
    for c in cons:
        inv.append(c + VIRAMA)
        for s in VOWEL_SIGNS:
            inv.append(c + s)
    if include_grantha:
        inv.append("ஸ்ரீ")
    return inv


def normalize(text: str, *, punctuation: bool = True, spaces: str = "collapse",
              strip_tamil_numerals: bool = False) -> str:
    """Canonical form used for scoring.

    - Unicode NFC (composes ொ ோ ௌ and ஔ from their two-part spellings)
    - drops zero-width characters (ZWJ/ZWNJ/ZWSP/BOM/soft hyphen)
    - maps typographic quotes/dashes/ellipsis/non-breaking spaces to ASCII equivalents
    - unifies the two spellings of ஸ்ரீ (ஶ்ரீ → ஸ்ரீ)
    - whitespace: "collapse" (runs → one space; newlines count as spaces) or "remove"
    """
    if text is None:
        return ""
    t = unicodedata.normalize("NFC", text).translate(_ZERO_WIDTH)
    if punctuation:
        t = t.translate(_PUNCT_MAP)
    t = t.replace("ஶ்ரீ", "ஸ்ரீ")
    if strip_tamil_numerals:
        t = re.sub(f"[{TAMIL_DIGITS}{TAMIL_NUMBER_SIGNS}]", "", t)
    if spaces == "remove":
        t = _WS.sub("", t)
    else:
        t = _WS.sub(" ", t).strip()
    return t


def strip_punctuation(text: str) -> str:
    """Remove punctuation and symbols (keeps letters, marks, digits and spaces)."""
    return "".join(ch for ch in text
                   if unicodedata.category(ch)[0] in ("L", "M", "N", "Z") or ch.isspace())


# ---------------------------------------------------------------------------
# Orthographic transforms — used to *render* historical conventions while the
# ground truth keeps the modern spelling.
# ---------------------------------------------------------------------------

def drop_pulli(text: str, prob: float = 1.0, rng=None) -> str:
    """Remove puḷḷi marks (palm-leaf scribes routinely omitted them)."""
    if prob >= 1.0:
        return text.replace(VIRAMA, "")
    return "".join(ch for ch in text if not (ch == VIRAMA and rng.random() < prob))


def merge_long_e_o(text: str) -> str:
    """Write ē/ō with the short-vowel signs, as before Beschi's 18th-century reform:
    ே→ெ, ோ→ொ, ஏ→எ, ஓ→ஒ."""
    t = unicodedata.normalize("NFD", text)  # ோ → ெ + ா ; ொ → ெ + ா as well
    t = t.replace("ே", "ெ").replace("ஏ", "எ").replace("ஓ", "ஒ")
    return unicodedata.normalize("NFC", t)


def scriptio_continua(text: str) -> str:
    """Remove word spaces (inscriptions and leaves rarely separate words)."""
    return _WS.sub("", text)


def contains_reform_syllable(text: str) -> bool:
    return any(s in text for s in REFORM_SYLLABLES)


def has_grantha(text: str) -> bool:
    return any(c in text for c in GRANTHA_CONSONANTS) or "ஸ்ரீ" in text


def tamil_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    return sum(is_tamil(c) for c in chars) / max(1, len(chars))
