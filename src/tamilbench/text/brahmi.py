"""Tamil → Tamil-Brahmi transliteration (Unicode Brahmi block, U+11000–U+1107F).

Tamil-Brahmi is the earliest attested Tamil writing. Iravatham Mahadevan (Early Tamil
Epigraphy, 2003/2014) distinguishes three orthographic stages, which differ in how a
bare consonant sign is read:

  TB-I   no puḷḷi; a bare consonant is *either* a pure consonant or consonant+a —
         the reader decides from the language. ā is marked by a stroke.
  TB-II  no puḷḷi; a bare consonant is a pure consonant, and both a and ā carry the
         stroke (so ka and kā look alike).
  TB-III the puḷḷi (dot, U+11070 BRAHMI SIGN OLD TAMIL VIRAMA) marks pure consonants;
         bare consonants carry the inherent a — i.e. today's Tamil convention.

The benchmark renders all three. The ground truth is always the modern Tamil spelling,
so TB-I/TB-II items test whether a model can resolve the ambiguity the way an
epigraphist does. Short e/o are written with the Old Tamil dotted forms only in
TB-III (as Tolkāppiyam describes); earlier stages do not distinguish e/ē, o/ō.
"""

from __future__ import annotations

import unicodedata
from enum import Enum

from .tamil import AYTHAM, VIRAMA


class Orthography(str, Enum):
    TB1 = "TB-I"
    TB2 = "TB-II"
    TB3 = "TB-III"


B_VIRAMA = "\U00011046"          # Brahmi virama (forms conjuncts) – not used for Tamil
B_OLD_TAMIL_VIRAMA = "\U00011070"  # puḷḷi
B_AA = "\U00011038"

INDEPENDENT = {
    "அ": "\U00011005", "ஆ": "\U00011006", "இ": "\U00011007", "ஈ": "\U00011008",
    "உ": "\U00011009", "ஊ": "\U0001100A", "எ": "\U0001100F", "ஏ": "\U0001100F",
    "ஐ": "\U00011010", "ஒ": "\U00011011", "ஓ": "\U00011011", "ஔ": "\U00011012",
}
INDEPENDENT_SHORT_TB3 = {"எ": "\U00011071", "ஒ": "\U00011072"}

CONSONANT = {
    "க": "\U00011013", "ங": "\U00011017", "ச": "\U00011018", "ஞ": "\U0001101C",
    "ட": "\U0001101D", "ண": "\U00011021", "த": "\U00011022", "ந": "\U00011026",
    "ப": "\U00011027", "ம": "\U0001102B", "ய": "\U0001102C", "ர": "\U0001102D",
    "ல": "\U0001102E", "வ": "\U0001102F",
    "ழ": "\U00011035",  # BRAHMI LETTER OLD TAMIL LLLA
    "ள": "\U00011034",  # BRAHMI LETTER LLA
    "ற": "\U00011036",  # BRAHMI LETTER OLD TAMIL RRA
    "ன": "\U00011037",  # BRAHMI LETTER OLD TAMIL NNNA
    # Prakrit/Grantha sounds occasionally found in Tamil-Brahmi records
    "ஜ": "\U0001101A", "ஶ": "\U00011030", "ஷ": "\U00011031", "ஸ": "\U00011032", "ஹ": "\U00011033",
}
# Variant glyph: U+11075 BRAHMI LETTER OLD TAMIL LLA, attested in some inscriptions.
LLA_VARIANT = "\U00011075"

SIGN = {
    "ா": "\U00011038",  # ா
    "ி": "\U0001103A",  # ி
    "ீ": "\U0001103B",  # ீ
    "ு": "\U0001103C",  # ு
    "ூ": "\U0001103D",  # ூ
    "ெ": "\U00011042",  # ெ (short e)
    "ே": "\U00011042",  # ே
    "ை": "\U00011043",  # ை
    "ொ": "\U00011044",  # ொ (short o)
    "ோ": "\U00011044",  # ோ
    "ௌ": "\U00011045",  # ௌ
}
SIGN_SHORT_TB3 = {"ெ": "\U00011073", "ொ": "\U00011074"}

SUPPORTED = set(INDEPENDENT) | set(CONSONANT) | set(SIGN) | {VIRAMA, " "}


def can_transliterate(text: str) -> bool:
    """True if every character has a Tamil-Brahmi rendering (no digits, āytam, punctuation)."""
    t = unicodedata.normalize("NFC", text)
    return all(ch in SUPPORTED for ch in t) and AYTHAM not in t


def to_brahmi(text: str, orthography: Orthography = Orthography.TB3, *,
              lla_variant: bool = False) -> str:
    """Transliterate modern-Tamil text into Tamil-Brahmi code points.

    >>> to_brahmi("தமிழ்") == "\\U00011022\\U0001102B\\U0001103A\\U00011035\\U00011070"
    True
    """
    t = unicodedata.normalize("NFC", text)
    if not can_transliterate(t):
        bad = sorted({c for c in t if c not in SUPPORTED})
        raise ValueError(f"cannot render in Tamil-Brahmi: {bad!r}")
    out: list[str] = []
    i = 0
    n = len(t)
    while i < n:
        ch = t[i]
        if ch == " ":
            out.append(" ")
            i += 1
            continue
        if ch in INDEPENDENT:
            if orthography == Orthography.TB3 and ch in INDEPENDENT_SHORT_TB3:
                out.append(INDEPENDENT_SHORT_TB3[ch])
            else:
                out.append(INDEPENDENT[ch])
            i += 1
            continue
        if ch in CONSONANT:
            base = CONSONANT[ch]
            if ch == "ள" and lla_variant:
                base = LLA_VARIANT
            nxt = t[i + 1] if i + 1 < n else ""
            if nxt == VIRAMA:                       # pure consonant
                out.append(base + (B_OLD_TAMIL_VIRAMA if orthography == Orthography.TB3 else ""))
                i += 2
            elif nxt in SIGN:                       # consonant + vowel sign
                sign = SIGN[nxt]
                if orthography == Orthography.TB3 and nxt in SIGN_SHORT_TB3:
                    sign = SIGN_SHORT_TB3[nxt]
                out.append(base + sign)
                i += 2
            else:                                   # consonant + inherent a
                out.append(base + (B_AA if orthography == Orthography.TB2 else ""))
                i += 1
            continue
        raise ValueError(f"unexpected character {ch!r} in {text!r}")
    return "".join(out)
