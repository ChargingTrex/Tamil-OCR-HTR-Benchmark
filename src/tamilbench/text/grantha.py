"""Grantha script support.

Grantha is the script the Tamil country used to write Sanskrit: the Sanskrit portions of
Pallava, Pandya and Chola copper plates, temple inscriptions and countless palm-leaf
manuscripts. The benchmark's Grantha items are Sanskrit authored in IAST (the
International Alphabet of Sanskrit Transliteration), which is also the answer key, and
converted letter by letter into the Unicode Grantha block for rendering — so the image and
the reference always come from one source.
"""

from __future__ import annotations

import unicodedata

# IAST letter -> Grantha code point. Two-letter aspirates are matched before single letters.
_CONS = {
    "kh": 0x11316, "gh": 0x11318, "ch": 0x1131B, "jh": 0x1131D, "ṭh": 0x11320, "ḍh": 0x11322,
    "th": 0x11325, "dh": 0x11327, "ph": 0x1132B, "bh": 0x1132D,
    "k": 0x11315, "g": 0x11317, "ṅ": 0x11319, "c": 0x1131A, "j": 0x1131C, "ñ": 0x1131E,
    "ṭ": 0x1131F, "ḍ": 0x11321, "ṇ": 0x11323, "t": 0x11324, "d": 0x11326, "n": 0x11328,
    "p": 0x1132A, "b": 0x1132C, "m": 0x1132E, "y": 0x1132F, "r": 0x11330, "l": 0x11332,
    "ḻ": 0x11333, "v": 0x11335, "ś": 0x11336, "ṣ": 0x11337, "s": 0x11338, "h": 0x11339,
}
# IAST vowel -> (independent letter, dependent sign); the inherent "a" has no sign.
_VOW = {
    "ai": (0x11310, 0x11348), "au": (0x11314, 0x1134C), "a": (0x11305, None), "ā": (0x11306, 0x1133E),
    "i": (0x11307, 0x1133F), "ī": (0x11308, 0x11340), "u": (0x11309, 0x11341), "ū": (0x1130A, 0x11342),
    "ṝ": (0x11360, 0x11344), "ṛ": (0x1130B, 0x11343), "ḹ": (0x11361, 0x11363), "ḷ": (0x1130C, 0x11362),
    "e": (0x1130F, 0x11347), "o": (0x11313, 0x1134B),
}
_VIRAMA, _ANUSVARA, _VISARGA, _CANDRABINDU, _AVAGRAHA, _OM = (chr(c) for c in (
    0x1134D, 0x11302, 0x11303, 0x11301, 0x1133D, 0x11350))
_DANDA, _DOUBLE_DANDA = "।", "॥"   # the Unicode Standard uses these daṇḍas with Grantha

# ISO 15919 spellings that differ from IAST; accepted as equivalent when scoring.
ISO_TO_IAST = {"ṁ": "ṃ", "r̥̄": "ṝ", "r̥": "ṛ", "l̥̄": "ḹ", "l̥": "ḷ", "ē": "e", "ō": "o"}


def _match(t: str, i: int, table: dict) -> str | None:
    for size in (2, 1):
        if t[i:i + size] in table:
            return t[i:i + size]
    return None


def iast_to_grantha(text: str) -> str:
    """Write IAST-romanised Sanskrit in the Grantha script.

    >>> iast_to_grantha("svasti śrī") == "\\U00011338\\U0001134D\\U00011335\\U00011338\\U0001134D\\U00011324\\U0001133F \\U00011336\\U0001134D\\U00011330\\U00011340"
    True
    """
    t = unicodedata.normalize("NFC", text.lower())
    out: list[str] = []
    i, n = 0, len(t)
    while i < n:
        # the sacred syllable oṃ as a word of its own takes the GRANTHA OM sign
        if t.startswith("oṃ", i) and (i == 0 or not t[i - 1].isalpha()) and (i + 2 == n or not t[i + 2].isalpha()):
            out.append(_OM)
            i += 2
            continue
        c = _match(t, i, _CONS)
        if c:
            out.append(chr(_CONS[c]))
            i += len(c)
            v = _match(t, i, _VOW)
            if v:
                sign = _VOW[v][1]
                if sign:
                    out.append(chr(sign))
                i += len(v)
            else:
                out.append(_VIRAMA)
            continue
        v = _match(t, i, _VOW)
        if v:
            out.append(chr(_VOW[v][0]))
            i += len(v)
            continue
        if t.startswith("m̐", i):
            out.append(_CANDRABINDU)
            i += 2
            continue
        ch = t[i]
        out.append({"ṃ": _ANUSVARA, "ḥ": _VISARGA, "'": _AVAGRAHA, "|": _DANDA}.get(ch, ch))
        i += 1
    return unicodedata.normalize("NFC", "".join(out)).replace(_DANDA + _DANDA, _DOUBLE_DANDA)


_DANDAS = str.maketrans("", "", "|।॥.")


def normalize_iast(text: str) -> str:
    """Fold ISO 15919 variants to IAST, lowercase, and drop daṇḍa marks (written as | || or
    । ॥ or full stops), which are punctuation rather than reading."""
    t = unicodedata.normalize("NFC", text).lower()
    for a, b in ISO_TO_IAST.items():
        t = t.replace(unicodedata.normalize("NFC", a), b)
    return t.translate(_DANDAS)
