"""Grantha script support.

Grantha is the script the Tamil country used for Sanskrit: the Sanskrit portions of
Pallava, Pandya and Chola copper plates, temple inscriptions and countless palm-leaf
manuscripts. The benchmark's Grantha items are Sanskrit, authored in Devanagari,
converted to Grantha for rendering and to IAST for the reference transcription — so the
image and the target are guaranteed to correspond.

The Unicode Grantha block (U+11300) was laid out in parallel with Devanagari, so most
characters map by a fixed offset.
"""

from __future__ import annotations

import unicodedata

_OFFSET = 0x11300 - 0x0900

_DEVA_INDEP = {
    "अ": "a", "आ": "ā", "इ": "i", "ई": "ī", "उ": "u", "ऊ": "ū", "ऋ": "ṛ", "ॠ": "ṝ",
    "ऌ": "ḷ", "ॡ": "ḹ", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
}
_DEVA_SIGN = {
    "ा": "ā", "ि": "i", "ी": "ī", "ु": "u", "ू": "ū", "ृ": "ṛ", "ॄ": "ṝ", "ॢ": "ḷ", "ॣ": "ḹ",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
}
_DEVA_CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ṅ",
    "च": "c", "छ": "ch", "ज": "j", "झ": "jh", "ञ": "ñ",
    "ट": "ṭ", "ठ": "ṭh", "ड": "ḍ", "ढ": "ḍh", "ण": "ṇ",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v",
    "श": "ś", "ष": "ṣ", "स": "s", "ह": "h",
}
_DEVA_OTHER = {"ं": "ṃ", "ः": "ḥ", "ँ": "m̐", "ऽ": "'", "ॐ": "oṃ", "।": "|", "॥": "||"}
_VIRAMA = "्"
_DIGITS = {chr(0x0966 + i): str(i) for i in range(10)}

# ISO 15919 spellings that differ from IAST; accepted as equivalent when scoring.
ISO_TO_IAST = {"ṁ": "ṃ", "r̥̄": "ṝ", "r̥": "ṛ", "l̥̄": "ḹ", "l̥": "ḷ", "ē": "e", "ō": "o"}


def devanagari_to_iast(text: str) -> str:
    """Romanise Sanskrit written in Devanagari into IAST.

    >>> devanagari_to_iast("स्वस्ति श्री")
    'svasti śrī'
    """
    t = unicodedata.normalize("NFC", text)
    out: list[str] = []
    i, n = 0, len(t)
    while i < n:
        ch = t[i]
        if ch in _DEVA_CONS:
            out.append(_DEVA_CONS[ch])
            nxt = t[i + 1] if i + 1 < n else ""
            if nxt == _VIRAMA:
                i += 2
            elif nxt in _DEVA_SIGN:
                out.append(_DEVA_SIGN[nxt])
                i += 2
            else:
                out.append("a")
                i += 1
            continue
        if ch in _DEVA_INDEP:
            out.append(_DEVA_INDEP[ch])
        elif ch in _DEVA_OTHER:
            out.append(_DEVA_OTHER[ch])
        elif ch in _DIGITS:
            out.append(_DIGITS[ch])
        elif ch == _VIRAMA:
            pass
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def devanagari_to_grantha(text: str) -> str:
    """Convert Devanagari-script Sanskrit to Grantha code points (dandas stay U+0964/5,
    as the Unicode Standard recommends for Grantha)."""
    out = []
    for ch in unicodedata.normalize("NFC", text):
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F and ch not in "।॥":
            g = chr(cp + _OFFSET)
            if unicodedata.name(g, None) is None:
                raise ValueError(f"no Grantha equivalent for {ch!r} (U+{cp:04X})")
            out.append(g)
        else:
            out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def normalize_iast(text: str) -> str:
    """Fold ISO 15919 variants to IAST and lowercase, for lenient comparison."""
    t = unicodedata.normalize("NFC", text).lower()
    for a, b in ISO_TO_IAST.items():
        t = t.replace(unicodedata.normalize("NFC", a), b)
    return t
