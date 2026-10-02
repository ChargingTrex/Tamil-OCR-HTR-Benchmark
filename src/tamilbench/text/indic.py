"""Render Tamil text in neighbouring Brahmic scripts.

Used only to make *distractor* images for the script-identification task: the
words are Tamil written in Malayalam/Kannada/Telugu/Devanagari/Sinhala letters, which
keeps the visual statistics of each script while making the task about the script, not
the language. (Tamil words in Malayalam script are a deliberately hard case — the two
scripts share a common ancestor.)
"""

from __future__ import annotations

import unicodedata

from ..taxonomy import Script

_OFFSETS = {
    Script.MALAYALAM: 0x0D00 - 0x0B80,
    Script.TELUGU: 0x0C00 - 0x0B80,
    Script.KANNADA: 0x0C80 - 0x0B80,
    Script.DEVANAGARI: 0x0900 - 0x0B80,
}

# Tamil letters whose parallel slot is empty or archaic in the target script.
_OVERRIDES = {
    Script.MALAYALAM: {"ன": "ന"},
    Script.TELUGU: {"ன": "న", "ழ": "ళ"},
    Script.KANNADA: {"ன": "ನ", "ழ": "ಳ"},
    Script.DEVANAGARI: {"ன": "न", "ழ": "ळ", "ற": "र", "எ": "ए", "ஒ": "ओ",
                        "ெ": "े", "ொ": "ो"},
}

_SINHALA = {
    "அ": "අ", "ஆ": "ආ", "இ": "ඉ", "ஈ": "ඊ", "உ": "උ", "ஊ": "ඌ", "எ": "එ", "ஏ": "ඒ",
    "ஐ": "ඓ", "ஒ": "ඔ", "ஓ": "ඕ", "ஔ": "ඖ",
    "க": "ක", "ங": "ඞ", "ச": "ච", "ஞ": "ඤ", "ட": "ට", "ண": "ණ", "த": "ත", "ந": "න",
    "ப": "ප", "ம": "ම", "ய": "ය", "ர": "ර", "ல": "ල", "வ": "ව", "ழ": "ළ", "ள": "ළ",
    "ற": "ර", "ன": "න", "ஜ": "ජ", "ஶ": "ශ", "ஷ": "ෂ", "ஸ": "ස", "ஹ": "හ",
    "ா": "ා", "ி": "ි", "ீ": "ී", "ு": "ු",
    "ூ": "ූ", "ெ": "ෙ", "ே": "ේ", "ை": "ෛ",
    "ொ": "ො", "ோ": "ෝ", "ௌ": "ෞ", "்": "්",
    "ஃ": "ඃ",
}

SUPPORTED_SCRIPTS = (Script.MALAYALAM, Script.KANNADA, Script.TELUGU, Script.DEVANAGARI, Script.SINHALA)


def convert(text: str, script: Script) -> str:
    """Transliterate Tamil text into ``script``. Non-Tamil characters pass through."""
    t = unicodedata.normalize("NFC", text)
    if script == Script.SINHALA:
        out = [_SINHALA.get(ch, ch) for ch in t]
        return unicodedata.normalize("NFC", "".join(out))
    if script not in _OFFSETS:
        raise ValueError(f"unsupported target script {script}")
    off, over = _OFFSETS[script], _OVERRIDES.get(script, {})
    out = []
    for ch in t:
        if ch in over:
            out.append(over[ch])
        elif 0x0B80 <= ord(ch) <= 0x0BFF:
            g = chr(ord(ch) + off)
            if unicodedata.name(g, None) is None:
                raise ValueError(f"{ch!r} has no {script.value} equivalent")
            out.append(g)
        else:
            out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))
