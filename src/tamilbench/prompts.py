"""Standardised instructions sent to every vision-language model.

All models receive exactly the same text for a given subset; the prompt version is
recorded with every result. Prompts state the *output convention* (what a correct answer
looks like) but never hint at the answer itself.
"""

from __future__ import annotations

from .taxonomy import MEDIUM_ID_LABELS, SCRIPT_ID_LABELS, LabelSet

PROMPT_VERSION = "prompts-v1"

SYSTEM = ("You are an expert reader of Tamil writing in every period and medium — from "
          "Tamil-Brahmi inscriptions and palm-leaf manuscripts to modern print, handwriting "
          "and screens. Answer with the requested output only.")

RECOGNITION = (
    "Transcribe all of the text in this image exactly as it is written. The text is in Tamil "
    "and may also contain English words, digits or symbols. Write it in Unicode, preserving "
    "the original spelling, punctuation and numerals — including Tamil numerals and signs such "
    "as ௧௨௩, ௰, ௵, ௹ and Tamil fraction signs. Put each line of text on its own line. Do not "
    "translate, correct, explain or add anything. Output only the transcription."
)

RECOGNITION_PALM_LEAF = (
    "This image shows a Tamil palm-leaf manuscript. Transcribe its text into Tamil Unicode, "
    "line by line, as it is written on the leaf. Do not correct the scribe's spelling. "
    "Word spacing, the puḷḷi (virama dot) and the e/ē, o/ō distinction are not scored, so you "
    "may write either exactly what is on the leaf or modern spelling. Output only the "
    "transcription."
)

RECOGNITION_PALM_LEAF_CICT = (
    "This image shows one leaf of a Tamil palm-leaf manuscript of the Tirukkural. Transcribe "
    "the main block of text line by line, exactly as the scribe wrote it — a diplomatic "
    "transcription: keep the scribe's own spellings and slips, and do not substitute the "
    "standard text of the Tirukkural. The last syllable of a line is sometimes written "
    "separately at the right; include it at the end of its line. Ignore the title in the left "
    "margin, the verse numbers at the right edge and any modern catalogue marks. Word spacing, "
    "the puḷḷi and the e/ē, o/ō distinction are not scored. Output only the transcription in "
    "Tamil Unicode, one manuscript line per output line."
)

RECOGNITION_EPIGRAPHIC = (
    "This image shows a Tamil inscription or engraved text. Read it and write the text in "
    "Tamil Unicode. Word spacing, the puḷḷi (virama dot) and the e/ē, o/ō distinction are not "
    "scored. Output only the text, nothing else."
)

RECOGNITION_TAMIL_BRAHMI = (
    "This image shows writing in the Tamil-Brahmi (Tamili) script, the ancient script of the "
    "earliest Tamil inscriptions. Read it and transliterate it into the modern Tamil script "
    "(Unicode). Tamil-Brahmi may or may not mark pure consonants with a dot (puḷḷi) and may not "
    "distinguish long and short e/o; word spacing, the puḷḷi and e/o length are not scored. "
    "Output only the modern Tamil transliteration."
)

RECOGNITION_GRANTHA = (
    "This image shows Sanskrit written in the Grantha script. Read it and transliterate it "
    "into IAST (International Alphabet of Sanskrit Transliteration), e.g. 'svasti śrī'. Word "
    "spacing is not scored. Output only the IAST transliteration."
)

RECOGNITION_GRANTHA_TAMIL = (
    "This image shows Grantha-Tamil (maṇipravāḷam) writing: Tamil in the Tamil script, with "
    "Sanskrit words in the Grantha script, sometimes switching script inside a word. Transcribe "
    "it in reading order, writing the Tamil-script parts in Tamil Unicode and the Grantha-script "
    "parts in IAST, e.g. 'kalyāṇaguṇaங்களை' for a Grantha stem followed by a Tamil suffix. Word "
    "spacing is not scored. Output only the transcription."
)


def classification_prompt(question: str, labels: LabelSet) -> str:
    options = "\n".join(f"- {lab}: {labels.descriptions.get(lab, lab)}" for lab in labels.labels)
    return (f"{question} Answer with exactly one label from this list:\n{options}\n"
            "Output only the label (for example: " + labels.labels[0] + ").")


SCRIPT_ID = classification_prompt("Which script is the writing in this image in?", SCRIPT_ID_LABELS)
MEDIUM_ID = classification_prompt("What is the text in this image written on, or displayed on?",
                                  MEDIUM_ID_LABELS)

TRANSLATION = (
    "Read the Tamil text in this image and translate it into natural English. "
    "Output only the English translation."
)

PROMPTS = {
    "recognition": RECOGNITION,
    "recognition-palm-leaf": RECOGNITION_PALM_LEAF,
    "recognition-palm-leaf-cict": RECOGNITION_PALM_LEAF_CICT,
    "recognition-epigraphic": RECOGNITION_EPIGRAPHIC,
    "recognition-tamil-brahmi": RECOGNITION_TAMIL_BRAHMI,
    "recognition-grantha": RECOGNITION_GRANTHA,
    "recognition-grantha-tamil": RECOGNITION_GRANTHA_TAMIL,
    "script-id": SCRIPT_ID,
    "medium-id": MEDIUM_ID,
    "translation": TRANSLATION,
}
