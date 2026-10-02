"""Standardised instructions sent to every vision-language model.

All models receive exactly the same text for a given subset; the prompt version is
recorded with every result. Prompts state the *output convention* (what a correct answer
looks like) but never hint at the answer itself.

Every reading prompt says what to answer when there is nothing to read (``NO_TEXT``): the
blank and effaced control items test whether a system invents text instead.

``PROMPT_VARIANTS`` holds two paraphrases of each prompt besides the canonical one
(variant 0). They change the wording, never the output convention, and are used only by
the prompt-robustness diagnostic (Mizrahi et al. 2024); ranked runs always use variant 0.
"""

from __future__ import annotations

from .taxonomy import MEDIUM_ID_LABELS, SCRIPT_ID_LABELS, LabelSet

PROMPT_VERSION = "prompts-v2"

NO_TEXT = "[no text]"
_ABSTAIN = f"If the image contains no legible text, output only {NO_TEXT}"

SYSTEM = ("You are an expert reader of Tamil writing in every period and medium — from "
          "Tamil-Brahmi inscriptions and palm-leaf manuscripts to modern print, handwriting "
          "and screens. Answer with the requested output only.")


def _reading(text: str) -> str:
    return f"{text} {_ABSTAIN}."


RECOGNITION = _reading(
    "Transcribe all of the text in this image exactly as it is written. The text is in Tamil "
    "and may also contain English words, digits or symbols. Write it in Unicode, preserving "
    "the original spelling, punctuation and numerals — including Tamil numerals and signs such "
    "as ௧௨௩, ௰, ௵, ௹ and Tamil fraction signs. Put each line of text on its own line. Do not "
    "translate, correct, explain or add anything. Output only the transcription."
)

RECOGNITION_PALM_LEAF = _reading(
    "This image shows a Tamil palm-leaf manuscript. Transcribe its text into Tamil Unicode, "
    "line by line, as it is written on the leaf. Do not correct the scribe's spelling. "
    "Word spacing, the puḷḷi (virama dot) and the e/ē, o/ō distinction are not scored, so you "
    "may write either exactly what is on the leaf or modern spelling. Output only the "
    "transcription."
)

RECOGNITION_PALM_LEAF_CICT = _reading(
    "This image shows one leaf of a Tamil palm-leaf manuscript of the Tirukkural. Transcribe "
    "the main block of text line by line, exactly as the scribe wrote it — a diplomatic "
    "transcription: keep the scribe's own spellings and slips, and do not substitute the "
    "standard text of the Tirukkural. The last syllable of a line is sometimes written "
    "separately at the right; include it at the end of its line. Ignore the title in the left "
    "margin, the verse numbers at the right edge and any modern catalogue marks. Word spacing, "
    "the puḷḷi and the e/ē, o/ō distinction are not scored. Output only the transcription in "
    "Tamil Unicode, one manuscript line per output line."
)

RECOGNITION_EPIGRAPHIC = _reading(
    "This image shows a Tamil inscription or engraved text. Read it and write the text in "
    "Tamil Unicode. Word spacing, the puḷḷi (virama dot) and the e/ē, o/ō distinction are not "
    "scored. Output only the text, nothing else."
)

RECOGNITION_TAMIL_BRAHMI = _reading(
    "This image shows writing in the Tamil-Brahmi (Tamili) script, the ancient script of the "
    "earliest Tamil inscriptions. Read it and transliterate it into the modern Tamil script "
    "(Unicode). Tamil-Brahmi may or may not mark pure consonants with a dot (puḷḷi) and may not "
    "distinguish long and short e/o; word spacing, the puḷḷi and e/o length are not scored. "
    "Output only the modern Tamil transliteration."
)

RECOGNITION_GRANTHA = _reading(
    "This image shows Sanskrit written in the Grantha script. Read it and transliterate it "
    "into IAST (International Alphabet of Sanskrit Transliteration), e.g. 'svasti śrī'. Word "
    "spacing is not scored. Output only the IAST transliteration."
)

RECOGNITION_GRANTHA_TAMIL = _reading(
    "This image shows Grantha-Tamil (maṇipravāḷam) writing: Tamil in the Tamil script, with "
    "Sanskrit words in the Grantha script, sometimes switching script inside a word. Transcribe "
    "it in reading order, writing the Tamil-script parts in Tamil Unicode and the Grantha-script "
    "parts in IAST, e.g. 'kalyāṇaguṇaங்களை' for a Grantha stem followed by a Tamil suffix. Word "
    "spacing is not scored. Output only the transcription."
)


_CLASSIFICATION_FRAMES = [
    ("{question} Answer with exactly one label from this list:\n{options}\n"
     "Output only the label (for example: {first})."),
    ("{question} Choose one label from the list below and reply with that label alone:\n{options}\n"
     "Example of the expected format: {first}"),
    ("{question}\nPossible answers:\n{options}\nReply with exactly one of these labels and nothing "
     "else (e.g. {first})."),
]


def classification_prompt(question: str, labels: LabelSet, frame: int = 0) -> str:
    options = "\n".join(f"- {lab}: {labels.descriptions.get(lab, lab)}" for lab in labels.labels)
    return _CLASSIFICATION_FRAMES[frame].format(question=question, options=options, first=labels.labels[0])


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

# Paraphrases for the prompt-robustness diagnostic: same conventions, different wording.
PROMPT_VARIANTS: dict[str, list[str]] = {
    "recognition": [RECOGNITION, _reading(
        "Read every piece of text in this image and write it out exactly as it appears. Expect "
        "Tamil, possibly mixed with English words, digits and symbols. Use Unicode and keep the "
        "original spelling, punctuation and numerals, including Tamil numerals and signs (for "
        "example ௧௨௩, ௰, ௵, ௹ and the Tamil fraction signs). Keep each line of the image on a "
        "separate line. Give the transcription only: no translation, correction, explanation or "
        "extra text."), _reading(
        "Task: exact transcription. Copy the text in the image character for character in "
        "Unicode — Tamil, and any English, digits or symbols it contains — keeping its spelling, "
        "punctuation and numerals as written (Tamil numerals and signs such as ௧௨௩, ௰, ௵, ௹ and "
        "fraction signs included). One output line per line of text. Reply with the "
        "transcription alone; do not translate, fix or comment.")],
    "recognition-palm-leaf": [RECOGNITION_PALM_LEAF, _reading(
        "The image is a leaf of a Tamil palm-leaf manuscript. Write out its text in Tamil "
        "Unicode, one line at a time, as the scribe wrote it, without correcting the spelling. "
        "Word spacing, the puḷḷi (virama dot) and the difference between e/ē and o/ō do not "
        "affect the score, so the leaf's own spelling and modern spelling are both acceptable. "
        "Reply with the transcription only."), _reading(
        "Transcribe this Tamil palm-leaf manuscript into Tamil Unicode, line by line, keeping "
        "the scribe's spelling. Spacing between words, the puḷḷi and e/ē, o/ō are ignored in "
        "scoring; write them as on the leaf or as in modern spelling. Output the transcription "
        "and nothing else.")],
    "recognition-palm-leaf-cict": [RECOGNITION_PALM_LEAF_CICT, _reading(
        "The image is a single leaf of a Tamil palm-leaf manuscript of the Tirukkural. Give a "
        "diplomatic transcription of its main text block, line by line: reproduce the scribe's "
        "own spellings and mistakes, and do not replace them with the standard Tirukkural text. "
        "Where the final syllable of a line is written apart on the right, put it at the end of "
        "that line. Leave out the title in the left margin, the verse numbers along the right "
        "edge and any modern catalogue marks. Word spacing, the puḷḷi and the e/ē, o/ō "
        "distinction are not scored. Reply only with the transcription in Tamil Unicode, one "
        "output line per manuscript line."), _reading(
        "Diplomatic transcription task: copy this Tirukkural palm leaf exactly as written, line "
        "by line, in Tamil Unicode — the scribe's spellings and slips included, never the "
        "standard edition's wording. Attach any final syllable written separately at the right "
        "to the end of its line. Skip the left-margin title, the verse numbers on the right "
        "edge and modern catalogue marks. Spacing, the puḷḷi and e/ē, o/ō are not scored. "
        "Output the transcription only, one line per manuscript line.")],
    "recognition-epigraphic": [RECOGNITION_EPIGRAPHIC, _reading(
        "The image shows a Tamil inscription or other engraved text. Read it and give the text "
        "in Tamil Unicode. Word spacing, the puḷḷi (virama dot) and the e/ē, o/ō distinction do "
        "not affect the score. Reply with the text alone."), _reading(
        "Read this engraved Tamil text (an inscription, plaque or plate) and write it in Tamil "
        "Unicode. Spacing, the puḷḷi and e/ē, o/ō length are not scored. Output only the "
        "text.")],
    "recognition-tamil-brahmi": [RECOGNITION_TAMIL_BRAHMI, _reading(
        "The writing in this image is Tamil-Brahmi (Tamili), the script of the oldest Tamil "
        "inscriptions. Read it and give it in the modern Tamil script (Unicode). Tamil-Brahmi "
        "may or may not mark a pure consonant with a dot (puḷḷi) and may not separate short and "
        "long e/o; word spacing, the puḷḷi and e/o length are not scored. Reply with the modern "
        "Tamil transliteration only."), _reading(
        "Transliterate this Tamil-Brahmi (Tamili) writing into the modern Tamil script in "
        "Unicode. The original may or may not mark the puḷḷi and may not distinguish e/o "
        "length; spacing, the puḷḷi and e/o length are ignored in scoring. Output the modern "
        "Tamil text and nothing else.")],
    "recognition-grantha": [RECOGNITION_GRANTHA, _reading(
        "This image shows Sanskrit in the Grantha script. Read it and give it in IAST (the "
        "International Alphabet of Sanskrit Transliteration), for example 'svasti śrī'. Word "
        "spacing does not affect the score. Reply with the IAST transliteration only."), _reading(
        "Transliterate the Grantha-script Sanskrit in this image into IAST, e.g. 'svasti śrī'. "
        "Spacing between words is not scored. Output only the IAST text.")],
    "recognition-grantha-tamil": [RECOGNITION_GRANTHA_TAMIL, _reading(
        "This image shows Grantha-Tamil (maṇipravāḷam) writing: Tamil in the Tamil script, with "
        "Sanskrit words in the Grantha script, sometimes changing script within a single word. "
        "Transcribe it in reading order, giving the Tamil-script parts in Tamil Unicode and the "
        "Grantha-script parts in IAST — for example 'kalyāṇaguṇaங்களை' for a Grantha stem with a "
        "Tamil suffix. Word spacing is not scored. Reply with the transcription only."), _reading(
        "Transcribe this maṇipravāḷam (Grantha-Tamil) text in reading order: Tamil-script "
        "letters as Tamil Unicode, Grantha-script letters as IAST, switching inside a word "
        "where the script switches (e.g. 'kalyāṇaguṇaங்களை'). Spacing is not scored. Output "
        "only the transcription.")],
    "script-id": [SCRIPT_ID,
                  classification_prompt("Identify the writing system (script) used in this image.",
                                        SCRIPT_ID_LABELS, 1),
                  classification_prompt("In which script is the text in this image written?", SCRIPT_ID_LABELS, 2)],
    "medium-id": [MEDIUM_ID,
                  classification_prompt("Identify the surface or medium that carries the text in this image.",
                                        MEDIUM_ID_LABELS, 1),
                  classification_prompt("On what material or display does the text in this image appear?",
                                        MEDIUM_ID_LABELS, 2)],
    "translation": [TRANSLATION,
                    "Translate the Tamil text shown in this image into fluent English. Reply with the English "
                    "translation only.",
                    "Read the Tamil in this image and give its meaning in natural English. Output only the "
                    "translation."],
}
N_VARIANTS = 3


def prompt_text(key: str, variant: int = 0) -> str:
    """The prompt for a prompt key; ``variant`` > 0 selects a paraphrase."""
    if variant == 0:
        return PROMPTS[key]
    return PROMPT_VARIANTS[key][variant]
