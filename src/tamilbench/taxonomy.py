"""The benchmark's vocabulary: scripts (eras), writing media, tasks and leaderboard tracks.

Everything that appears in a manifest, a prompt or the leaderboard is defined here once,
with an English and a Tamil display name, so the data builder, the scorer and the
website cannot drift apart.

Dates are conventional palaeographic ranges and are deliberately written as "c." —
several (notably the start of Tamil-Brahmi) are actively debated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Script(str, Enum):
    """Writing systems / script stages. The first seven are the Tamil lineage under test;
    the rest are distractors used by the script-identification task."""

    TAMIL_BRAHMI = "tamil-brahmi"
    VATTELUTTU = "vatteluttu"
    GRANTHA = "grantha"
    GRANTHA_TAMIL = "grantha-tamil"
    TAMIL_MEDIEVAL = "tamil-medieval"
    TAMIL_PRE_REFORM = "tamil-pre-reform"
    TAMIL_MODERN = "tamil-modern"
    MALAYALAM = "malayalam"
    KANNADA = "kannada"
    TELUGU = "telugu"
    SINHALA = "sinhala"
    LATIN = "latin"


class Medium(str, Enum):
    """The physical (or digital) support the text was written on."""

    BORN_DIGITAL = "born-digital"
    SCREEN = "screen"
    PRINT_SCAN = "print-scan"
    PRINT_PHOTO = "print-photo"
    HANDWRITTEN = "handwritten"
    SCENE = "scene"
    PALM_LEAF = "palm-leaf"
    STONE = "stone"
    ESTAMPAGE = "estampage"
    COPPER_PLATE = "copper-plate"
    POTTERY = "pottery"


class Task(str, Enum):
    RECOGNITION = "recognition"   # OCR / HTR / epigraphic reading -> Unicode text
    SCRIPT_ID = "script-id"       # which script / script stage is this?
    MEDIUM_ID = "medium-id"       # what is the text written on?
    TRANSLATION = "translation"   # read the Tamil and translate it into English


class Granularity(str, Enum):
    CHARACTER = "character"
    WORD = "word"
    LINE = "line"
    BLOCK = "block"   # a paragraph, a UI panel, a plaque, a few lines of an inscription
    PAGE = "page"     # a full page / a full palm leaf


class Provenance(str, Enum):
    SYNTHETIC = "synthetic"   # procedurally rendered by this repository
    REAL = "real"             # photographs / scans of real artefacts with human transcriptions


class Track(str, Enum):
    """Leaderboard columns, by medium and task. Each subset belongs to exactly one track; a
    track's score is the unweighted mean of its subsets. The headline scores are not means
    of tracks: they weigh the modern script and the older scripts equally (see ``Era``).
    ``DIAGNOSTICS`` holds control subsets that are reported but never ranked."""

    PRINT = "print"
    SCREEN = "screen"
    HANDWRITING = "handwriting"
    SCENE = "scene"
    MANUSCRIPTS = "manuscripts"
    EPIGRAPHY = "epigraphy"
    CLASSIFICATION = "classification"
    TRANSLATION = "translation"
    DIAGNOSTICS = "diagnostics"


class Era(str, Enum):
    """The two halves of the benchmark, weighted equally in every headline score: the
    reformed script of today, and everything older (Tamil-Brahmi, Vatteluttu, Grantha,
    Grantha–Tamil, medieval and pre-reform Tamil)."""

    MODERN = "modern"
    OLDER = "older"


@dataclass(frozen=True)
class ScriptInfo:
    name: str
    tamil: str
    period: str
    description: str
    in_unicode: bool
    lineage: bool = True  # part of the Tamil lineage under test (vs a distractor)


SCRIPTS: dict[Script, ScriptInfo] = {
    Script.TAMIL_BRAHMI: ScriptInfo(
        "Tamil-Brahmi (Tamiḻi)", "தமிழி", "c. 5th–3rd c. BCE to c. 3rd c. CE",
        "The earliest attested Tamil writing: cave-bed inscriptions (Mangulam, Jambai, Pugalur), "
        "potsherd graffiti (Keezhadi, Kodumanal, Arikamedu), coins and seals. Adds letters for "
        "ḻ, ḷ, ṟ, ṉ to Brahmi and, in its later stage, the puḷḷi (virama dot).",
        in_unicode=True),
    Script.VATTELUTTU: ScriptInfo(
        "Vatteluttu", "வட்டெழுத்து", "c. 5th–12th c. CE (into the 19th c. in Kerala)",
        "Rounded descendant of Tamil-Brahmi used in Pandya and Chera country: hero stones, "
        "temple and copper-plate grants. Not yet encoded in Unicode; read into modern Tamil.",
        in_unicode=False),
    Script.GRANTHA: ScriptInfo(
        "Grantha", "கிரந்தம்", "c. 6th c. CE to present",
        "Script used in the Tamil country to write Sanskrit — Pallava and Chola copper plates, "
        "temple inscriptions, palm-leaf manuscripts. Sibling of the Tamil script; source of "
        "ஜ ஷ ஸ ஹ க்ஷ ஸ்ரீ.",
        in_unicode=True),
    Script.GRANTHA_TAMIL: ScriptInfo(
        "Grantha–Tamil (maṇipravāḷam)", "கிரந்தத் தமிழ் (மணிப்பிரவாளம்)", "c. 7th c. CE to early 20th c.",
        "Tamil with its Sanskrit words in Grantha letters, often switching script inside a word "
        "(a Sanskrit stem in Grantha, a Tamil suffix in Tamil letters): Śrīvaiṣṇava and Śaiva "
        "commentaries, astrological and medical manuscripts, colophons, 19th-century Grantha–Tamil "
        "print, and the svasti śrī openings of Tamil inscriptions.",
        in_unicode=True),
    Script.TAMIL_MEDIEVAL: ScriptInfo(
        "Medieval Tamil script", "இடைக்காலத் தமிழ் எழுத்து", "c. 7th–16th c. CE",
        "The Tamil script of Pallava, Chola, Pandya and Vijayanagara inscriptions and copper "
        "plates — same letters as today, very different letterforms.",
        in_unicode=True),
    Script.TAMIL_PRE_REFORM: ScriptInfo(
        "Pre-reform Tamil", "சீர்திருத்தத்திற்கு முந்தைய தமிழ்", "c. 16th c. to 1978",
        "Palm-leaf manuscripts and print before the 1978 script reform: old ligatures for "
        "ணா றா னா and ணை லை ளை னை; manuscripts often omit the puḷḷi and do not distinguish "
        "e/ē, o/ō.",
        in_unicode=True),
    Script.TAMIL_MODERN: ScriptInfo(
        "Modern Tamil", "தற்காலத் தமிழ்", "1978–present",
        "The reformed Tamil script of contemporary print, signage, handwriting and screens.",
        in_unicode=True),
    Script.MALAYALAM: ScriptInfo("Malayalam", "மலையாளம்", "modern", "Distractor script.", True, False),
    Script.KANNADA: ScriptInfo("Kannada", "கன்னடம்", "modern", "Distractor script.", True, False),
    Script.TELUGU: ScriptInfo("Telugu", "தெலுங்கு", "modern", "Distractor script.", True, False),
    Script.SINHALA: ScriptInfo("Sinhala", "சிங்களம்", "modern", "Distractor script.", True, False),
    Script.LATIN: ScriptInfo("Latin", "இலத்தீன்", "modern", "Distractor script.", True, False),
}


@dataclass(frozen=True)
class MediumInfo:
    name: str
    tamil: str
    description: str


MEDIA: dict[Medium, MediumInfo] = {
    Medium.BORN_DIGITAL: MediumInfo("Born-digital", "எண்ணிம உரை",
                                    "Text rendered directly from Unicode — the clean upper bound."),
    Medium.SCREEN: MediumInfo("Screen", "திரை",
                              "Screenshots of web pages, apps, chats, subtitles and tables."),
    Medium.PRINT_SCAN: MediumInfo("Printed paper (scan)", "அச்சு (வருடல்)",
                                  "Flatbed scans of printed pages."),
    Medium.PRINT_PHOTO: MediumInfo("Printed paper (photo)", "அச்சு (புகைப்படம்)",
                                   "Phone photographs of printed pages: perspective, curl, shadows."),
    Medium.HANDWRITTEN: MediumInfo("Handwritten paper", "கையெழுத்து",
                                   "Pen or pencil on paper."),
    Medium.SCENE: MediumInfo("Scene text", "காட்சி உரை",
                             "Shop boards, road signs, bus boards, wall paintings, banners."),
    Medium.PALM_LEAF: MediumInfo("Palm leaf", "ஓலைச்சுவடி",
                                 "Stylus-incised, lamp-black-filled palm-leaf manuscripts."),
    Medium.STONE: MediumInfo("Stone", "கல்வெட்டு",
                             "Rock-cut and temple inscriptions, plaques and foundation stones."),
    Medium.ESTAMPAGE: MediumInfo("Estampage", "மைப்படி",
                                 "Ink rubbings / impressions taken from inscriptions."),
    Medium.COPPER_PLATE: MediumInfo("Copper plate / metal", "செப்பேடு",
                                    "Engraved copper-plate grants and metal plaques."),
    Medium.POTTERY: MediumInfo("Pottery", "மட்பாண்ட எழுத்து",
                               "Graffiti scratched on potsherds, coins and seals."),
}


@dataclass(frozen=True)
class TrackInfo:
    name: str
    tamil: str
    description: str


TRACKS: dict[Track, TrackInfo] = {
    Track.PRINT: TrackInfo("Print & Digital", "அச்சு & எண்ணிமம்",
                           "Born-digital, scanned and photographed print; numerals and symbols."),
    Track.SCREEN: TrackInfo("Screen Text", "திரை உரை", "Screenshots of web, apps, chats and video."),
    Track.HANDWRITING: TrackInfo("Handwriting (HTR)", "கையெழுத்து", "Handwritten text on paper."),
    Track.SCENE: TrackInfo("Scene Text", "காட்சி உரை", "Signboards and text in the wild."),
    Track.MANUSCRIPTS: TrackInfo("Manuscripts & Old Print", "சுவடிகள் & பழைய அச்சு",
                                 "Palm-leaf manuscripts, Grantha–Tamil writing and pre-1978 print."),
    Track.EPIGRAPHY: TrackInfo("Epigraphy", "கல்வெட்டியல்",
                               "Stone, copper plates, pottery; Tamil-Brahmi and Grantha."),
    Track.CLASSIFICATION: TrackInfo("Script & Medium ID", "எழுத்து & ஊடக அடையாளம்",
                                    "Identify the script stage and the writing support."),
    Track.TRANSLATION: TrackInfo("Image → English", "படம் → ஆங்கிலம்",
                                 "Read Tamil in an image and translate it into English."),
    Track.DIAGNOSTICS: TrackInfo("Controls (not ranked)", "கட்டுப்பாட்டுச் சோதனைகள்",
                                 "Perturbed famous texts and blank or effaced surfaces: do systems read, "
                                 "recite or invent?"),
}

# Tracks with leaderboard columns, and the six that are about reading.
SCORED_TRACKS: list[Track] = [t for t in Track if t != Track.DIAGNOSTICS]
READING_TRACKS: list[Track] = [Track.PRINT, Track.SCREEN, Track.HANDWRITING, Track.SCENE,
                               Track.MANUSCRIPTS, Track.EPIGRAPHY]
TRACK_ORDER: list[Track] = SCORED_TRACKS


@dataclass(frozen=True)
class EraInfo:
    name: str
    tamil: str
    description: str


ERAS: dict[Era, EraInfo] = {
    Era.MODERN: EraInfo("Modern script", "தற்கால எழுத்து",
                        "The reformed Tamil script of today, on every medium: print, screens, handwriting, "
                        "signs and modern plaques."),
    Era.OLDER: EraInfo("Older scripts", "பழைய எழுத்துகள்",
                       "Tamil-Brahmi, Grantha, Grantha–Tamil and pre-reform Tamil on stone, copper, palm "
                       "leaves and old print. Each script stage counts equally."),
}

# Which half of the benchmark each script stage belongs to; distractors belong to neither.
SCRIPT_ERA: dict[Script, Era | None] = {
    Script.TAMIL_MODERN: Era.MODERN,
    Script.TAMIL_PRE_REFORM: Era.OLDER,
    Script.TAMIL_MEDIEVAL: Era.OLDER,
    Script.GRANTHA_TAMIL: Era.OLDER,
    Script.GRANTHA: Era.OLDER,
    Script.VATTELUTTU: Era.OLDER,
    Script.TAMIL_BRAHMI: Era.OLDER,
    Script.MALAYALAM: None, Script.KANNADA: None, Script.TELUGU: None, Script.SINHALA: None,
    Script.LATIN: None,
}

# The Tamil lineage, oldest first: the order of the per-script page and coverage table.
LINEAGE_ORDER: list[Script] = [Script.TAMIL_BRAHMI, Script.VATTELUTTU, Script.GRANTHA, Script.GRANTHA_TAMIL,
                               Script.TAMIL_MEDIEVAL, Script.TAMIL_PRE_REFORM, Script.TAMIL_MODERN]


def era_of(script) -> Era | None:
    """The era of a script stage (a ``Script`` or its value); ``None`` for distractors and
    for anything that is not a script, such as a medium label."""
    try:
        return SCRIPT_ERA.get(Script(script))
    except ValueError:
        return None


@dataclass
class LabelSet:
    """The closed label vocabulary for a classification subset, with the short
    descriptions shown to models in the prompt."""

    labels: list[str]
    descriptions: dict[str, str] = field(default_factory=dict)


SCRIPT_ID_LABELS = LabelSet(
    labels=[s.value for s in (
        Script.TAMIL_MODERN, Script.TAMIL_PRE_REFORM, Script.TAMIL_BRAHMI, Script.GRANTHA,
        Script.GRANTHA_TAMIL, Script.MALAYALAM, Script.KANNADA, Script.TELUGU, Script.SINHALA)],
    descriptions={
        Script.TAMIL_MODERN.value: "modern Tamil script (post-1978 reformed letterforms)",
        Script.TAMIL_PRE_REFORM.value: "Tamil script with pre-1978 letterforms (old ligatures for ணா றா னா ணை லை ளை னை)",
        Script.TAMIL_BRAHMI.value: "Tamil-Brahmi (Tamili), the ancient Brahmi-derived script of early Tamil inscriptions",
        Script.GRANTHA.value: "Grantha script (used in the Tamil country to write Sanskrit)",
        Script.GRANTHA_TAMIL.value: "Grantha–Tamil (maṇipravāḷam): Tamil letters with Sanskrit words in Grantha "
                                    "letters, often switching script inside a word",
        Script.MALAYALAM.value: "Malayalam script",
        Script.KANNADA.value: "Kannada script",
        Script.TELUGU.value: "Telugu script",
        Script.SINHALA.value: "Sinhala script",
    },
)

MEDIUM_ID_LABELS = LabelSet(
    labels=["screen", "printed-paper", "handwritten-paper", "scene-signage", "palm-leaf",
            "stone", "estampage", "copper-plate", "pottery"],
    descriptions={
        "screen": "a screenshot of a digital screen (web page, app, chat, video subtitle)",
        "printed-paper": "printed text on paper (scanned or photographed)",
        "handwritten-paper": "handwriting on paper",
        "scene-signage": "a signboard, banner, bus board or painted wall photographed in the wild",
        "palm-leaf": "a palm-leaf manuscript",
        "stone": "an inscription or plaque carved into stone",
        "estampage": "an ink rubbing (estampage) taken from an inscription",
        "copper-plate": "text engraved on a copper plate or metal plaque",
        "pottery": "graffiti scratched on a potsherd",
    },
)

# How each synthetic medium is labelled in the medium-identification task.
MEDIUM_TO_ID_LABEL: dict[Medium, str] = {
    Medium.BORN_DIGITAL: "screen",
    Medium.SCREEN: "screen",
    Medium.PRINT_SCAN: "printed-paper",
    Medium.PRINT_PHOTO: "printed-paper",
    Medium.HANDWRITTEN: "handwritten-paper",
    Medium.SCENE: "scene-signage",
    Medium.PALM_LEAF: "palm-leaf",
    Medium.STONE: "stone",
    Medium.ESTAMPAGE: "estampage",
    Medium.COPPER_PLATE: "copper-plate",
    Medium.POTTERY: "pottery",
}
