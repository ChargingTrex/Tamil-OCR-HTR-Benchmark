"""The benchmark's subsets: one place that says what each subset is, which track it
counts towards, which prompt models see, and how it is scored.

Two subsets are *controls* (track ``diagnostics``, never ranked): their items imitate items
of other subsets — same prompt, same reference field, same scoring policy — named per item
by ``as_subset`` in the manifest, so a system cannot tell a control from a test item."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .taxonomy import (MEDIUM_ID_LABELS, READING_TRACKS, SCORED_TRACKS, SCRIPT_ID_LABELS, Granularity,
                       Medium, Provenance, Script, Task, Track)

DEFAULT = {"spaces": "collapse"}
EPIGRAPHIC = {"spaces": "remove", "fold_pulli": True, "fold_vowel_length": True}


@dataclass(frozen=True)
class SubsetSpec:
    id: str
    title: str
    task: Task
    track: Track
    prompt: str
    description: str
    scripts: tuple[Script, ...]
    media: tuple[Medium, ...]
    provenance: Provenance = Provenance.SYNTHETIC
    granularity: tuple[Granularity, ...] = (Granularity.LINE,)
    count: int = 100
    policy: dict = field(default_factory=lambda: dict(DEFAULT))
    labels: tuple[str, ...] = ()
    target: str = "text"          # manifest field holding the reference
    license: str = "CC-BY-4.0"
    attribution: str = ""
    scoring: str = "default"      # "default", or "abstention" (blank controls: any reading is an error)

    @property
    def ranked(self) -> bool:
        return self.track != Track.DIAGNOSTICS

    def to_json(self) -> dict:
        d = asdict(self)
        for k in ("task", "track", "provenance"):
            d[k] = getattr(self, k).value
        d["scripts"] = [s.value for s in self.scripts]
        d["media"] = [m.value for m in self.media]
        d["granularity"] = [g.value for g in self.granularity]
        d["labels"] = list(self.labels)
        d["ranked"] = self.ranked
        return d


R, T = Task.RECOGNITION, Track

SUBSETS: list[SubsetSpec] = [
    # ---------------------------------------------------------------- Print & Digital
    SubsetSpec("print-digital", "Born-digital text", R, T.PRINT, "recognition",
               "Clean Unicode text rendered in a dozen Tamil typefaces and weights — words, lines and paragraphs. "
               "The upper bound: failures here are failures of script knowledge, not of vision.",
               (Script.TAMIL_MODERN,), (Medium.BORN_DIGITAL,),
               granularity=(Granularity.WORD, Granularity.LINE, Granularity.BLOCK)),
    SubsetSpec("print-scan", "Scanned print", R, T.PRINT, "recognition",
               "Book and newspaper text as a flatbed scanner sees it: paper texture, ink spread, "
               "skew, speckle, bleed-through, JPEG.",
               (Script.TAMIL_MODERN,), (Medium.PRINT_SCAN,),
               granularity=(Granularity.LINE, Granularity.BLOCK)),
    SubsetSpec("print-photo", "Photographed print", R, T.PRINT, "recognition",
               "Pages photographed with a phone: perspective, page curl, shadows, uneven light, blur.",
               (Script.TAMIL_MODERN,), (Medium.PRINT_PHOTO,),
               granularity=(Granularity.LINE, Granularity.BLOCK)),
    SubsetSpec("numerals-symbols", "Tamil numerals & signs", R, T.PRINT, "recognition",
               "Ledger, deed and horoscope lines using Tamil numerals (positional and ௰ ௱ ௲), "
               "the day/month/year/debit/credit/rupee signs and Tamil Supplement fractions "
               "and measures (U+11FC0–U+11FFF).",
               (Script.TAMIL_MODERN,), (Medium.PRINT_SCAN, Medium.BORN_DIGITAL), count=80),
    # ---------------------------------------------------------------- Screen
    SubsetSpec("screen", "Screenshots", R, T.SCREEN, "recognition",
               "Real browser renderings (Chromium) of news pages, chat bubbles, app settings, "
               "forms, tables and video subtitles — light and dark themes, phone and desktop "
               "widths, downscaled and re-compressed like forwarded screenshots.",
               (Script.TAMIL_MODERN,), (Medium.SCREEN,), count=120,
               granularity=(Granularity.LINE, Granularity.BLOCK)),
    # ---------------------------------------------------------------- Handwriting
    SubsetSpec("handwriting", "Handwriting (synthetic)", R, T.HANDWRITING, "recognition",
               "Letters, notes and lists in handwriting-style typefaces with per-word elastic "
               "distortion, slant, baseline drift, pen-pressure variation and ruled paper. A proxy "
               "until a licensed real-HTR set is added — see docs/real-data.md.",
               (Script.TAMIL_MODERN,), (Medium.HANDWRITTEN,),
               granularity=(Granularity.LINE, Granularity.BLOCK)),
    # ---------------------------------------------------------------- Scene
    SubsetSpec("scene", "Scene text", R, T.SCENE, "recognition",
               "Shop boards, road and bus signs, LED destination boards, painted walls and flex "
               "banners, photographed in perspective under uneven light.",
               (Script.TAMIL_MODERN,), (Medium.SCENE,),
               granularity=(Granularity.WORD, Granularity.LINE)),
    # ---------------------------------------------------------------- Manuscripts & old print
    SubsetSpec("pre-reform-print", "Pre-1978 print", R, T.MANUSCRIPTS, "recognition",
               "Letterpress pages in pre-reform orthography (old ligatures for ணா றா னா ணை லை "
               "ளை னை) on aged, foxed paper. Every item contains at least one reform syllable.",
               (Script.TAMIL_PRE_REFORM,), (Medium.PRINT_SCAN,),
               granularity=(Granularity.LINE, Granularity.BLOCK)),
    SubsetSpec("palm-leaf-synth", "Palm leaf (synthetic)", R, T.MANUSCRIPTS, "recognition-palm-leaf",
               "Simulated ōlaiccuvaṭi: stylus-incised, lamp-black-filled lines on fibrous leaf with "
               "string holes and edge damage; scriptio continua, puḷḷi often omitted, e/ē and o/ō "
               "merged, pre-reform letterforms.",
               (Script.TAMIL_PRE_REFORM,), (Medium.PALM_LEAF,), count=100,
               granularity=(Granularity.BLOCK,), policy=dict(EPIGRAPHIC)),
    SubsetSpec("palm-leaf-cict", "Palm leaf (real, CICT Tirukkural)", R, T.MANUSCRIPTS,
               "recognition-palm-leaf-cict",
               "Real leaves of a Tirukkural palm-leaf manuscript with expert diplomatic "
               "transcriptions (10 lines per leaf), from the CICT Tirukkural Ground Truth Corpus. "
               "Uses only leaves from the source's held-out val+test splits.",
               (Script.TAMIL_PRE_REFORM,), (Medium.PALM_LEAF,), provenance=Provenance.REAL,
               count=26, granularity=(Granularity.PAGE,),
               policy=dict(EPIGRAPHIC, strip_tamil_numerals=True),
               attribution="Central Institute of Classical Tamil (CICT), CICT Tirukkural Ground "
                           "Truth Corpus, CC BY 4.0; per-leaf DOIs in the manifest."),
    SubsetSpec("grantha-tamil", "Grantha–Tamil (maṇipravāḷam)", R, T.MANUSCRIPTS, "recognition-grantha-tamil",
               "Tamil manuscripts and old print that write Sanskrit words in Grantha inside Tamil text, "
               "often switching script mid-word (commentaries, astrology, medicine, colophons, "
               "inscription formulae). Transcribed with Tamil in Tamil Unicode and Grantha in IAST.",
               (Script.GRANTHA_TAMIL,), (Medium.PALM_LEAF, Medium.PRINT_SCAN), count=100,
               granularity=(Granularity.LINE, Granularity.BLOCK), target="text",
               policy={"spaces": "remove", "script": "iast", "fold_pulli": True, "fold_vowel_length": True}),
    # ---------------------------------------------------------------- Epigraphy
    SubsetSpec("stone", "Stone inscriptions & plaques", R, T.EPIGRAPHY, "recognition-epigraphic",
               "Engraved granite and marble: modern foundation stones, donor and memorial plaques, "
               "and medieval-style temple inscriptions (scriptio continua, Chola-style formulae) "
               "with weathering, lichen and raking light.",
               (Script.TAMIL_MODERN, Script.TAMIL_PRE_REFORM), (Medium.STONE,), count=100,
               granularity=(Granularity.LINE, Granularity.BLOCK), policy=dict(EPIGRAPHIC)),
    SubsetSpec("copper-plate", "Copper plates & metal", R, T.EPIGRAPHY, "recognition-epigraphic",
               "Engraved copper-plate grants with ring holes and patina, and brass donor plaques.",
               (Script.TAMIL_PRE_REFORM, Script.TAMIL_MODERN), (Medium.COPPER_PLATE,), count=80,
               granularity=(Granularity.BLOCK,), policy=dict(EPIGRAPHIC)),
    SubsetSpec("tamil-brahmi", "Tamil-Brahmi", R, T.EPIGRAPHY, "recognition-tamil-brahmi",
               "Tamil-Brahmi in its three orthographic stages (TB-I, TB-II, TB-III) on cave rock, "
               "as estampages and as potsherd graffiti, read into modern Tamil.",
               (Script.TAMIL_BRAHMI,), (Medium.STONE, Medium.ESTAMPAGE, Medium.POTTERY), count=120,
               granularity=(Granularity.WORD, Granularity.LINE), policy=dict(EPIGRAPHIC)),
    SubsetSpec("grantha", "Grantha", R, T.EPIGRAPHY, "recognition-grantha",
               "Sanskrit in Grantha — as on Pallava/Chola copper plates, stone and palm leaves — "
               "transliterated to IAST.",
               (Script.GRANTHA,), (Medium.COPPER_PLATE, Medium.STONE, Medium.PALM_LEAF, Medium.PRINT_SCAN),
               count=100, granularity=(Granularity.LINE, Granularity.BLOCK), target="iast",
               policy={"spaces": "remove", "script": "iast"}),
    # ---------------------------------------------------------------- Classification
    SubsetSpec("script-id", "Script identification", Task.SCRIPT_ID, T.CLASSIFICATION, "script-id",
               "Which script is this? Modern vs pre-reform Tamil, Tamil-Brahmi, Grantha, Grantha–Tamil, "
               "and the neighbouring South Indian and Sri Lankan scripts (Malayalam, Kannada, Telugu, Sinhala).",
               tuple(Script(s) for s in SCRIPT_ID_LABELS.labels), tuple(Medium), count=162,
               labels=tuple(SCRIPT_ID_LABELS.labels), target="label"),
    SubsetSpec("medium-id", "Medium identification", Task.MEDIUM_ID, T.CLASSIFICATION, "medium-id",
               "What is the text written on? Screen, paper (print / handwritten), signage, palm "
               "leaf, stone, estampage, copper plate or pottery.",
               (Script.TAMIL_MODERN, Script.TAMIL_PRE_REFORM, Script.TAMIL_BRAHMI), tuple(Medium),
               count=144, labels=tuple(MEDIUM_ID_LABELS.labels), target="label"),
    # ---------------------------------------------------------------- Translation
    SubsetSpec("translate-en", "Image → English", Task.TRANSLATION, T.TRANSLATION, "translation",
               "Read Tamil sentences and signs from print, screens, signboards and handwriting and "
               "translate them into English; scored with chrF++ against a reference translation.",
               (Script.TAMIL_MODERN,), (Medium.PRINT_SCAN, Medium.SCREEN, Medium.SCENE, Medium.HANDWRITTEN),
               count=100, target="english"),
    # ---------------------------------------------------------------- Controls (not ranked)
    SubsetSpec("perturbed", "Perturbed famous texts", R, T.DIAGNOSTICS, "recognition",
               "Famous texts with a few letters deliberately changed — Tirukkural couplets in modern print, "
               "on palm leaves, in pre-reform print and in Tamil-Brahmi, and well-known Sanskrit verses in "
               "Grantha. The reference is what the image shows; the prior-pull index measures how far "
               "answers drift back to the text as usually written. Half modern script, half older scripts.",
               (Script.TAMIL_MODERN, Script.TAMIL_PRE_REFORM, Script.GRANTHA, Script.TAMIL_BRAHMI),
               (Medium.PRINT_SCAN, Medium.PRINT_PHOTO, Medium.BORN_DIGITAL, Medium.HANDWRITTEN, Medium.PALM_LEAF,
                Medium.STONE, Medium.ESTAMPAGE, Medium.POTTERY, Medium.COPPER_PLATE),
               count=96, granularity=(Granularity.LINE, Granularity.BLOCK)),
    SubsetSpec("blank-controls", "Blank & effaced surfaces", R, T.DIAGNOSTICS, "recognition",
               "Paper, signboards, palm leaves, stone, copper and potsherds with no legible text: either "
               "blank, or with writing worn away beyond reading. The correct answer is no text; any "
               "letters count as hallucination. Half modern media, half older.",
               (Script.TAMIL_MODERN, Script.TAMIL_PRE_REFORM, Script.TAMIL_BRAHMI),
               (Medium.PRINT_SCAN, Medium.PRINT_PHOTO, Medium.HANDWRITTEN, Medium.SCENE, Medium.PALM_LEAF,
                Medium.STONE, Medium.COPPER_PLATE, Medium.ESTAMPAGE, Medium.POTTERY),
               count=80, granularity=(Granularity.BLOCK,), scoring="abstention"),
]

BY_ID: dict[str, SubsetSpec] = {s.id: s for s in SUBSETS}
TRACK_SUBSETS: dict[Track, list[str]] = {t: [s.id for s in SUBSETS if s.track == t] for t in Track}
RECOGNITION_TRACKS = READING_TRACKS
RANKED_SUBSETS: list[str] = [s.id for s in SUBSETS if s.ranked]
CONTROL_SUBSETS: list[str] = [s.id for s in SUBSETS if not s.ranked]
__all__ = ["SUBSETS", "BY_ID", "TRACK_SUBSETS", "SCORED_TRACKS", "READING_TRACKS", "RECOGNITION_TRACKS",
           "RANKED_SUBSETS", "CONTROL_SUBSETS", "SubsetSpec", "get", "presentation"]


def get(subset_id: str) -> SubsetSpec:
    try:
        return BY_ID[subset_id]
    except KeyError:
        raise KeyError(f"unknown subset {subset_id!r}; known: {', '.join(BY_ID)}") from None


def presentation(row: dict) -> SubsetSpec:
    """The subset whose prompt, reference field and scoring policy apply to a manifest row:
    its own, or — for a control item — the subset it imitates (``as_subset``)."""
    return get(row.get("as_subset") or row["subset"])
