"""Vendored font registry (all SIL Open Font License; see assets/fonts/licenses)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

FONT_DIR = Path(os.environ.get("TAMILBENCH_FONT_DIR", Path(__file__).resolve().parents[3] / "assets" / "fonts"))


@dataclass(frozen=True)
class FontSpec:
    id: str
    file: str
    tags: frozenset[str]
    variation: str | None = None   # named instance for variable fonts ("Bold", "Light", …)

    @property
    def path(self) -> Path:
        return FONT_DIR / self.file


def _f(id_, file, tags, variation=None):
    return FontSpec(id_, file, frozenset(tags.split()), variation)


FONTS: list[FontSpec] = [
    # Modern Tamil — text faces
    _f("noto-sans", "NotoSansTamil-VF.ttf", "tamil sans text ui", "Regular"),
    _f("noto-sans-bold", "NotoSansTamil-VF.ttf", "tamil sans bold display", "Bold"),
    _f("noto-sans-cond", "NotoSansTamil-VF.ttf", "tamil sans text", "Condensed"),
    _f("noto-serif", "NotoSerifTamil-VF.ttf", "tamil serif text print", "Regular"),
    _f("noto-serif-bold", "NotoSerifTamil-VF.ttf", "tamil serif bold display print", "Bold"),
    _f("noto-sans-ui", "NotoSansTamilUI-VF.ttf", "tamil sans ui nolatin", "Regular"),
    _f("mukta", "MuktaMalar-Regular.ttf", "tamil sans text ui"),
    _f("mukta-bold", "MuktaMalar-Bold.ttf", "tamil sans bold display"),
    _f("mukta-light", "MuktaMalar-Light.ttf", "tamil sans text"),
    _f("hind", "HindMadurai-Regular.ttf", "tamil sans text ui"),
    _f("hind-bold", "HindMadurai-Bold.ttf", "tamil sans bold display"),
    _f("hind-light", "HindMadurai-Light.ttf", "tamil sans text"),
    _f("catamaran", "Catamaran-VF.ttf", "tamil sans text ui", "Regular"),
    _f("catamaran-black", "Catamaran-VF.ttf", "tamil sans bold display", "Black"),
    _f("anek", "AnekTamil-VF.ttf", "tamil sans text", "Regular"),
    _f("anek-bold", "AnekTamil-VF.ttf", "tamil sans bold display", "Bold"),
    _f("tiro", "TiroTamil-Regular.ttf", "tamil serif text print"),
    _f("tiro-italic", "TiroTamil-Italic.ttf", "tamil serif text print"),
    _f("meera-inimai", "MeeraInimai-Regular.ttf", "tamil sans text"),
    _f("pavanam", "Pavanam-Regular.ttf", "tamil sans text display"),
    _f("lohit", "Lohit-Tamil.ttf", "tamil sans text nolatin print"),
    # Display & handwriting-like
    _f("baloo", "BalooThambi2-VF.ttf", "tamil display rounded", "Regular"),
    _f("baloo-bold", "BalooThambi2-VF.ttf", "tamil display rounded bold", "ExtraBold"),
    _f("arima", "Arima-VF.ttf", "tamil display hand", "Regular"),
    _f("arima-bold", "Arima-VF.ttf", "tamil display hand bold", "Bold"),
    _f("coiny", "Coiny-Regular.ttf", "tamil display rounded hand"),
    _f("kavivanar", "Kavivanar-Regular.ttf", "tamil hand"),
    # Historical
    _f("lohit-classical", "Lohit-Tamil-Classical.ttf", "tamil classical old-forms nolatin"),
    _f("brahmi", "NotoSansBrahmi-Regular.ttf", "brahmi"),
    _f("grantha-sans", "NotoSansGrantha-Regular.ttf", "grantha"),
    _f("grantha-serif", "NotoSerifGrantha-Regular.ttf", "grantha"),
    _f("tamil-supplement", "NotoSansTamilSupplement-Regular.ttf", "supplement"),
    # Distractor scripts
    _f("malayalam-sans", "NotoSansMalayalam-Regular.ttf", "malayalam"),
    _f("malayalam-serif", "NotoSerifMalayalam-Regular.ttf", "malayalam"),
    _f("kannada-sans", "NotoSansKannada-Regular.ttf", "kannada"),
    _f("kannada-serif", "NotoSerifKannada-Regular.ttf", "kannada"),
    _f("telugu-sans", "NotoSansTelugu-Regular.ttf", "telugu"),
    _f("telugu-serif", "NotoSerifTelugu-Regular.ttf", "telugu"),
    _f("sinhala-sans", "NotoSansSinhala-Regular.ttf", "sinhala"),
    _f("sinhala-serif", "NotoSerifSinhala-Regular.ttf", "sinhala"),
]
BY_ID = {f.id: f for f in FONTS}
LATIN_FALLBACK = BY_ID["noto-sans"]
SUPPLEMENT = BY_ID["tamil-supplement"]


def grantha_for(font: FontSpec) -> FontSpec:
    """The Grantha face that matches ``font`` (serif with serif), for Grantha–Tamil mixed text."""
    return BY_ID["grantha-serif" if "serif" in font.tags else "grantha-sans"]


def select(*tags: str, exclude: tuple[str, ...] = ()) -> list[FontSpec]:
    """Fonts that carry all of ``tags`` and none of ``exclude``."""
    return [f for f in FONTS if set(tags) <= f.tags and not (set(exclude) & f.tags)]


@lru_cache(maxsize=None)
def cmap(file: str) -> frozenset[int]:
    from fontTools.ttLib import TTFont
    return frozenset(TTFont(str(FONT_DIR / file), lazy=True).getBestCmap())


def covers(font: FontSpec, text: str) -> bool:
    cm = cmap(font.file)
    return all(ord(c) in cm or c.isspace() or ord(c) in (0x200C, 0x200D) for c in text)


@lru_cache(maxsize=512)
def load(font_id: str, size: int) -> ImageFont.FreeTypeFont:
    spec = BY_ID[font_id]
    font = ImageFont.truetype(str(spec.path), size, layout_engine=ImageFont.Layout.RAQM)
    if spec.variation:
        try:
            names = [n.decode() if isinstance(n, bytes) else n for n in font.get_variation_names()]
            if spec.variation in names:
                font.set_variation_by_name(spec.variation)
            elif spec.variation == "Condensed":
                axes = font.get_variation_axes()
                font.set_variation_by_axes([a["default"] if a.get("name") not in (b"Width", "Width")
                                            else max(a["minimum"], 75) for a in axes])
        except OSError:
            pass
    return font
