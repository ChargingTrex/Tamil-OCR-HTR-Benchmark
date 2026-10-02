"""Text sources for the synthetic tiers.

All text in ``corpus/data`` was written for this benchmark (modern sentences, signage,
plaques, manuscript- and inscription-style lines) or is public-domain classical Tamil and
Sanskrit. The inscription-style lines are *authored in the style of* donative formulae
and royal eulogies; they are not transcriptions of specific inscriptions. Because every
synthetic image is rendered *from* this text, the ground truth matches the image by
construction regardless of any variant reading.
"""

from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources

from ..text import brahmi
from ..text.grantha import iast_to_grantha
from ..text import manipravalam as mp
from ..text.pseudo import pseudo_line

_SPLIT = " / "


@dataclass(frozen=True)
class Item:
    id: str
    text: str                     # modern-Tamil text (IAST for Sanskrit); lines joined by "\n"
    source: str                   # pool name, e.g. "modern", "classical", "historical"
    tag: str = ""                 # domain / register / kind
    english: str | None = None    # reference translation when available
    meta: dict = field(default_factory=dict, hash=False, compare=False)

    @property
    def lines(self) -> list[str]:
        return self.text.split("\n")


def _rows(name: str) -> list[dict]:
    with resources.files(__package__).joinpath("data", name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def _lines(text: str) -> str:
    return "\n".join(part.strip() for part in text.split(_SPLIT))


@lru_cache(maxsize=None)
def modern() -> tuple[Item, ...]:
    return tuple(Item(r["id"], r["tamil"].strip(), "modern", r["domain"], r["english"].strip())
                 for r in _rows("modern.tsv"))


@lru_cache(maxsize=None)
def classical() -> tuple[Item, ...]:
    return tuple(Item(r["id"], _lines(r["tamil"]), "classical", r["source"])
                 for r in _rows("classical.tsv"))


@lru_cache(maxsize=None)
def historical() -> tuple[Item, ...]:
    return tuple(Item(r["id"], r["tamil"].strip(), "historical", r["register"])
                 for r in _rows("historical.tsv"))


@lru_cache(maxsize=None)
def signage() -> tuple[Item, ...]:
    return tuple(Item(r["id"], r["tamil"].strip(), "signage", r["kind"], r["english"].strip())
                 for r in _rows("signage.tsv"))


@lru_cache(maxsize=None)
def plaques() -> tuple[Item, ...]:
    return tuple(Item(r["id"], r["tamil"].strip(), "plaques", r["kind"]) for r in _rows("plaques.tsv"))


@lru_cache(maxsize=None)
def names() -> tuple[Item, ...]:
    return tuple(Item(r["id"], r["tamil"].strip(), "names", r["kind"]) for r in _rows("names.tsv"))


@lru_cache(maxsize=None)
def ui_strings() -> tuple[Item, ...]:
    with resources.files(__package__).joinpath("data", "ui.txt").open(encoding="utf-8") as f:
        return tuple(Item(f"u{i + 1:03d}", ln.strip(), "ui")
                     for i, ln in enumerate(f) if ln.strip())


@lru_cache(maxsize=None)
def sanskrit() -> tuple[Item, ...]:
    """Sanskrit in IAST (the answer key), with the Grantha rendering precomputed in ``meta``."""
    out = []
    for r in _rows("sanskrit.tsv"):
        iast = _lines(r["iast"])
        out.append(Item(r["id"], iast, "sanskrit", r["source"], meta={"grantha": iast_to_grantha(iast)}))
    return tuple(out)



@lru_cache(maxsize=None)
def manipravalam() -> tuple[Item, ...]:
    """Grantha–Tamil mixed lines. ``text`` is the reference (Tamil in Tamil, Grantha in IAST);
    ``meta["markup"]`` keeps the segment markup ({IAST} = Grantha)."""
    out = []
    for r in _rows("manipravalam.tsv"):
        words = mp.parse(r["text"])
        out.append(Item(r["id"], " ".join(mp.reference(w) for w in words), "manipravalam", r["register"],
                        meta={"markup": r["text"].strip()}))
    return tuple(out)

@lru_cache(maxsize=None)
def tirukkural() -> dict[int, str]:
    """The standard text of the 1,330 couplets (public domain), keyed by number, the two
    lines joined by a space and the closing full stop dropped. Used as the "edition" text
    against which readings of real Tirukkural manuscripts are checked for recitation.
    Source: the ``thirukkural`` Python package (MIT, © Vaasudevan Srinivasan)."""
    return {int(r["kural"]): " ".join(part.strip() for part in r["tamil"].split(_SPLIT)).rstrip(".")
            for r in _rows("tirukkural.tsv")}


_TOKEN = re.compile(r"[஀-௿]+")


@lru_cache(maxsize=None)
def lexicon() -> tuple[str, ...]:
    """Every distinct Tamil word in the corpora (for random-word lines)."""
    words: set[str] = set()
    for pool in (modern(), classical(), historical(), signage(), plaques(), names(), ui_strings()):
        for it in pool:
            words.update(w for w in _TOKEN.findall(it.text) if len(w) >= 2)
    return tuple(sorted(words))


def random_word_line(rng: random.Random, n_words: tuple[int, int] = (2, 6)) -> str:
    lex = lexicon()
    return " ".join(rng.choice(lex) for _ in range(rng.randint(*n_words)))


def nonce_line(rng: random.Random, n_words: tuple[int, int] = (2, 6)) -> str:
    return pseudo_line(rng, n_words, avoid=set(lexicon()))


def brahmi_pool() -> list[Item]:
    """Lines that can be written in Tamil-Brahmi (no digits, āytam or punctuation)."""
    cands = [it for it in historical() if it.tag in ("tamil-brahmi-style", "pottery-name")]
    for it in classical():
        for ln in it.lines:
            cands.append(Item(it.id, ln, "classical", it.tag))
    out = []
    for it in cands:
        t = re.sub(r"[^஀-௿ ]", "", it.text).strip()
        t = re.sub(r"\s+", " ", t)
        if t and brahmi.can_transliterate(t) and not any(c in t for c in "ஜஷஸஹஶ"):
            out.append(Item(it.id, t, it.source, it.tag))
    return out


def all_pools() -> dict[str, tuple[Item, ...]]:
    return {"modern": modern(), "classical": classical(), "historical": historical(),
            "signage": signage(), "plaques": plaques(), "names": names(),
            "ui": ui_strings(), "sanskrit": sanskrit(), "manipravalam": manipravalam()}
