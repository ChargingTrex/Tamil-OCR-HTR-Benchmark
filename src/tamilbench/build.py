"""Build the benchmark: render every synthetic subset, import the real ones, write manifests.

Every sample is a pure function of (seed, subset, index), so the public test set can be
re-rendered exactly, and a *private* test set with the same distribution can be produced
by changing the seed (``tamilbench build --split private --seed …``) — the defence against
benchmark contamination that a procedurally generated benchmark gets for free.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import BENCHMARK_VERSION, corpus
from . import subsets as S
from .render import fonts as F
from .render import media as M
from .taxonomy import Medium, Script, Task
from .text import brahmi, indic, numerals
from .text import manipravalam as mp
from .text.grantha import iast_to_grantha
from .text.tamil import (REFORM_SYLLABLES, contains_reform_syllable, drop_pulli, merge_long_e_o,
                         scriptio_continua)

DEFAULT_SEED = 20261002
LITE_PER_SUBSET = 20

# ----------------------------------------------------------------------------- helpers


def wrap(text: str, width: int) -> list[str]:
    """Greedy word wrap by code points."""
    lines, cur = [], ""
    for w in text.split():
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines or [text]


def chunk_letters(text: str, width: int) -> list[str]:
    """Split space-less text into lines of ~width code points at letter boundaries."""
    from .text.tamil import letters
    out, cur = [], ""
    for lt in letters(text):
        if len(cur) + len(lt) > width and cur:
            out.append(cur)
            cur = ""
        cur += lt
    if cur:
        out.append(cur)
    return out


@dataclass
class Sample:
    """What a generator returns; the builder adds ids, paths and checksums."""

    image: object                    # PIL.Image
    text: str | None = None
    fmt: str = "jpg"
    fields: dict = field(default_factory=dict)   # label / english / iast / text_diplomatic / …
    script: str = Script.TAMIL_MODERN.value
    medium: str = Medium.BORN_DIGITAL.value
    granularity: str = "line"
    lexical: str = "corpus"
    text_source: str = ""
    render: dict = field(default_factory=dict)


def _pick(rng: random.Random, seq):
    return seq[rng.randrange(len(seq))]


def _tamil_text(pr: random.Random, *, pools=("modern",), corpus_p=0.7, lexicon_p=0.15,
                width=(26, 42), lines=(1, 3)) -> tuple[list[str], str, str]:
    """Pick Tamil text for a recognition sample: real sentences, random lexicon words or
    phonotactic nonce words. Returns (lines, lexical_kind, source)."""
    r = pr.random()
    if r < corpus_p:
        pool = [it for p in pools for it in corpus.all_pools()[p]]
        it = _pick(pr, pool)
        text = it.text.replace("\n", " ")
        src = f"{it.source}:{it.id}"
        kind = "corpus"
    elif r < corpus_p + lexicon_p:
        text, src, kind = corpus.random_word_line(pr, (3, 7)), "lexicon", "lexicon"
    else:
        text, src, kind = corpus.nonce_line(pr, (3, 7)), "nonce", "nonce"
    ls = wrap(text, pr.randint(*width))
    n = pr.randint(*lines)
    return ls[:max(1, n)], kind, src


def _np(pr: random.Random) -> np.random.Generator:
    return np.random.default_rng(pr.randrange(1 << 62))


# ----------------------------------------------------------------------------- generators

def gen_print_digital(pr: random.Random) -> Sample:
    g = pr.random()
    if g < 0.2:
        it = _pick(pr, corpus.ui_strings() + corpus.names() + corpus.signage())
        lines, kind, src, gran = [it.text], "corpus", f"{it.source}:{it.id}", "word"
    else:
        lines, kind, src = _tamil_text(pr, pools=("modern", "classical"), lines=(1, 1) if g < 0.7 else (2, 4))
        gran = "line" if len(lines) == 1 else "block"
    img, meta = M.born_digital(lines, _np(pr))
    return Sample(img, "\n".join(lines), "png", medium=Medium.BORN_DIGITAL.value, granularity=gran,
                  lexical=kind, text_source=src, render=meta)


def gen_print_scan(pr: random.Random) -> Sample:
    lines, kind, src = _tamil_text(pr, pools=("modern", "classical"), lines=(1, 4))
    img, meta = M.print_scan(lines, _np(pr))
    return Sample(img, "\n".join(lines), medium=Medium.PRINT_SCAN.value,
                  granularity="line" if len(lines) == 1 else "block", lexical=kind, text_source=src, render=meta)


def gen_print_photo(pr: random.Random) -> Sample:
    lines, kind, src = _tamil_text(pr, pools=("modern", "classical"), lines=(1, 4))
    img, meta = M.print_photo(lines, _np(pr))
    return Sample(img, "\n".join(lines), medium=Medium.PRINT_PHOTO.value,
                  granularity="line" if len(lines) == 1 else "block", lexical=kind, text_source=src, render=meta)


def gen_numerals(pr: random.Random) -> Sample:
    n = pr.randint(1, 3)
    items = [numerals.random_numeral_line(pr) for _ in range(n)]
    lines = [t for t, _ in items]
    rng = _np(pr)
    if pr.random() < 0.5:
        img, meta = M.print_scan(lines, rng, font=F.BY_ID[_pick(pr, ["noto-sans", "noto-serif", "tiro", "hind"])])
        medium = Medium.PRINT_SCAN.value
    else:
        img, meta = M.born_digital(lines, rng, font=F.BY_ID[_pick(pr, ["noto-sans", "noto-serif", "mukta"])])
        medium = Medium.BORN_DIGITAL.value
    return Sample(img, "\n".join(lines), "png" if medium == "born-digital" else "jpg", medium=medium,
                  granularity="line" if n == 1 else "block", lexical="numerals",
                  text_source="numerals:" + ",".join(k for _, k in items), render=meta)


def gen_handwriting(pr: random.Random) -> Sample:
    lines, kind, src = _tamil_text(pr, pools=("modern",), lines=(1, 3), width=(18, 32))
    img, meta = M.handwriting(lines, _np(pr))
    return Sample(img, "\n".join(lines), medium=Medium.HANDWRITTEN.value,
                  granularity="line" if len(lines) == 1 else "block", lexical=kind, text_source=src, render=meta)


_SIGN_BOARD = {"shop": ["shop", "shop", "banner", "wall"], "road": ["road"], "bus": ["led", "led", "road"],
               "office": ["office", "office", "wall"], "worship": ["office", "wall", "banner"],
               "notice": ["office", "wall"], "banner": ["banner", "banner", "shop"]}


def gen_scene(pr: random.Random) -> Sample:
    r = pr.random()
    if r < 0.75:
        it = _pick(pr, corpus.signage())
        text, kind, src, board = it.text, "corpus", f"signage:{it.id}", _pick(pr, _SIGN_BOARD[it.tag])
    elif r < 0.85:
        it = _pick(pr, [n for n in corpus.names() if n.tag == "place"])
        text, kind, src, board = it.text, "corpus", f"names:{it.id}", _pick(pr, ["led", "road"])
    else:
        text, kind, src, board = corpus.nonce_line(pr, (1, 3)), "nonce", "nonce", _pick(pr, ["shop", "banner", "led"])
    lines = wrap(text, 18) if len(text) > 22 and pr.random() < 0.7 else [text]
    img, meta = M.scene(lines, _np(pr), kind=board)
    return Sample(img, "\n".join(lines), medium=Medium.SCENE.value,
                  granularity="word" if len(text.split()) <= 2 else "line", lexical=kind, text_source=src, render=meta)


def _reform_text(pr: random.Random) -> tuple[str, str, str]:
    for _ in range(200):
        r = pr.random()
        if r < 0.75:
            it = _pick(pr, corpus.modern() + corpus.classical())
            text, kind, src = it.text.replace("\n", " "), "corpus", f"{it.source}:{it.id}"
        else:
            text, kind, src = corpus.nonce_line(pr, (3, 6)), "nonce", "nonce"
            if not contains_reform_syllable(text):
                w = text.split()
                w[pr.randrange(len(w))] += _pick(pr, REFORM_SYLLABLES)
                text = " ".join(w)
        if contains_reform_syllable(text) and not re.search(r"[A-Za-z0-9₹%°]", text):
            return text, kind, src
    raise RuntimeError("no reform text found")


def gen_pre_reform(pr: random.Random) -> Sample:
    text, kind, src = _reform_text(pr)
    lines = wrap(text, pr.randint(24, 40))[:pr.randint(1, 3)]
    if not contains_reform_syllable(" ".join(lines)):
        lines = wrap(text, 200)
    img, meta = M.print_scan(lines, _np(pr), font=F.BY_ID["lohit-classical"], aged=pr.uniform(0.3, 0.9), letterpress=True)
    return Sample(img, "\n".join(lines), script=Script.TAMIL_PRE_REFORM.value, medium=Medium.PRINT_SCAN.value,
                  granularity="line" if len(lines) == 1 else "block", lexical=kind, text_source=src, render=meta)


def _manuscript_items(pr: random.Random, n_target: int) -> tuple[str, str, str]:
    """Concatenate classical / manuscript lines (or nonce words) to fill a leaf."""
    r = pr.random()
    parts, srcs = [], []
    if r < 0.85:
        pool = list(corpus.classical()) + [h for h in corpus.historical() if h.tag == "manuscript"]
        while sum(len(p) for p in parts) < n_target:
            it = _pick(pr, pool)
            parts.append(re.sub(r"[^஀-௿ ]", " ", it.text.replace("\n", " ")))
            srcs.append(f"{it.source}:{it.id}")
        kind = "corpus"
    else:
        while sum(len(p) for p in parts) < n_target:
            parts.append(corpus.nonce_line(pr, (4, 8)))
        srcs, kind = ["nonce"], "nonce"
    text = re.sub(r"\s+", " ", " ".join(parts)).strip()
    return text, kind, ",".join(srcs[:6])


def _diplomatic_fit(text: str, pr: random.Random, max_chars: int, *, p_drop: float,
                    merge: bool) -> tuple[str, str]:
    """Word-aligned (normalised, diplomatic) pair: drop the puḷḷi / merge e-o word by word and
    keep as many whole words as fit in ``max_chars`` diplomatic code points."""
    words, dws = text.split(), []
    for w in words:
        d = drop_pulli(w, p_drop, pr)
        dws.append(merge_long_e_o(d) if merge else d)
    k, total = 0, 0
    while k < len(words) and total + len(dws[k]) <= max_chars:
        total += len(dws[k])
        k += 1
    k = max(1, k)
    return " ".join(words[:k]), "".join(dws[:k])


def gen_palm_leaf(pr: random.Random) -> Sample:
    n_lines = pr.randint(3, 6)
    width = pr.randint(38, 56)
    text, kind, src = _manuscript_items(pr, n_lines * width + 40)
    p_drop = _pick(pr, [1.0, 1.0, 0.6])
    merge = pr.random() < 0.7
    ref, dip = _diplomatic_fit(text, pr, n_lines * width, p_drop=p_drop, merge=merge)
    lines = chunk_letters(dip, width)
    img, meta = M.palm_leaf(lines, _np(pr))
    meta.update({"pulli_dropped": p_drop, "e_o_merged": merge})
    return Sample(img, ref, script=Script.TAMIL_PRE_REFORM.value, medium=Medium.PALM_LEAF.value,
                  granularity="block", lexical=kind, text_source=src, render=meta,
                  fields={"text_diplomatic": "\n".join(lines)})


def gen_stone(pr: random.Random) -> Sample:
    rng = _np(pr)
    if pr.random() < 0.5:
        it = _pick(pr, corpus.plaques())
        lines = wrap(it.text, pr.randint(18, 30))[:3]
        style = _pick(pr, ["plaque-black", "plaque-grey"])
        img, meta = M.stone(lines, rng, style=style)
        return Sample(img, "\n".join(lines), medium=Medium.STONE.value, granularity="block",
                      text_source=f"plaques:{it.id}", render=meta)
    pool = [h for h in corpus.historical() if h.tag in ("chola-meykeerthi", "donation")]
    if pr.random() < 0.85:
        it = _pick(pr, pool)
        text, kind, src = it.text, "corpus", f"historical:{it.id}"
    else:
        text, kind, src = corpus.nonce_line(pr, (4, 7)), "nonce", "nonce"
    width = pr.randint(16, 26)
    ref, dip = _diplomatic_fit(text, pr, 3 * width, p_drop=_pick(pr, [1.0, 0.5]), merge=False)
    lines = chunk_letters(dip, width)
    img, meta = M.stone(lines, rng, style="inscription")
    return Sample(img, ref, script=Script.TAMIL_PRE_REFORM.value, medium=Medium.STONE.value, granularity="block",
                  lexical=kind, text_source=src, render=meta, fields={"text_diplomatic": "\n".join(lines)})


def gen_copper(pr: random.Random) -> Sample:
    rng = _np(pr)
    if pr.random() < 0.3:
        it = _pick(pr, [q for q in corpus.plaques() if q.tag in ("donor", "temple", "inauguration")])
        lines = wrap(it.text, pr.randint(18, 28))[:3]
        img, meta = M.copper_plate(lines, rng, style="plaque")
        return Sample(img, "\n".join(lines), medium=Medium.COPPER_PLATE.value, granularity="block",
                      text_source=f"plaques:{it.id}", render=meta)
    pool = [h for h in corpus.historical() if h.tag in ("chola-meykeerthi", "donation")]
    it1, it2 = _pick(pr, pool), _pick(pr, pool)
    text = f"{it1.text} {it2.text}" if pr.random() < 0.5 else it1.text
    width = pr.randint(20, 30)
    ref, dip = _diplomatic_fit(text, pr, 4 * width, p_drop=_pick(pr, [1.0, 0.5]), merge=False)
    lines = chunk_letters(dip, width)
    img, meta = M.copper_plate(lines, rng, style="grant")
    return Sample(img, ref, script=Script.TAMIL_PRE_REFORM.value, medium=Medium.COPPER_PLATE.value,
                  granularity="block", text_source=f"historical:{it1.id}", render=meta,
                  fields={"text_diplomatic": "\n".join(lines)})


def gen_tamil_brahmi(pr: random.Random) -> Sample:
    rng = _np(pr)
    surface = _pick(pr, ["cave", "cave", "estampage", "pottery"])
    if pr.random() < 0.15:
        text, kind, src = corpus.nonce_line(pr, (1, 3)), "nonce", "nonce"
        text = re.sub(r"[ஜஷஸஹஶஃ]", "", text)
        orth = brahmi.Orthography.TB3
    else:
        pool = corpus.brahmi_pool()
        if surface == "pottery":
            pool = [p for p in pool if p.tag == "pottery-name" or len(p.text) <= 12]
        it = _pick(pr, pool)
        text, kind, src = it.text, "corpus", f"{it.source}:{it.id}"
        orth = pr.choices(list(brahmi.Orthography), weights=[0.3, 0.2, 0.5])[0]
    if surface == "pottery" and len(text) > 14:
        text = " ".join(text.split()[:2])
    native = brahmi.to_brahmi(text, orth, lla_variant=pr.random() < 0.1)
    continuous = pr.random() < 0.8
    shown = native.replace(" ", "") if continuous else native
    lines = [shown] if len(shown) <= 14 or surface == "pottery" else chunk_letters(shown, pr.randint(10, 16))
    font = F.BY_ID["brahmi"]
    if surface == "cave":
        img, meta = M.stone(lines, rng, style="cave", font=font, spacing=1.5)
        medium = Medium.STONE.value
    elif surface == "estampage":
        img, meta = M.estampage(lines, rng, font=font, spacing=1.5)
        medium = Medium.ESTAMPAGE.value
    else:
        img, meta = M.pottery(lines, rng, font=font)
        medium = Medium.POTTERY.value
    meta.update({"orthography": orth.value, "scriptio_continua": continuous})
    return Sample(img, text, script=Script.TAMIL_BRAHMI.value, medium=medium,
                  granularity="word" if len(text.split()) == 1 else "line", lexical=kind, text_source=src,
                  render=meta, fields={"text_native": "\n".join(lines)})


_SKT_SYL = ["k", "t", "p", "n", "m", "r", "v", "s", "dh", "g", "y", "j", "ś", "h", "l", "d", "bh"]
_SKT_VOWEL = ["a", "a", "ā", "i", "ī", "u", "e", "o", "aṃ"]


def _sanskrit_nonce(pr: random.Random) -> str:
    words = []
    for _ in range(pr.randint(3, 5)):
        words.append("".join(_pick(pr, _SKT_SYL) + _pick(pr, _SKT_VOWEL) for _ in range(pr.randint(2, 4))))
    return " ".join(words)


def gen_grantha(pr: random.Random) -> Sample:
    rng = _np(pr)
    if pr.random() < 0.15:
        iast_lines, kind, src = [_sanskrit_nonce(pr)], "nonce", "nonce"
    else:
        it = _pick(pr, corpus.sanskrit())
        iast_lines, kind, src = it.text.split("\n"), "corpus", f"sanskrit:{it.id}"
        if len(iast_lines) > 2:
            k = pr.randrange(len(iast_lines) - 1)
            iast_lines = iast_lines[k:k + 2]
    lines = [iast_to_grantha(x) for x in iast_lines]
    iast = "\n".join(iast_lines)
    font = F.BY_ID[_pick(pr, ["grantha-sans", "grantha-sans", "grantha-serif"])]
    surface = _pick(pr, ["copper-plate", "stone", "palm-leaf", "print-scan"])
    if surface == "copper-plate":
        img, meta = M.copper_plate(lines, rng, style="grant", font=font)
    elif surface == "stone":
        img, meta = M.stone(lines, rng, style=_pick(pr, ["inscription", "plaque-grey"]), font=font, spacing=1.6)
    elif surface == "palm-leaf":
        img, meta = M.palm_leaf(lines, rng, font=font, size=int(rng.integers(28, 36)))
    else:
        img, meta = M.print_scan(lines, rng, font=font, aged=pr.uniform(0.2, 0.7), letterpress=True)
    return Sample(img, iast, script=Script.GRANTHA.value, medium=surface,
                  granularity="line" if len(lines) == 1 else "block", lexical=kind, text_source=src, render=meta,
                  fields={"iast": iast, "text_native": "\n".join(lines)})



def _fill_lines(words: list[str], width: int, max_lines: int) -> list[list[int]]:
    """Greedy word wrap by code points; returns word indices per line (at most ``max_lines``)."""
    lines: list[list[int]] = [[]]
    used = 0
    for i, w in enumerate(words):
        if lines[-1] and used + 1 + len(w) > width:
            if len(lines) == max_lines:
                break
            lines.append([])
            used = 0
        used += len(w) + (1 if lines[-1] else 0)
        lines[-1].append(i)
    return lines


def gen_grantha_tamil(pr: random.Random) -> Sample:
    rng = _np(pr)
    surface = _pick(pr, ["palm-leaf", "palm-leaf", "palm-leaf", "print-scan", "print-scan"])
    if pr.random() < 0.2:
        markups, kind, srcs = [mp.nonce_markup(pr)], "nonce", ["nonce"]
    else:
        items = [_pick(pr, corpus.manipravalam()) for _ in range(pr.randint(1, 3 if surface == "palm-leaf" else 2))]
        markups, kind = [it.meta["markup"] for it in items], "corpus"
        srcs = [f"manipravalam:{it.id}" for it in items]
    words = [w for m in markups for w in mp.parse(m)]
    if surface == "palm-leaf":
        p_drop, continua, width, max_lines = _pick(pr, [1.0, 0.6, 0.0]), pr.random() < 0.7, pr.randint(30, 44), 5
    else:
        p_drop, continua, width, max_lines = 0.0, False, pr.randint(28, 40), 3
    nat = [mp.native(w) for w in words]
    if p_drop:
        nat = [drop_pulli(n, p_drop, pr) for n in nat]   # Tamil puḷḷi only; the Grantha virama stays
    rows = _fill_lines(nat, width, max_lines)
    sep = "" if continua else " "
    lines = [sep.join(nat[i] for i in row) for row in rows]
    ref = "\n".join(" ".join(mp.reference(words[i]) for i in row) for row in rows)
    tamil_font = F.BY_ID["lohit-classical"]
    if surface == "palm-leaf":
        img, meta = M.palm_leaf(lines, rng, font=tamil_font, fallback=[F.BY_ID["grantha-sans"]])
    else:
        gfont = F.BY_ID[_pick(pr, ["grantha-serif", "grantha-sans"])]
        img, meta = M.print_scan(lines, rng, font=tamil_font, aged=pr.uniform(0.3, 0.8), letterpress=True,
                                 fallback=[gfont])
        meta["grantha_font"] = gfont.id
    meta.update({"pulli_dropped": p_drop, "scriptio_continua": continua})
    return Sample(img, ref, script=Script.GRANTHA_TAMIL.value, medium=surface,
                  granularity="line" if len(lines) == 1 else "block", lexical=kind,
                  text_source=",".join(srcs), render=meta, fields={"text_native": "\n".join(lines)})


# ---- classification ---------------------------------------------------------------------------------

def _modern_with_reform(pr: random.Random) -> str:
    for _ in range(200):
        it = _pick(pr, corpus.modern())
        if contains_reform_syllable(it.text) and not re.search(r"[A-Za-z0-9₹%°]", it.text):
            return it.text
    return "தமிழ் மொழி இனிமையானது"


def gen_script_id(pr: random.Random, label: str) -> Sample:
    rng = _np(pr)
    medium = Medium.PRINT_SCAN.value
    if label == Script.TAMIL_MODERN.value:
        lines = wrap(_modern_with_reform(pr), pr.randint(22, 36))[:2]
        fn = _pick(pr, ["print_scan", "born_digital", "print_photo", "scene", "handwriting"])
        img, meta = getattr(M, fn)(lines, rng)
        medium = {"print_scan": "print-scan", "born_digital": "born-digital", "print_photo": "print-photo",
                  "scene": "scene", "handwriting": "handwritten"}[fn]
        text = "\n".join(lines)
    elif label == Script.TAMIL_PRE_REFORM.value:
        text, _, _ = _reform_text(pr)
        lines = wrap(text, pr.randint(22, 36))[:2]
        if not contains_reform_syllable("".join(lines)):
            lines = [w for w in text.split() if contains_reform_syllable(w)][:3]
        kind = _pick(pr, ["print", "print", "palm", "stone", "copper"])
        if kind == "print":
            img, meta = M.print_scan(lines, rng, font=F.BY_ID["lohit-classical"], aged=pr.uniform(0.3, 0.9), letterpress=True)
        elif kind == "palm":
            img, meta = M.palm_leaf([scriptio_continua(x) for x in lines], rng)
            medium = Medium.PALM_LEAF.value
        elif kind == "stone":
            img, meta = M.stone([scriptio_continua(x) for x in lines], rng, style="inscription")
            medium = Medium.STONE.value
        else:
            img, meta = M.copper_plate([scriptio_continua(x) for x in lines], rng, style="grant")
            medium = Medium.COPPER_PLATE.value
        text = "\n".join(lines)
    elif label == Script.TAMIL_BRAHMI.value:
        it = _pick(pr, corpus.brahmi_pool())
        nat = brahmi.to_brahmi(it.text, _pick(pr, list(brahmi.Orthography))).replace(" ", "")
        lines = chunk_letters(nat, 14)[:2]
        kind = _pick(pr, ["cave", "estampage", "pottery", "print"])
        if kind == "cave":
            img, meta = M.stone(lines, rng, style="cave", font=F.BY_ID["brahmi"], spacing=1.5)
            medium = Medium.STONE.value
        elif kind == "estampage":
            img, meta = M.estampage(lines, rng, font=F.BY_ID["brahmi"], spacing=1.5)
            medium = Medium.ESTAMPAGE.value
        elif kind == "pottery":
            img, meta = M.pottery(lines[:1], rng)
            medium = Medium.POTTERY.value
        else:
            img, meta = M.born_digital(lines, rng, font=F.BY_ID["brahmi"])
            medium = Medium.BORN_DIGITAL.value
        text = it.text
    elif label == Script.GRANTHA.value:
        it = _pick(pr, corpus.sanskrit())
        lines = [iast_to_grantha(x) for x in it.text.split("\n")[:2]]
        kind = _pick(pr, ["print", "copper", "palm", "stone"])
        font = F.BY_ID[_pick(pr, ["grantha-sans", "grantha-serif"])]
        if kind == "print":
            img, meta = M.print_scan(lines, rng, font=font, aged=pr.uniform(0, 0.6))
        elif kind == "copper":
            img, meta = M.copper_plate(lines, rng, style="grant", font=font)
            medium = Medium.COPPER_PLATE.value
        elif kind == "palm":
            img, meta = M.palm_leaf(lines, rng, font=font)
            medium = Medium.PALM_LEAF.value
        else:
            img, meta = M.stone(lines, rng, style="plaque-grey", font=font)
            medium = Medium.STONE.value
        text = it.text
    else:
        script = Script(label)
        src = _modern_with_reform(pr)
        native = indic.convert(src, script)
        lines = wrap(native, pr.randint(22, 36))[:2]
        font = F.BY_ID[f"{label}-{_pick(pr, ['sans', 'serif'])}"]
        fn = _pick(pr, ["print_scan", "born_digital", "print_photo"])
        img, meta = getattr(M, fn)(lines, rng, font=font)
        medium = {"print_scan": "print-scan", "born_digital": "born-digital", "print_photo": "print-photo"}[fn]
        text = "\n".join(lines)
    return Sample(img, None, "jpg", fields={"label": label, "shown_text": text}, script=label, medium=medium,
                  granularity="block", lexical="corpus", text_source="script-id", render=meta)


def gen_medium_id(pr: random.Random, label: str) -> Sample:
    rng = _np(pr)
    script = Script.TAMIL_MODERN.value
    if label == "printed-paper":
        lines, *_ = _tamil_text(pr, lines=(1, 3), corpus_p=1.0)
        if pr.random() < 0.5:
            img, meta = M.print_scan(lines, rng)
        else:
            img, meta = M.print_photo(lines, rng)
    elif label == "handwritten-paper":
        lines, *_ = _tamil_text(pr, lines=(1, 3), width=(18, 30), corpus_p=1.0)
        img, meta = M.handwriting(lines, rng)
    elif label == "scene-signage":
        it = _pick(pr, corpus.signage())
        lines = [it.text]
        img, meta = M.scene(lines, rng, kind=_pick(pr, _SIGN_BOARD[it.tag]))
    elif label == "palm-leaf":
        text, _, _ = _manuscript_items(pr, 150)
        lines = chunk_letters(scriptio_continua(drop_pulli(text)), pr.randint(38, 52))[:pr.randint(3, 5)]
        img, meta = M.palm_leaf(lines, rng)
        script = Script.TAMIL_PRE_REFORM.value
    elif label == "stone":
        if pr.random() < 0.5:
            it = _pick(pr, corpus.plaques())
            img, meta = M.stone(wrap(it.text, 24)[:3], rng, style=_pick(pr, ["plaque-black", "plaque-grey"]))
        else:
            it = _pick(pr, [h for h in corpus.historical() if h.tag in ("chola-meykeerthi", "donation")])
            img, meta = M.stone(chunk_letters(scriptio_continua(it.text), 22)[:3], rng, style="inscription")
            script = Script.TAMIL_PRE_REFORM.value
    elif label == "estampage":
        if pr.random() < 0.5:
            it = _pick(pr, corpus.brahmi_pool())
            img, meta = M.estampage(chunk_letters(brahmi.to_brahmi(it.text).replace(" ", ""), 14)[:2], rng,
                                    font=F.BY_ID["brahmi"], spacing=1.5)
            script = Script.TAMIL_BRAHMI.value
        else:
            it = _pick(pr, [h for h in corpus.historical() if h.tag in ("chola-meykeerthi", "donation")])
            img, meta = M.estampage(chunk_letters(scriptio_continua(it.text), 22)[:3], rng)
            script = Script.TAMIL_PRE_REFORM.value
    elif label == "copper-plate":
        if pr.random() < 0.7:
            it = _pick(pr, [h for h in corpus.historical() if h.tag in ("chola-meykeerthi", "donation")])
            img, meta = M.copper_plate(chunk_letters(scriptio_continua(it.text), 24)[:3], rng, style="grant")
            script = Script.TAMIL_PRE_REFORM.value
        else:
            it = _pick(pr, corpus.plaques())
            img, meta = M.copper_plate(wrap(it.text, 24)[:3], rng, style="plaque")
    elif label == "pottery":
        it = _pick(pr, [p for p in corpus.brahmi_pool() if p.tag == "pottery-name"])
        if pr.random() < 0.7:
            img, meta = M.pottery([brahmi.to_brahmi(it.text)], rng)
            script = Script.TAMIL_BRAHMI.value
        else:
            img, meta = M.pottery([it.text], rng, font=F.BY_ID["lohit-classical"])
            script = Script.TAMIL_PRE_REFORM.value
    else:
        raise ValueError(label)  # "screen" is rendered by the screen pass
    return Sample(img, None, "jpg", fields={"label": label}, script=script,
                  medium=label, granularity="block", text_source="medium-id", render=meta)


def gen_translation(pr: random.Random, item, medium: str) -> Sample:
    rng = _np(pr)
    lines = wrap(item.text, pr.randint(20, 34))
    if medium == "print-scan":
        img, meta = M.print_scan(lines, rng)
    elif medium == "scene":
        img, meta = M.scene(lines[:2] if len(lines) > 2 else lines, rng,
                            kind=_pick(pr, ["shop", "office", "banner", "road"]))
        lines = lines[:2] if len(lines) > 2 else lines
    else:
        img, meta = M.handwriting(lines, rng)
    return Sample(img, "\n".join(lines), fields={"english": item.english}, medium=medium,
                  granularity="line" if len(lines) == 1 else "block", text_source=f"{item.source}:{item.id}",
                  render=meta)


# ----------------------------------------------------------------------------- screen pass

def _screen_data(template: str, pr: random.Random) -> tuple[dict, str, str]:
    modern = corpus.modern()
    sent = lambda: _pick(pr, modern).text  # noqa: E731
    nonce = pr.random() < 0.12
    body = corpus.nonce_line(pr, (5, 9)) if nonce else sent()
    kind = "nonce" if nonce else "corpus"
    hhmm = f"{pr.randint(6, 23)}:{pr.randint(0, 59):02d}"
    if template == "news":
        head = _pick(pr, [m for m in modern if len(m.text) <= 48]).text
        d = {"kicker": _pick(pr, ["செய்திகள்", "வானிலை", "விளையாட்டு", "தொழில்நுட்பம்", "கல்வி"]) +
                       f" · {pr.randint(1, 11)} மணி நேரம் முன்", "headline": head, "body": body}
    elif template == "chat":
        msgs = [(body if i == 0 else sent(), f"{pr.randint(6, 23)}:{pr.randint(0, 59):02d}") for i in range(pr.randint(1, 3))]
        d = {"messages": msgs}
    elif template == "settings":
        ui = corpus.ui_strings()
        d = {"title": _pick(pr, ["அமைப்புகள்", "சுயவிவரம்", "தனியுரிமை", "அறிவிப்புகள்"]),
             "items": [_pick(pr, ui).text for _ in range(pr.randint(3, 6))]}
        kind = "corpus"
    elif template == "subtitle":
        d = {"subtitle": body}
    elif template == "form":
        names = corpus.names()
        d = {"title": _pick(pr, ["விண்ணப்பப் படிவம்", "பதிவுப் படிவம்", "தொடர்பு விவரங்கள்"]),
             "fields": [("பெயர்", _pick(pr, [n for n in names if n.tag == "person"]).text),
                        ("ஊர்", _pick(pr, [n for n in names if n.tag == "place"]).text),
                        ("கைபேசி எண்", f"{pr.randint(70000, 99999)} {pr.randint(10000, 99999)}"),
                        ("பிறந்த தேதி", f"{pr.randint(1, 28):02d}/{pr.randint(1, 12):02d}/{pr.randint(1950, 2010)}")][:pr.randint(2, 4)]}
        kind = "corpus"
    elif template == "table":
        heads = [["பொருள்", "அளவு", "விலை"], ["பெயர்", "ஊர்", "மதிப்பெண்"], ["நாள்", "வெப்பநிலை", "மழை"]]
        h = _pick(pr, heads)
        rows = []
        for _ in range(pr.randint(2, 4)):
            if h[0] == "பொருள்":
                rows.append([_pick(pr, ["அரிசி", "பருப்பு", "எண்ணெய்", "சர்க்கரை", "உப்பு", "பால்"]),
                             f"{pr.randint(1, 10)} கிலோ", f"₹{pr.randint(20, 900)}"])
            elif h[0] == "பெயர்":
                rows.append([_pick(pr, [n for n in corpus.names() if n.tag == "person"]).text,
                             _pick(pr, [n for n in corpus.names() if n.tag == "place"]).text, str(pr.randint(35, 100))])
            else:
                rows.append([_pick(pr, ["திங்கள்", "செவ்வாய்", "புதன்", "வியாழன்", "வெள்ளி", "சனி", "ஞாயிறு"]),
                             f"{pr.randint(24, 41)}°C", f"{pr.randint(0, 80)} மி.மீ"])
        d = {"header": h, "rows": rows}
        kind = "corpus"
    elif template == "notification":
        d = {"app": _pick(pr, ["வங்கி", "செய்திகள்", "வானிலை", "அஞ்சல்", "தொடர்வண்டி"]),
             "time": _pick(pr, ["இப்போது", f"{pr.randint(2, 59)} நிமிடம் முன்", hhmm]),
             "title": _pick(pr, corpus.ui_strings()).text, "body": body}
    else:
        raise ValueError(template)
    return d, kind, f"screen:{template}"


SCREEN_TEMPLATES = ["news", "chat", "settings", "subtitle", "form", "table", "notification"]


# ----------------------------------------------------------------------------- driver

GENERATORS: dict[str, Callable[[random.Random], Sample]] = {
    "print-digital": gen_print_digital, "print-scan": gen_print_scan, "print-photo": gen_print_photo,
    "numerals-symbols": gen_numerals, "handwriting": gen_handwriting, "scene": gen_scene,
    "pre-reform-print": gen_pre_reform, "palm-leaf-synth": gen_palm_leaf, "stone": gen_stone,
    "copper-plate": gen_copper, "tamil-brahmi": gen_tamil_brahmi, "grantha": gen_grantha,
    "grantha-tamil": gen_grantha_tamil,
}


def sample_rng(seed: int, split: str, subset: str, index: int) -> random.Random:
    return random.Random(f"{BENCHMARK_VERSION}:{seed}:{split}:{subset}:{index}")


def _save(sample: Sample, out_dir: Path, subset: str, sid: str) -> dict:
    rel = Path("images") / subset / f"{sid}.{sample.fmt}"
    path = out_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    img = sample.image.convert("RGB")
    if sample.fmt == "png":
        img.save(path, "PNG", optimize=True)
    else:
        img.save(path, "JPEG", quality=88, optimize=True, progressive=True)
    data = path.read_bytes()
    return {"image": rel.as_posix(), "sha256": hashlib.sha256(data).hexdigest(),
            "width": img.width, "height": img.height, "bytes": len(data)}


def _row(spec: S.SubsetSpec, sid: str, split: str, sample: Sample, saved: dict) -> dict:
    row = {"id": sid, "subset": spec.id, "split": split, "task": spec.task.value, **saved,
           "text": sample.text, "script": sample.script, "medium": sample.medium,
           "granularity": sample.granularity, "provenance": spec.provenance.value,
           "lexical": sample.lexical, "text_source": sample.text_source,
           "render": sample.render, "license": spec.license}
    row.update(sample.fields)
    return row


def _work(args) -> dict:
    out_dir, seed, split, subset, index, label = args
    spec = S.get(subset)
    pr = sample_rng(seed, split, subset, index)
    if spec.task == Task.SCRIPT_ID:
        sample = gen_script_id(pr, label)
    elif spec.task == Task.MEDIUM_ID:
        sample = gen_medium_id(pr, label)
    elif spec.task == Task.TRANSLATION:
        item, medium = label
        sample = gen_translation(pr, item, medium)
    else:
        sample = GENERATORS[subset](pr)
    sid = f"{subset}-{index:04d}"
    return _row(spec, sid, split, sample, _save(sample, Path(out_dir), subset, sid))


def _translation_plan(seed: int, split: str, count: int):
    pr = random.Random(f"{BENCHMARK_VERSION}:{seed}:{split}:translate-plan")
    items = list(corpus.modern()) + [g for g in corpus.signage()]
    pr.shuffle(items)
    media = ["print-scan", "screen", "scene", "handwritten"]
    plan = []
    for i, it in enumerate(items[:count]):
        m = media[i % 4] if it.source == "modern" else "scene"
        plan.append((it, m))
    return plan


def plan_tasks(out_dir: Path, seed: int, split: str, subset_ids: list[str], counts: dict[str, int]):
    tasks, screen_tasks = [], []
    for sid in subset_ids:
        spec = S.get(sid)
        n = counts.get(sid, spec.count)
        if spec.provenance.value == "real":
            continue
        if sid == "screen":
            screen_tasks += [(sid, i, None) for i in range(n)]
        elif spec.task in (Task.SCRIPT_ID, Task.MEDIUM_ID):
            labels = list(spec.labels)
            for i in range(n):
                lab = labels[i % len(labels)]
                if lab == "screen":
                    screen_tasks.append((sid, i, lab))
                else:
                    tasks.append((str(out_dir), seed, split, sid, i, lab))
        elif spec.task == Task.TRANSLATION:
            for i, (it, medium) in enumerate(_translation_plan(seed, split, n)):
                if medium == "screen":
                    screen_tasks.append((sid, i, it))
                else:
                    tasks.append((str(out_dir), seed, split, sid, i, (it, medium)))
        else:
            tasks += [(str(out_dir), seed, split, sid, i, None) for i in range(n)]
    return tasks, screen_tasks


def run_screen_tasks(out_dir: Path, seed: int, split: str, screen_tasks) -> list[dict]:
    if not screen_tasks:
        return []
    from .render.screen import ScreenRenderer
    renderer = ScreenRenderer()
    rows = []
    try:
        for sid, i, extra in screen_tasks:
            spec = S.get(sid)
            pr = sample_rng(seed, split, sid, i)
            fields, lexical = {}, "corpus"
            if spec.task == Task.TRANSLATION:
                template, data, src = "subtitle", {"subtitle": extra.text}, f"{extra.source}:{extra.id}"
                fields["english"] = extra.english
            else:
                template = _pick(pr, SCREEN_TEMPLATES)
                data, lexical, src = _screen_data(template, pr)
            img, text, meta = renderer.render(template, data, pr)
            if spec.task == Task.MEDIUM_ID:
                fields["label"] = "screen"
            fmt = "jpg" if "recompressed" in meta else "png"
            sample = Sample(img, None if spec.task == Task.MEDIUM_ID else text, fmt, fields=fields,
                            medium=Medium.SCREEN.value, granularity="block", lexical=lexical, text_source=src,
                            render=meta)
            sample_id = f"{sid}-{i:04d}"
            rows.append(_row(spec, sample_id, split, sample, _save(sample, out_dir, sid, sample_id)))
    finally:
        renderer.close()
    return rows


def build(out_dir: Path, *, seed: int = DEFAULT_SEED, split: str = "test", subset_ids: list[str] | None = None,
          counts: dict[str, int] | None = None, workers: int = 4, cict_root: Path | None = None,
          progress: bool = True) -> list[dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subset_ids = subset_ids or [s.id for s in S.SUBSETS]
    counts = counts or {}
    tasks, screen_tasks = plan_tasks(out_dir, seed, split, subset_ids, counts)
    rows: list[dict] = []
    if tasks:
        if workers > 1:
            from multiprocessing import get_context
            with get_context("fork").Pool(workers) as pool:
                for k, row in enumerate(pool.imap_unordered(_work, tasks, chunksize=4)):
                    rows.append(row)
                    if progress and k % 50 == 0:
                        print(f"  rendered {k + 1}/{len(tasks)}", flush=True)
        else:
            rows += [_work(t) for t in tasks]
    rows += run_screen_tasks(out_dir, seed, split, screen_tasks)
    if "palm-leaf-cict" in subset_ids:
        from .external import cict
        if cict_root is None:
            raise SystemExit("palm-leaf-cict needs --cict-root (path to the cict-htr directory)")
        rows += cict.build_subset(Path(cict_root), out_dir, split=split, limit=counts.get("palm-leaf-cict"))
    order = {s.id: k for k, s in enumerate(S.SUBSETS)}
    rows.sort(key=lambda r: (order[r["subset"]], r["id"]))
    return rows


def write_manifests(out_dir: Path, rows: list[dict], *, split: str = "test", seed: int = DEFAULT_SEED,
                    lite: int = LITE_PER_SUBSET) -> None:
    out_dir = Path(out_dir)
    with open(out_dir / f"manifest-{split}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if split == "test":
        seen: dict[str, int] = {}
        with open(out_dir / "manifest-lite.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                k = seen.get(r["subset"], 0)
                spec = S.get(r["subset"])
                cap = 3 * len(spec.labels) if spec.labels else lite   # 3 per class for identification
                if k < cap:
                    f.write(json.dumps({**r, "split": "lite"}, ensure_ascii=False) + "\n")
                    seen[r["subset"]] = k + 1
    meta = {"benchmark": "Tamil OCR/HTR Benchmark", "version": BENCHMARK_VERSION, "seed": seed if split != "private" else None,
            "subsets": [s.to_json() for s in S.SUBSETS],
            "counts": {sid: sum(1 for r in rows if r["subset"] == sid) for sid in {r["subset"] for r in rows}}}
    (out_dir / "subsets.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_dir / f"CHECKSUMS-{split}.sha256", "w") as f:
        for r in rows:
            f.write(f"{r['sha256']}  {r['image']}\n")
