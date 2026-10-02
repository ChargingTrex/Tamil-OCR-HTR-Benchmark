"""Shaping-correct text → ink-mask rendering.

Tamil needs complex text layout (the two-part vowel signs of கொ/கோ/கௌ wrap around the
consonant), so all rendering goes through Pillow's Raqm/HarfBuzz backend. Characters a
font does not cover fall back run-by-run to another font (e.g. Tamil Supplement signs,
Latin inside a Tamil-only face, or Grantha words inside Tamil text).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from . import fonts as F


@dataclass
class Rendered:
    mask: np.ndarray          # float32 in [0, 1], 1 = ink
    line_boxes: list[tuple[int, int, int, int]]


def _runs(text: str, chain: list[F.FontSpec]) -> list[tuple[str, F.FontSpec]]:
    runs: list[tuple[str, F.FontSpec]] = []
    for ch in text:
        if ch.isspace() and runs:
            runs[-1] = (runs[-1][0] + ch, runs[-1][1])
            continue
        font = next((f for f in chain if F.covers(f, ch)), chain[0])
        if runs and runs[-1][1] is font:
            runs[-1] = (runs[-1][0] + ch, font)
        elif runs and _is_mark(ch):
            runs[-1] = (runs[-1][0] + ch, runs[-1][1])   # never split a cluster
        else:
            runs.append((ch, font))
    return runs


def _is_mark(ch: str) -> bool:
    import unicodedata
    return unicodedata.category(ch) in ("Mn", "Mc")


def _language(font: F.FontSpec) -> str | None:
    if "tamil" in font.tags:
        return "ta"
    return None


def render_line(text: str, font: F.FontSpec, size: int, *, fallback: list[F.FontSpec] | None = None,
                tracking: float = 0.0) -> np.ndarray:
    """Render one line to a tight float mask (height ≈ 1.6 × size)."""
    chain = [font] + (fallback or []) + [F.LATIN_FALLBACK, F.SUPPLEMENT, F.grantha_for(font)]
    runs = _runs(text, chain)
    pieces = []
    asc_max, desc_max = 0, 0
    for run, f in runs:
        pil = F.load(f.id, size)
        asc, desc = pil.getmetrics()
        asc_max, desc_max = max(asc_max, asc), max(desc_max, desc)
        pieces.append((run, f, pil, asc))
    pad = int(size * 0.35)
    height = asc_max + desc_max + 2 * pad
    widths = []
    for run, f, pil, _ in pieces:
        w = pil.getlength(run, language=_language(f)) if run else 0
        widths.append(int(np.ceil(w + tracking * len(run))))
    width = max(1, sum(widths) + 2 * pad)
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    x = pad
    for (run, f, pil, asc), w in zip(pieces, widths):
        y = pad + asc_max - asc
        if tracking:
            for cl in _clusters(run):
                draw.text((x, y), cl, font=pil, fill=255, language=_language(f))
                x += pil.getlength(cl, language=_language(f)) + tracking
        else:
            draw.text((x, y), run, font=pil, fill=255, language=_language(f))
            x += w
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return _trim_x(arr, pad)


def _clusters(text: str) -> list[str]:
    out: list[str] = []
    for ch in text:
        if out and (_is_mark(ch) or ch in "்‍‌"):
            out[-1] += ch
        else:
            out.append(ch)
    return out


def _trim_x(arr: np.ndarray, pad: int) -> np.ndarray:
    cols = np.where(arr.max(0) > 0.02)[0]
    if cols.size == 0:
        return arr
    x0, x1 = max(0, cols[0] - pad), min(arr.shape[1], cols[-1] + pad + 1)
    return arr[:, x0:x1]


def render_block(lines: list[str], font: F.FontSpec, size: int, *, line_spacing: float = 1.15,
                 align: str = "left", fallback: list[F.FontSpec] | None = None,
                 tracking: float = 0.0, margin: int | None = None,
                 jitter_px: int = 0, rng: np.random.Generator | None = None) -> Rendered:
    """Stack rendered lines into one mask; returns per-line boxes for layout-aware media."""
    masks = [render_line(ln, font, size, fallback=fallback, tracking=tracking) if ln.strip()
             else np.zeros((int(size * 1.6), 4), np.float32) for ln in lines]
    pitch = int(size * line_spacing * 1.35)
    margin = int(size * 0.6) if margin is None else margin
    width = max(m.shape[1] for m in masks) + 2 * margin
    height = pitch * (len(masks) - 1) + max(m.shape[0] for m in masks) + 2 * margin
    out = np.zeros((height, width), np.float32)
    boxes = []
    for i, m in enumerate(masks):
        y = margin + i * pitch
        if jitter_px and rng is not None:
            y += int(rng.integers(-jitter_px, jitter_px + 1))
            y = max(0, min(height - m.shape[0], y))
        if align == "center":
            x = (width - m.shape[1]) // 2
        elif align == "right":
            x = width - margin - m.shape[1]
        else:
            x = margin
        region = out[y:y + m.shape[0], x:x + m.shape[1]]
        np.maximum(region, m[:region.shape[0], :region.shape[1]], out=region)
        boxes.append((x, y, m.shape[1], m.shape[0]))
    return Rendered(out, boxes)


def ink_bbox(mask: np.ndarray, thr: float = 0.05) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > thr)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
