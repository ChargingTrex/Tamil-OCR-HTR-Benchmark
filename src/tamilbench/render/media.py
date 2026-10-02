"""Medium simulators: text lines in, a photograph/scan/screenshot-like image out.

Every function has the signature ``fn(lines, rng, **options) -> (PIL.Image, meta)`` and is
fully determined by ``rng``. ``meta`` records the choices made (font, size, style,
degradation knobs) so results can be sliced by them later.

These are *proxies*: they reproduce the visual properties that make each medium hard
(texture, relief, curvature, damage, missing marks) but they are not real artefacts. Real
artefacts live in separate ``provenance: real`` subsets.

``text_layer(fn)`` transforms the ink of every text mask before it is drawn, after the
layout has been decided from the real text. The blank and effaced controls use it to
render a surface exactly as it would look with writing — and then without it, or with
the writing worn away beyond reading.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage as ndi

from . import effects as E
from . import fonts as F
from .text import render_block, render_line

Rng = np.random.Generator

_TEXT_LAYER: Callable[[np.ndarray], np.ndarray] | None = None


@contextmanager
def text_layer(fn: Callable[[np.ndarray], np.ndarray]):
    """Within the block, every text mask passes through ``fn`` just before it is drawn."""
    global _TEXT_LAYER
    prev, _TEXT_LAYER = _TEXT_LAYER, fn
    try:
        yield
    finally:
        _TEXT_LAYER = prev


def _ink(mask: np.ndarray) -> np.ndarray:
    return mask if _TEXT_LAYER is None else _TEXT_LAYER(mask).astype(np.float32)


def _pick(rng: Rng, seq):
    return seq[int(rng.integers(0, len(seq)))]


def _fonts_for(rng: Rng, tags: tuple[str, ...], text: str, exclude=("nolatin",)) -> F.FontSpec:
    cands = F.select(*tags, exclude=exclude) or F.select(*tags)
    ok = [f for f in cands if F.covers(f, text.replace("\n", ""))]
    return _pick(rng, ok or cands)


def _canvas(mask: np.ndarray, pad_y: int, pad_x: int) -> np.ndarray:
    h, w = mask.shape
    return E.pad_to(mask, h + 2 * pad_y, w + 2 * pad_x, pad_y, pad_x)


def _finish_photo(img: Image.Image, rng: Rng, *, persp=(0.02, 0.08), blur=(0.3, 1.2), quality=(65, 90),
                  bg=(0.3, 0.3, 0.3), max_edge=1600) -> tuple[Image.Image, dict]:
    w, h = img.size
    margin = int(0.08 * max(w, h))
    bgarr = E.tint(E.fbm(h + 2 * margin, w + 2 * margin, rng, scale=80), np.array(bg) * 0.7, np.array(bg) * 1.3)
    canvas = E.to_image(bgarr)
    canvas.paste(img, (margin, margin))
    s = rng.uniform(*persp)
    canvas = E.perspective(canvas, rng, strength=s, fill=tuple(int(c * 255) for c in bg))
    arr = E.from_image(canvas)
    arr = E.light_gradient(arr, rng, strength=rng.uniform(0.1, 0.35))
    if rng.random() < 0.35:
        arr = E.shadow_band(arr, rng, strength=rng.uniform(0.15, 0.4))
    arr = E.color_cast(arr, rng)
    b = rng.uniform(*blur)
    arr = E.gaussian_blur(arr, b)
    if rng.random() < 0.15:
        arr = E.motion_blur(arr, int(rng.integers(3, 7)), rng.uniform(0, 180))
    arr = E.noise(arr, rng, rng.uniform(0.008, 0.03))
    arr = E.vignette(arr, rng.uniform(0.05, 0.3))
    out = E.fit_long_edge(E.to_image(arr), max_edge)
    q = int(rng.integers(*quality))
    return E.jpeg_roundtrip(out, q), {"perspective": round(s, 3), "blur": round(b, 2), "jpeg": q}


def _finish_scan(arr: np.ndarray, rng: Rng, *, skew=1.5, gray_p=0.4, quality=(70, 93),
                 fill=(245, 243, 238), max_edge=1600) -> tuple[Image.Image, dict]:
    arr = E.gaussian_blur(arr, rng.uniform(0.25, 0.7))
    arr = E.noise(arr, rng, rng.uniform(0.005, 0.025))
    if rng.random() < 0.4:
        arr = E.salt_pepper(arr, rng, rng.uniform(0.0003, 0.002))
    img = E.to_image(arr)
    deg = rng.uniform(-skew, skew)
    img = E.rotate(img, deg, fill=fill)
    if rng.random() < gray_p:
        img = img.convert("L").convert("RGB")
    img = E.fit_long_edge(img, max_edge)
    q = int(rng.integers(*quality))
    return E.jpeg_roundtrip(img, q), {"skew": round(deg, 2), "jpeg": q}


# ============================================================================ born-digital

def born_digital(lines, rng: Rng, font: F.FontSpec | None = None, size: int | None = None):
    text = "\n".join(lines)
    font = font or _fonts_for(rng, ("tamil",), text, exclude=("nolatin", "classical", "hand"))
    size = size or int(rng.integers(22, 46))
    r = render_block(lines, font, size, line_spacing=rng.uniform(1.0, 1.3),
                     align=_pick(rng, ["left", "left", "left", "center"]))
    scheme = rng.random()
    if scheme < 0.7:
        bg, fg = (1, 1, 1), (0.05, 0.05, 0.06)
    elif scheme < 0.85:
        bg, fg = (rng.uniform(.85, 1), rng.uniform(.85, 1), rng.uniform(.85, 1)), (rng.uniform(0, .4), 0.1, rng.uniform(0, .5))
    else:
        bg, fg = (rng.uniform(0, .2), rng.uniform(0, .2), rng.uniform(.1, .3)), (0.95, 0.95, 0.9)
    m = _ink(r.mask)
    img = E.composite(E.solid(*m.shape, bg), m, fg)
    return E.to_image(img), {"font": font.id, "size": size, "style": "dark-on-light" if scheme < 0.85 else "light-on-dark"}


# ============================================================================ print

def _print_page(lines, rng: Rng, font: F.FontSpec, size: int, *, aged: float, letterpress: bool,
                fallback: list[F.FontSpec] | None = None):
    r = render_block(lines, font, size, line_spacing=rng.uniform(1.0, 1.25),
                     align=_pick(rng, ["left", "left", "center"]), fallback=fallback)
    m = E.ink_spread(r.mask, sigma=rng.uniform(0.3, 0.8), gain=rng.uniform(1.0, 1.4))
    if letterpress:
        m = E.ink_dropout(m, rng, amount=rng.uniform(0.05, 0.25), scale=rng.uniform(3, 8))
        m = np.clip(m * rng.uniform(0.9, 1.0), 0, 1)
    elif rng.random() < 0.4:
        m = E.ink_dropout(m, rng, amount=rng.uniform(0.02, 0.15))
    m = _ink(_canvas(m, int(size * rng.uniform(0.8, 2.0)), int(size * rng.uniform(1.0, 2.5))))
    page = E.paper(*m.shape, rng, aged=aged)
    if rng.random() < (0.5 if aged else 0.2):
        ghost = np.fliplr(np.roll(m, int(rng.integers(-size, size)), axis=0))
        page = E.composite(page, E.gaussian_blur(ghost, 1.2), (0.25, 0.22, 0.2), alpha=rng.uniform(0.05, 0.14))
    ink = (0.06, 0.06, 0.08) if not aged else (0.16, 0.12, 0.09)
    return E.composite(page, m, ink, alpha=rng.uniform(0.85, 1.0))


def print_scan(lines, rng: Rng, font: F.FontSpec | None = None, *, aged: float = 0.0, letterpress=False,
               size: int | None = None, fallback: list[F.FontSpec] | None = None):
    text = "\n".join(lines)
    font = font or _fonts_for(rng, ("tamil",), text, exclude=("nolatin", "hand", "classical", "rounded"))
    size = size or int(rng.integers(26, 42))
    arr = _print_page(lines, rng, font, size, aged=aged, letterpress=letterpress, fallback=fallback)
    img, meta = _finish_scan(arr, rng, gray_p=0.1 if aged else 0.4)
    return img, {"font": font.id, "size": size, "aged": round(aged, 2), **meta}


def print_photo(lines, rng: Rng, font: F.FontSpec | None = None, size: int | None = None):
    text = "\n".join(lines)
    font = font or _fonts_for(rng, ("tamil",), text, exclude=("nolatin", "hand", "classical", "rounded"))
    size = size or int(rng.integers(28, 44))
    arr = _print_page(lines, rng, font, size, aged=rng.uniform(0, 0.25), letterpress=False)
    if rng.random() < 0.6:
        arr = E.warp_rows(arr, rng.uniform(1, 5), rng.uniform(600, 1600), rng.uniform(0, 6))
    img, meta = _finish_photo(E.to_image(arr), rng, persp=(0.02, 0.1),
                              bg=_pick(rng, [(0.45, 0.32, 0.2), (0.2, 0.2, 0.22), (0.6, 0.6, 0.58)]))
    return img, {"font": font.id, "size": size, **meta}


# ============================================================================ handwriting

HAND_FONTS = ("kavivanar", "kavivanar", "kavivanar", "kavivanar", "arima")


def handwriting(lines, rng: Rng, font: F.FontSpec | None = None):
    font = font or F.BY_ID[_pick(rng, HAND_FONTS)]
    size = int(rng.integers(30, 44))
    pitch = int(size * rng.uniform(1.7, 2.1))
    slant = rng.uniform(-0.12, 0.25)
    heavy = rng.random() < 0.4
    line_masks = []
    for ln in lines:
        words = ln.split(" ")
        pieces = []
        for w in words:
            if not w:
                continue
            m = render_line(w, font, int(size * rng.uniform(0.92, 1.08)))
            m = E.shear(m, slant + rng.uniform(-0.05, 0.05))
            m = E.elastic(m, rng, alpha=rng.uniform(1.5, 3.0), sigma=rng.uniform(4, 7))
            if heavy:
                m = E.thicken(m, 1)
            dy = int(rng.normal(0, size * 0.06))
            pieces.append((m, dy))
        if not pieces:
            line_masks.append(np.zeros((size, 10), np.float32))
            continue
        H = max(p.shape[0] for p, _ in pieces) + 2 * size // 3
        gap = int(size * rng.uniform(0.35, 0.6))
        W = sum(p.shape[1] for p, _ in pieces) + gap * len(pieces)
        canvas = np.zeros((H, W), np.float32)
        x = 0
        for p, dy in pieces:
            y = max(0, min(H - p.shape[0], size // 3 + dy))
            reg = canvas[y:y + p.shape[0], x:x + p.shape[1]]
            np.maximum(reg, p[:reg.shape[0], :reg.shape[1]], out=reg)
            x += p.shape[1] + gap + int(rng.integers(-4, 5))
        canvas = E.warp_rows(canvas, rng.uniform(0, size * 0.12), rng.uniform(500, 1500), rng.uniform(0, 6))
        line_masks.append(canvas)
    W = max(m.shape[1] for m in line_masks) + 2 * size
    H = pitch * len(line_masks) + 2 * size
    mask = np.zeros((H, W), np.float32)
    left = int(size * rng.uniform(1.2, 2.2))
    for i, m in enumerate(line_masks):
        y = size // 2 + i * pitch + int(rng.integers(-3, 4))
        x = left + int(rng.integers(-6, 7))
        reg = mask[y:y + m.shape[0], x:x + m.shape[1]]
        np.maximum(reg, m[:reg.shape[0], :reg.shape[1]], out=reg)
    ruled = rng.random() < 0.6
    page = E.paper(H, W, rng, base=(0.98, 0.98, 0.96))
    if ruled:
        rl = np.zeros((H, W), np.float32)
        for i in range(len(line_masks) + 1):
            y = size // 2 + i * pitch + int(size * 1.05)
            if 0 <= y < H:
                rl[y:y + 2, :] = 1
        page = E.composite(page, rl, (0.55, 0.7, 0.9), alpha=0.6)
        mg = np.zeros((H, W), np.float32)
        mg[:, int(left * 0.6):int(left * 0.6) + 2] = 1
        page = E.composite(page, mg, (0.9, 0.4, 0.4), alpha=0.5)
    ink = _pick(rng, [(0.08, 0.15, 0.5), (0.1, 0.1, 0.12), (0.05, 0.1, 0.35)])
    pressure = 0.7 + 0.3 * E.fbm(H, W, rng, scale=40, octaves=2)
    img = E.composite(page, _ink(mask) * pressure, ink)
    if rng.random() < 0.5:
        out, meta = _finish_scan(img, rng, skew=2.5, gray_p=0.15)
    else:
        out, meta = _finish_photo(E.to_image(img), rng, persp=(0.02, 0.08),
                                  bg=_pick(rng, [(0.45, 0.32, 0.2), (0.25, 0.25, 0.28)]))
    return out, {"font": font.id, "size": size, "slant": round(slant, 2), "ruled": ruled, **meta}


# ============================================================================ scene text

_BOARDS = {
    "shop": [((0.95, 0.8, 0.1), (0.75, 0.05, 0.05)), ((0.1, 0.25, 0.6), (1, 1, 1)), ((0.8, 0.1, 0.1), (1, 0.95, 0.3)),
             ((1, 1, 1), (0.05, 0.3, 0.1)), ((0.05, 0.4, 0.2), (1, 1, 0.9)), ((0.95, 0.5, 0.1), (0.1, 0.05, 0.3))],
    "road": [((0.05, 0.45, 0.2), (1, 1, 1)), ((0.1, 0.25, 0.65), (1, 1, 1))],
    "office": [((0.05, 0.25, 0.55), (1, 1, 1)), ((1, 1, 1), (0.1, 0.1, 0.1)), ((0.95, 0.9, 0.75), (0.5, 0.05, 0.05))],
    "banner": [((0.6, 0.0, 0.3), (1, 0.9, 0.2)), ((0.0, 0.3, 0.6), (1, 1, 1)), ((0.9, 0.3, 0.0), (1, 1, 1))],
}


def scene(lines, rng: Rng, kind: str | None = None):
    kind = kind or _pick(rng, ["shop", "shop", "road", "office", "banner", "led", "wall"])
    text = "\n".join(lines)
    if kind == "led":
        return _led_board(lines, rng)
    if kind == "wall":
        font = _fonts_for(rng, ("tamil", "display"), text)
    else:
        font = _fonts_for(rng, ("tamil", "bold"), text)
    size = int(rng.integers(40, 72))
    r = render_block(lines, font, size, line_spacing=1.05, align="center", margin=int(size * 0.7))
    m = _ink(r.mask)
    h, w = m.shape
    if kind == "wall":
        bg = E.tint(E.fbm(h, w, rng, scale=30), (0.75, 0.72, 0.65), (0.95, 0.93, 0.88))
        bg = E.composite(bg, (E.fbm(h, w, rng, scale=60) > 0.75).astype(np.float32), (0.55, 0.5, 0.45), alpha=0.25)
        fg = _pick(rng, [(0.1, 0.1, 0.45), (0.6, 0.05, 0.05), (0.05, 0.3, 0.1)])
        mm = E.roughen(m, rng, 0.25)
        img = E.composite(bg, mm, fg, alpha=rng.uniform(0.75, 0.95))
    else:
        bgc, fg = _pick(rng, _BOARDS[kind])
        bg = E.solid(h, w, bgc)
        if kind == "banner":
            g = np.linspace(0, 1, w)[None, :, None]
            bg = bg * (0.75 + 0.5 * g)
        if rng.random() < 0.5:   # outline / drop shadow
            o = np.clip(E.thicken(m, int(max(1, size // 18))) - m, 0, 1)
            bg = E.composite(bg, o, (0.05, 0.05, 0.05) if np.mean(fg) > 0.5 else (1, 1, 1))
        img = E.composite(bg, m, fg)
        border = np.zeros((h, w), np.float32)
        t = max(3, size // 10)
        border[:t, :] = border[-t:, :] = 1
        border[:, :t] = border[:, -t:] = 1
        img = E.composite(img, border, _pick(rng, [(0.9, 0.9, 0.9), (0.2, 0.2, 0.2), fg]))
        img = img * (0.92 + 0.08 * E.fbm(h, w, rng, scale=50)[..., None])
    out, meta = _finish_photo(E.to_image(np.clip(img, 0, 1)), rng, persp=(0.04, 0.15), blur=(0.3, 1.4),
                              bg=_pick(rng, [(0.5, 0.55, 0.6), (0.35, 0.3, 0.25), (0.6, 0.65, 0.7)]))
    return out, {"kind": kind, "font": font.id, "size": size, **meta}


def _led_board(lines, rng: Rng):
    font = F.BY_ID[_pick(rng, ["noto-sans-bold", "hind-bold", "mukta-bold"])]
    small = int(rng.integers(13, 17))
    r = render_block(lines, font, small, line_spacing=1.0, align="center", margin=small // 2)
    on = (_ink(r.mask) > 0.45).astype(np.float32)
    k = int(rng.integers(5, 7))
    up = np.kron(on, np.ones((k, k), np.float32))
    yy, xx = np.mgrid[0:k, 0:k]
    dot = (((yy - k / 2 + .5) ** 2 + (xx - k / 2 + .5) ** 2) <= (k * 0.38) ** 2).astype(np.float32)
    dots = up * np.tile(dot, on.shape)
    h, w = dots.shape
    col = np.array(_pick(rng, [(1.0, 0.55, 0.05), (1.0, 0.2, 0.05), (0.2, 1.0, 0.2)]), np.float32)
    img = E.solid(h, w, (0.04, 0.04, 0.05)) + 0.06 * np.tile(dot, (h // k + 1, w // k + 1))[:h, :w, None]
    img = E.composite(img, dots, col)
    glow = E.gaussian_blur(dots, k * 0.6)
    img = np.clip(img + 0.6 * glow[..., None] * col, 0, 1)
    out, meta = _finish_photo(E.to_image(img), rng, persp=(0.03, 0.12), blur=(0.4, 1.2), bg=(0.15, 0.15, 0.17))
    return out, {"kind": "led", "font": font.id, "size": small, **meta}


# ============================================================================ palm leaf

def _cut_before(mask: np.ndarray, limit: int, lo: int, window: int) -> int:
    """Rightmost (near-)ink-free column in ``(lo, limit]`` so a line can be split at a hole
    without cutting through a letter."""
    a, b = max(lo + 1, limit - window), max(lo + 1, min(limit, mask.shape[1] - 1))
    if b <= a:
        return b
    col = mask[:, a:b + 1].sum(0)
    free = np.where(col <= col.min() + 1e-3)[0]
    return int(a + free[-1])


def palm_leaf(lines, rng: Rng, font: F.FontSpec | None = None, size: int | None = None,
              fallback: list[F.FontSpec] | None = None):
    font = font or F.BY_ID["lohit-classical"]
    size = size or int(rng.integers(26, 34))
    pitch = int(size * rng.uniform(1.25, 1.45))
    line_masks = [render_line(ln, font, size, fallback=fallback) for ln in lines]
    text_w = max(m.shape[1] for m in line_masks)
    n = len(lines)
    margin_x = int(size * rng.uniform(2.0, 4.0))
    hole_r = int(size * rng.uniform(0.45, 0.6))
    zone_pad = int(size * 0.4)
    zw = 2 * hole_r + 2 * zone_pad
    W = text_w + 2 * margin_x + 2 * zw
    H = pitch * n + int(size * 1.6)
    holes_x = [int(W * rng.uniform(0.27, 0.33)), int(W * rng.uniform(0.67, 0.73))]
    hole_y = H // 2 + int(rng.integers(-size // 4, size // 4 + 1))
    zones = [(hx - hole_r - zone_pad, hx + hole_r + zone_pad) for hx in holes_x]
    mask = np.zeros((H, W), np.float32)

    def put(seg, y, x):
        reg = mask[y:y + seg.shape[0], x:x + seg.shape[1]]
        np.maximum(reg, seg[:reg.shape[0], :reg.shape[1]], out=reg)

    for i, m in enumerate(line_masks):
        y = int(size * 0.5) + i * pitch + int(rng.integers(-2, 3))
        through = abs((y + m.shape[0] / 2) - hole_y) < hole_r + size * 0.6
        if not through:
            put(m, y, margin_x)
            continue
        cursor, src, mw = margin_x, 0, m.shape[1]
        for zl, zr in zones:
            if cursor >= zl:
                cursor = max(cursor, zr)
                continue
            if src + (zl - cursor) >= mw:
                break
            cut = _cut_before(m, src + (zl - cursor), src, 2 * size)
            put(m[:, src:cut], y, cursor)
            src, cursor = cut, zr
        if src < mw:
            put(m[:, src:], y, cursor)
    mask = E.roughen(mask, rng, rng.uniform(0.15, 0.35))
    mask = E.ink_dropout(mask, rng, amount=rng.uniform(0.0, 0.25), scale=rng.uniform(8, 20))
    # leaf body
    fib = E.palm_fibre(H, W, rng)
    hue = rng.uniform(0, 1)
    light = np.array([0.86, 0.72, 0.47]) * (1 - 0.15 * hue) + np.array([0.0, 0.02, 0.06]) * hue
    dark = light * rng.uniform(0.55, 0.7)
    leaf = E.tint(fib, dark, light)
    stains = E.fbm(H, W, rng, scale=140, octaves=3) > rng.uniform(0.7, 0.85)
    leaf = E.composite(leaf, ndi.gaussian_filter(stains.astype(np.float32), 6), (0.35, 0.25, 0.12), alpha=0.45)
    shape = np.zeros((H, W), np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    edge_noise_t = (E.fbm(1, W, rng, scale=60, octaves=3)[0] - 0.5) * size * 0.25
    edge_noise_b = (E.fbm(1, W, rng, scale=60, octaves=3)[0] - 0.5) * size * 0.25
    top = int(size * 0.15) + edge_noise_t
    bot = H - int(size * 0.15) + edge_noise_b
    inside = (yy >= top[None, :]) & (yy <= bot[None, :])
    end_r = int(size * rng.uniform(0.8, 1.6))      # rounded leaf ends
    cy0, half = H / 2, (H - size * 0.3) / 2
    for xs in (np.arange(0, end_r), W - 1 - np.arange(0, end_r)):
        t = (end_r - np.arange(0, end_r)) / end_r
        lim = half * np.sqrt(np.clip(1 - t ** 2, 0, 1))
        inside[:, xs] &= (np.abs(yy[:, xs] - cy0) <= lim[None, :])
    shape[inside] = 1.0
    ink_soft = ndi.gaussian_filter(mask, size * 0.3)
    for _ in range(int(rng.integers(0, 4))):  # chips and worm-holes, only where there is no writing
        for _try in range(30):
            cx = int(rng.integers(0, W))
            cy = int(_pick(rng, [top[cx], bot[cx], rng.integers(0, H)]))
            r = rng.uniform(0.2, 0.6) * size
            disk = ((yy - cy) ** 2 + (xx - cx) ** 2) < r * r
            if ink_soft[disk].max(initial=0) < 0.02:
                shape[disk] = 0
                break
    for hx in holes_x:
        shape[((yy - hole_y) ** 2 + (xx - hx) ** 2) < hole_r ** 2] = 0
    dist = ndi.distance_transform_edt(shape)
    leaf = leaf * (0.65 + 0.35 * np.clip(dist / (size * 0.6), 0, 1))[..., None]
    ink = (0.1, 0.07, 0.05)
    leaf = E.composite(leaf, _ink(mask) * shape, ink, alpha=rng.uniform(0.85, 1.0))
    bg_kind = _pick(rng, ["cloth", "white", "dark"])
    bgc = {"cloth": (0.22, 0.32, 0.55), "white": (0.92, 0.93, 0.95), "dark": (0.12, 0.12, 0.13)}[bg_kind]
    bg = E.tint(E.fbm(H, W, rng, scale=50), np.array(bgc) * 0.85, np.array(bgc) * 1.1)
    sm = ndi.gaussian_filter(shape, 0.8)[..., None]
    img = bg * (1 - sm) + leaf * sm
    img = E.bow(img, rng.uniform(-size * 0.25, size * 0.25))
    pil = E.to_image(img)
    out, meta = _finish_photo(pil, rng, persp=(0.0, 0.025), blur=(0.3, 0.9), quality=(75, 92),
                              bg=bgc, max_edge=2000)
    return out, {"font": font.id, "size": size, "background": bg_kind, **meta}


# ============================================================================ stone

def stone(lines, rng: Rng, style: str = "inscription", font: F.FontSpec | None = None,
          size: int | None = None, spacing: float = 1.25):
    text = "\n".join(lines)
    if font is None:
        if style == "inscription":
            font = F.BY_ID["lohit-classical"]
        else:
            font = _fonts_for(rng, ("tamil",), text, exclude=("nolatin", "hand", "classical", "rounded", "light"))
    size = size or int(rng.integers(34, 54))
    r = render_block(lines, font, size, line_spacing=spacing, align="left" if style in ("inscription", "cave") else "center",
                     margin=int(size * 1.0), jitter_px=int(size * 0.08) if style in ("inscription", "cave") else 0, rng=rng)
    m = r.mask
    if style in ("inscription", "cave"):
        m = E.elastic(m, rng, alpha=rng.uniform(0.5, 1.5), sigma=rng.uniform(4, 8))
        m = E.roughen(E.thicken(m, 1) if style == "inscription" else m, rng, 0.3)
        m = E.ink_dropout(m, rng, amount=rng.uniform(0.0, 0.3), scale=rng.uniform(4, 10))
    m = _ink(m)
    h, w = m.shape
    if style == "plaque-black":
        base = E.granite(h, w, rng, base=(0.1, 0.1, 0.11), contrast=0.08, grain=0.8)
        base = E.relief(base, m, depth=0.6, blur=1.0, groove_dark=0.0)
        paint = _pick(rng, [(0.85, 0.7, 0.35), (0.92, 0.92, 0.9)])
        img = E.composite(base, E.gaussian_blur(m, 0.5), paint, alpha=rng.uniform(0.8, 0.95))
        streak = np.clip(1 - np.abs(np.linspace(-1, 1, w)[None, :] * 3 - rng.uniform(-1, 1)), 0, 1)
        img = np.clip(img + 0.12 * streak[..., None], 0, 1)
    elif style == "plaque-grey":
        base = E.granite(h, w, rng, base=_pick(rng, [(0.6, 0.58, 0.55), (0.7, 0.62, 0.6)]), contrast=0.12)
        img = E.relief(base, m, depth=1.2, blur=1.4, groove_dark=0.35)
    else:  # weathered temple wall or cave brow
        tone = (0.62, 0.52, 0.42) if style == "cave" else _pick(rng, [(0.55, 0.52, 0.48), (0.6, 0.5, 0.42), (0.45, 0.44, 0.42)])
        base = E.granite(h, w, rng, base=tone, contrast=0.35, grain=0.7)
        if style == "inscription" and rng.random() < 0.7:   # dressed stone blocks
            joints = np.zeros((h, w), np.float32)
            for y in np.linspace(0, h, int(rng.integers(2, 4)))[1:-1]:
                joints[int(y):int(y) + 2, :] = 1
            for x in np.linspace(0, w, int(rng.integers(2, 5)))[1:-1]:
                joints[:, int(x):int(x) + 2] = 1
            base = E.composite(base, E.gaussian_blur(joints, 1.0), (0.2, 0.18, 0.16), alpha=0.7)
        if style == "cave":   # carved drip-line above the record, typical of Jaina cave beds
            drip = np.zeros((h, w), np.float32)
            y0 = max(2, int(size * 0.3))
            drip[y0:y0 + max(3, size // 6), int(w * 0.02):int(w * 0.98)] = 1
            base = E.relief(base, E.warp_rows(drip, size * 0.15, w * 1.5), depth=1.5, blur=2.0, groove_dark=0.2)
        img = E.relief(base, m, depth=rng.uniform(1.0, 2.2), blur=rng.uniform(1.2, 2.2),
                       light=(rng.uniform(-1, -0.3), rng.uniform(-1, -0.3)), groove_dark=rng.uniform(0.15, 0.4))
        lichen = E.fbm(h, w, rng, scale=70, octaves=4)
        img = E.composite(img, np.clip((lichen - 0.72) * 4, 0, 1), _pick(rng, [(0.45, 0.5, 0.3), (0.75, 0.55, 0.25), (0.3, 0.3, 0.28)]),
                          alpha=rng.uniform(0.2, 0.55))
    out, meta = _finish_photo(E.to_image(img), rng, persp=(0.02, 0.09), blur=(0.4, 1.3),
                              bg=_pick(rng, [(0.45, 0.42, 0.38), (0.3, 0.3, 0.3)]))
    return out, {"style": style, "font": font.id, "size": size, **meta}


def estampage(lines, rng: Rng, font: F.FontSpec | None = None, size: int | None = None, spacing: float = 1.25):
    font = font or F.BY_ID["lohit-classical"]
    size = size or int(rng.integers(34, 52))
    r = render_block(lines, font, size, line_spacing=spacing, margin=int(size * 1.2),
                     jitter_px=int(size * 0.08), rng=rng)
    m = _ink(E.roughen(E.thicken(E.elastic(r.mask, rng, alpha=1.0, sigma=6), 1), rng, 0.3))
    h, w = m.shape
    coverage = 0.75 + 0.25 * E.fbm(h, w, rng, scale=25, octaves=4)
    ink = coverage * (1 - 0.92 * E.gaussian_blur(m, rng.uniform(0.8, 1.8)))
    grain = E.granite(h, w, rng, base=(1, 1, 1), contrast=0.2, grain=0.8)[..., 0]
    ink = np.clip(ink * (0.85 + 0.25 * grain), 0, 1)
    pap = E.paper(h, w, rng, base=(0.93, 0.92, 0.88))
    img = E.composite(pap, ink, (0.06, 0.06, 0.07), alpha=rng.uniform(0.8, 0.97))
    for _ in range(int(rng.integers(0, 3))):   # folds
        fold = np.zeros((h, w), np.float32)
        if rng.random() < 0.5:
            x = int(rng.integers(w // 5, 4 * w // 5))
            fold[:, x:x + 2] = 1
        else:
            y = int(rng.integers(h // 5, 4 * h // 5))
            fold[y:y + 2, :] = 1
        img = E.composite(img, E.gaussian_blur(fold, 1.5), (0.95, 0.95, 0.95), alpha=0.5)
    out, meta = _finish_scan(img, rng, skew=2.0, gray_p=0.6)
    return out, {"style": "estampage", "font": font.id, "size": size, **meta}


# ============================================================================ copper plate

def copper_plate(lines, rng: Rng, style: str = "grant", font: F.FontSpec | None = None, size: int | None = None):
    text = "\n".join(lines)
    if font is None:
        font = F.BY_ID["lohit-classical"] if style == "grant" else _fonts_for(
            rng, ("tamil",), text, exclude=("nolatin", "hand", "classical", "rounded", "light"))
    size = size or int(rng.integers(30, 46))
    r = render_block(lines, font, size, line_spacing=1.15 if style == "grant" else 1.3,
                     align="left" if style == "grant" else "center", margin=int(size * 1.0),
                     jitter_px=int(size * 0.05) if style == "grant" else 0, rng=rng)
    m = r.mask
    if style == "grant":
        m = E.roughen(E.elastic(m, rng, alpha=0.8, sigma=6), rng, 0.25)
    h, w = m.shape
    ring = style == "grant" and rng.random() < 0.8
    left_pad = int(size * 2.6) if ring else 0
    m = _ink(E.pad_to(m, h, w + left_pad, 0, left_pad))
    h, w = m.shape
    if style == "grant":
        cu = E.tint(E.fbm(h, w, rng, scale=8, octaves=2, aspect=6), (0.48, 0.25, 0.14), (0.78, 0.47, 0.28))
    else:
        cu = E.tint(E.fbm(h, w, rng, scale=8, octaves=2, aspect=10), (0.62, 0.48, 0.2), (0.88, 0.74, 0.38))
    spec = np.clip(1 - np.abs(np.linspace(-1, 1, w)[None, :] * 2 - rng.uniform(-0.8, 0.8)), 0, 1)
    cu = np.clip(cu * (0.85 + 0.3 * spec[..., None]), 0, 1)
    if style == "grant":
        rim = np.zeros((h, w), np.float32)
        t = max(4, size // 5)
        rim[:t, :] = rim[-t:, :] = 1
        rim[:, :t] = rim[:, -t:] = 1
        cu = E.relief(cu, rim, depth=1.2, blur=2.0, groove_dark=0.0, incised=False)
        patina = np.clip((E.fbm(h, w, rng, scale=60, octaves=4) - rng.uniform(0.55, 0.75)) * 3, 0, 1)
        cu = E.composite(cu, patina, (0.3, 0.52, 0.42), alpha=rng.uniform(0.4, 0.8))
    img = E.relief(cu, m, depth=rng.uniform(0.8, 1.6), blur=1.0, groove_dark=0.55 if style == "grant" else 0.75)
    shape = np.ones((h, w), np.float32)
    if ring:
        yy, xx = np.mgrid[0:h, 0:w]
        rr = size * 0.75
        shape[((yy - h / 2) ** 2 + (xx - left_pad * 0.5) ** 2) < rr * rr] = 0
    bgc = _pick(rng, [(0.85, 0.85, 0.85), (0.15, 0.15, 0.17), (0.4, 0.33, 0.25)])
    bg = E.solid(h, w, bgc)
    sm = ndi.gaussian_filter(shape, 0.8)[..., None]
    img = bg * (1 - sm) + img * sm
    out, meta = _finish_photo(E.to_image(img), rng, persp=(0.02, 0.08), blur=(0.3, 1.0), bg=bgc)
    return out, {"style": style, "font": font.id, "size": size, "ring": ring, **meta}


# ============================================================================ pottery

def pottery(lines, rng: Rng, font: F.FontSpec | None = None, size: int | None = None):
    font = font or F.BY_ID["brahmi"]
    size = size or int(rng.integers(40, 60))
    r = render_block(lines, font, size, line_spacing=1.1, margin=int(size * 0.5))
    m = E.elastic(r.mask, rng, alpha=1.5, sigma=5)
    m = E.roughen(E.thin(m, 1), rng, 0.4)
    mh, mw = m.shape
    H, W = int(mh * rng.uniform(2.0, 2.6)), int(mw * rng.uniform(1.3, 1.6))
    m = _ink(E.pad_to(m, H, W, (H - mh) // 2, (W - mw) // 2))
    cy, cx = H / 2, W / 2
    n = int(rng.integers(6, 10))
    ang = np.sort(rng.uniform(0, 2 * np.pi, n))
    pts = [(cx + np.cos(a) * W * 0.5 * rng.uniform(0.85, 1.0), cy + np.sin(a) * H * 0.5 * rng.uniform(0.8, 1.0)) for a in ang]
    poly = Image.new("L", (W, H), 0)
    ImageDraw.Draw(poly).polygon(pts, fill=255)
    poly = poly.filter(ImageFilter.GaussianBlur(2))
    shape = np.asarray(poly, np.float32) / 255
    ware = _pick(rng, ["red", "red", "black-and-red"])
    if ware == "red":
        clay = E.tint(E.fbm(H, W, rng, scale=20), (0.5, 0.25, 0.15), (0.72, 0.42, 0.26))
    else:
        g = np.linspace(0, 1, H)[:, None, None]
        clay = (1 - g) * np.array([0.12, 0.1, 0.1]) + g * np.array([0.62, 0.32, 0.2])
        clay = clay * (0.85 + 0.3 * E.fbm(H, W, rng, scale=20)[..., None])
    grit = (rng.random((H, W)) > 0.995).astype(np.float32)
    clay = E.composite(clay, grit, (0.9, 0.85, 0.75), alpha=0.6)
    yy, xx = np.mgrid[0:H, 0:W]
    curv = 1 - 0.35 * (((xx - cx) / (W / 2)) ** 2)
    clay = clay * curv[..., None]
    img = E.composite(clay, m * shape, (0.88, 0.75, 0.6), alpha=rng.uniform(0.6, 0.9))
    bgc = _pick(rng, [(0.9, 0.9, 0.88), (0.55, 0.55, 0.55), (0.2, 0.2, 0.22)])
    sm = shape[..., None]
    img = E.solid(H, W, bgc) * (1 - sm) + img * sm
    out, meta = _finish_photo(E.to_image(np.clip(img, 0, 1)), rng, persp=(0.0, 0.05), blur=(0.3, 1.0), bg=bgc)
    return out, {"ware": ware, "font": font.id, "size": size, **meta}


MEDIA = {
    "born-digital": born_digital,
    "print-scan": print_scan,
    "print-photo": print_photo,
    "handwritten": handwriting,
    "scene": scene,
    "palm-leaf": palm_leaf,
    "stone": stone,
    "estampage": estampage,
    "copper-plate": copper_plate,
    "pottery": pottery,
}
