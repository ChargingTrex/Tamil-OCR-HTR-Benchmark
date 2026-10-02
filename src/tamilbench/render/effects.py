"""Procedural textures and image degradations (numpy + scipy; deterministic given an RNG).

Conventions: colour images are float32 arrays in [0, 1] with shape (H, W, 3); masks are
float32 (H, W) in [0, 1] where 1 means ink / carved / painted.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

Rng = np.random.Generator


# --------------------------------------------------------------------------- noise & textures

def fbm(h: int, w: int, rng: Rng, *, scale: float = 64.0, octaves: int = 4,
        persistence: float = 0.5, aspect: float = 1.0) -> np.ndarray:
    """Fractal value noise in [0, 1]. ``aspect`` > 1 stretches features horizontally."""
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    s = scale
    for _ in range(octaves):
        gh = max(2, int(h / s) + 2)
        gw = max(2, int(w / (s * aspect)) + 2)
        grid = rng.random((gh, gw)).astype(np.float32)
        layer = np.asarray(Image.fromarray(grid).resize((w, h), Image.BICUBIC), np.float32)
        out += amp * layer
        total += amp
        amp *= persistence
        s /= 2.0
    out /= total
    lo, hi = np.percentile(out, [1, 99])
    return np.clip((out - lo) / max(1e-6, hi - lo), 0, 1)


def solid(h: int, w: int, rgb) -> np.ndarray:
    return np.ones((h, w, 3), np.float32) * np.asarray(rgb, np.float32)


def tint(gray: np.ndarray, rgb_lo, rgb_hi) -> np.ndarray:
    lo, hi = np.asarray(rgb_lo, np.float32), np.asarray(rgb_hi, np.float32)
    return lo + gray[..., None] * (hi - lo)


def paper(h: int, w: int, rng: Rng, *, base=(0.97, 0.96, 0.93), aged: float = 0.0) -> np.ndarray:
    n = fbm(h, w, rng, scale=120, octaves=5)
    fine = rng.normal(0, 0.012, (h, w)).astype(np.float32)
    img = solid(h, w, base) * (0.97 + 0.05 * n[..., None]) + fine[..., None]
    if aged > 0:
        yellow = np.array([0.93, 0.83, 0.62], np.float32)
        img = img * (1 - aged * 0.55) + yellow * aged * 0.55 * (0.9 + 0.1 * n[..., None])
        img = foxing(img, rng, density=aged)
        img = edge_darken(img, strength=0.25 * aged)
    return np.clip(img, 0, 1)


def foxing(img: np.ndarray, rng: Rng, density: float = 0.5) -> np.ndarray:
    h, w = img.shape[:2]
    spots = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(3, 12) * density) + 1):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        r = rng.uniform(2, 14) * (1 + density)
        yy, xx = np.ogrid[:h, :w]
        spots += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r)) * rng.uniform(0.2, 0.6)
    brown = np.array([0.55, 0.38, 0.2], np.float32)
    a = np.clip(spots, 0, 0.7)[..., None]
    return img * (1 - a) + brown * a


def edge_darken(img: np.ndarray, strength: float = 0.2) -> np.ndarray:
    h, w = img.shape[:2]
    yy = np.linspace(-1, 1, h)[:, None]
    xx = np.linspace(-1, 1, w)[None, :]
    d = np.clip(np.maximum(np.abs(yy), np.abs(xx)) - 0.75, 0, None) / 0.25
    return img * (1 - strength * d[..., None])


def palm_fibre(h: int, w: int, rng: Rng) -> np.ndarray:
    """Horizontal striations of a palm leaf, in [0, 1]."""
    rows = fbm(h, 64, rng, scale=6, octaves=3)
    streak = np.asarray(Image.fromarray(rows).resize((w, h), Image.BICUBIC), np.float32)
    blotch = fbm(h, w, rng, scale=90, octaves=4, aspect=4.0)
    fine = fbm(h, w, rng, scale=3, octaves=2, aspect=12.0)
    return np.clip(0.45 * streak + 0.35 * blotch + 0.2 * fine, 0, 1)


def granite(h: int, w: int, rng: Rng, *, base=(0.55, 0.53, 0.5), contrast: float = 0.25,
            grain: float = 0.5) -> np.ndarray:
    low = fbm(h, w, rng, scale=160, octaves=4)
    mid = fbm(h, w, rng, scale=12, octaves=3)
    g = rng.random((h, w)).astype(np.float32)
    dark = (g < 0.04 * grain).astype(np.float32)
    light = (g > 1 - 0.03 * grain).astype(np.float32)
    dark = ndi.gaussian_filter(dark, 0.7)
    light = ndi.gaussian_filter(light, 0.7)
    v = 0.8 + contrast * (low - 0.5) + 0.12 * (mid - 0.5) - 1.6 * dark + 1.0 * light
    return np.clip(solid(h, w, base) * v[..., None], 0, 1)


# --------------------------------------------------------------------------- ink & relief

def composite(bg: np.ndarray, mask: np.ndarray, color, alpha: float = 1.0) -> np.ndarray:
    a = np.clip(mask * alpha, 0, 1)[..., None]
    return bg * (1 - a) + np.asarray(color, np.float32) * a


def ink_spread(mask: np.ndarray, sigma: float = 0.6, gain: float = 1.4) -> np.ndarray:
    return np.clip(ndi.gaussian_filter(mask, sigma) * gain, 0, 1)


def ink_dropout(mask: np.ndarray, rng: Rng, amount: float = 0.3, scale: float = 6.0) -> np.ndarray:
    """Patchy fading (worn type, faded ink, weathering). Ink never drops below 35 % so every
    letter stays legible — damage that destroys letters would make the ground truth unfair."""
    n = fbm(*mask.shape, rng, scale=scale, octaves=2)
    keep = np.clip((n - amount * 0.6) / max(1e-3, 1 - amount * 0.6), 0, 1) ** 0.5
    return mask * (0.35 + 0.65 * keep)


def thin(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Erode strokes (stylus-incised and scratched lines are thinner than type)."""
    out = mask.astype(np.float32)
    for _ in range(iterations):
        out = ndi.grey_erosion(out, size=(2, 2))
    return out


def thicken(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    return ndi.grey_dilation(mask, size=(2 * radius + 1, 2 * radius + 1))


def roughen(mask: np.ndarray, rng: Rng, amount: float = 0.35) -> np.ndarray:
    """Irregular, eroded stroke edges (chisel, stylus, worn type) that keep thin strokes alive:
    the softened mask is re-thresholded at 0.5 with a noise-perturbed threshold."""
    n = fbm(*mask.shape, rng, scale=2.5, octaves=2)
    soft = ndi.gaussian_filter(mask.astype(np.float32), 0.6)
    return np.clip((soft - 0.5 + amount * (n - 0.5)) * 4.0 + 0.5, 0, 1) * (soft > 0.05)


def relief(img: np.ndarray, mask: np.ndarray, *, depth: float = 1.0, blur: float = 1.6,
           light=(-0.7, -0.7), groove_dark: float = 0.45, incised: bool = True) -> np.ndarray:
    """Shade an engraved (incised) or raised mask under a raking light."""
    h = ndi.gaussian_filter(mask.astype(np.float32), blur)
    if incised:
        h = -h
    gy, gx = np.gradient(h)
    lx, ly = light
    shade = (gx * lx + gy * ly) * depth * 18.0
    out = img * (1 + shade[..., None])
    if incised:
        out = out * (1 - groove_dark * ndi.gaussian_filter(mask, 0.6))[..., None]
    return np.clip(out, 0, 1)


# --------------------------------------------------------------------------- geometry

def pad_to(arr: np.ndarray, h: int, w: int, y: int, x: int) -> np.ndarray:
    out = np.zeros((h, w) + arr.shape[2:], arr.dtype)
    hh, ww = min(arr.shape[0], h - y), min(arr.shape[1], w - x)
    out[y:y + hh, x:x + ww] = arr[:hh, :ww]
    return out


def warp_rows(arr: np.ndarray, amplitude: float, period: float, phase: float = 0.0) -> np.ndarray:
    """Bend horizontal lines into a sine (curved baselines, sagging leaves)."""
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dy = amplitude * np.sin(2 * np.pi * xx / period + phase)
    return _remap(arr, yy + dy, xx)


def bow(arr: np.ndarray, amplitude: float) -> np.ndarray:
    """Parabolic sag (a leaf or page bending under its own weight)."""
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = (xx / max(1, w - 1)) * 2 - 1
    return _remap(arr, yy - amplitude * (1 - t * t), xx)


def elastic(arr: np.ndarray, rng: Rng, alpha: float = 4.0, sigma: float = 8.0) -> np.ndarray:
    h, w = arr.shape[:2]
    dx = ndi.gaussian_filter(rng.uniform(-1, 1, (h, w)).astype(np.float32), sigma) * alpha * sigma
    dy = ndi.gaussian_filter(rng.uniform(-1, 1, (h, w)).astype(np.float32), sigma) * alpha * sigma
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return _remap(arr, yy + dy, xx + dx)


def shear(arr: np.ndarray, slant: float) -> np.ndarray:
    """Slant a mask: positive leans right (top shifted right). Output is widened to fit."""
    h, w = arr.shape[:2]
    extra = int(abs(slant) * (h - 1)) + 1
    min_off = min(0.0, slant * (h - 1))
    yy, xx = np.mgrid[0:h, 0:w + extra].astype(np.float32)
    src_x = xx + min_off - slant * (h - 1 - yy)
    return _remap(arr, yy, src_x)


def _remap(arr: np.ndarray, yy: np.ndarray, xx: np.ndarray, out_shape=None) -> np.ndarray:
    coords = np.stack([yy, xx])
    if arr.ndim == 2:
        return ndi.map_coordinates(arr, coords, order=1, mode="constant", cval=0.0).astype(np.float32)
    chans = [ndi.map_coordinates(arr[..., c], coords, order=1, mode="nearest") for c in range(arr.shape[2])]
    return np.stack(chans, -1).astype(np.float32)


def perspective(img: Image.Image, rng: Rng, strength: float = 0.08, fill=(0, 0, 0)) -> Image.Image:
    w, h = img.size
    d = strength
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [(rng.uniform(0, d) * w, rng.uniform(0, d) * h), (w - rng.uniform(0, d) * w, rng.uniform(0, d) * h),
           (w - rng.uniform(0, d) * w, h - rng.uniform(0, d) * h), (rng.uniform(0, d) * w, h - rng.uniform(0, d) * h)]
    coeffs = _persp_coeffs(dst, src)
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor=fill)


def _persp_coeffs(pa, pb):
    A = []
    for (x, y), (u, v) in zip(pa, pb):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    A = np.array(A, dtype=np.float64)
    b = np.array([c for p in pb for c in p], dtype=np.float64)
    return np.linalg.solve(A, b).tolist()


def rotate(img: Image.Image, deg: float, fill=(255, 255, 255)) -> Image.Image:
    return img.rotate(deg, resample=Image.BICUBIC, expand=True, fillcolor=fill)


# --------------------------------------------------------------------------- photometric

def light_gradient(img: np.ndarray, rng: Rng, strength: float = 0.3) -> np.ndarray:
    h, w = img.shape[:2]
    ang = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = (np.cos(ang) * xx / w + np.sin(ang) * yy / h)
    g = (g - g.min()) / max(1e-6, g.max() - g.min())
    return np.clip(img * (1 - strength + strength * 1.2 * g[..., None]), 0, 1)


def shadow_band(img: np.ndarray, rng: Rng, strength: float = 0.35) -> np.ndarray:
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, np.pi)
    c = rng.uniform(0.2, 0.8)
    d = (np.cos(ang) * xx / w + np.sin(ang) * yy / h) - c
    soft = rng.uniform(0.03, 0.12)
    m = 1 / (1 + np.exp(-d / soft))
    return np.clip(img * (1 - strength * m[..., None]), 0, 1)


def vignette(img: np.ndarray, strength: float = 0.3) -> np.ndarray:
    h, w = img.shape[:2]
    yy = np.linspace(-1, 1, h)[:, None]
    xx = np.linspace(-1, 1, w)[None, :]
    r = np.sqrt(xx ** 2 + yy ** 2) / np.sqrt(2)
    return np.clip(img * (1 - strength * r[..., None] ** 2), 0, 1)


def color_cast(img: np.ndarray, rng: Rng, strength: float = 0.06) -> np.ndarray:
    return np.clip(img * (1 + rng.uniform(-strength, strength, 3).astype(np.float32)), 0, 1)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    s = (sigma, sigma, 0) if img.ndim == 3 else sigma
    return ndi.gaussian_filter(img, s)


def motion_blur(img: np.ndarray, length: int, angle_deg: float) -> np.ndarray:
    if length <= 1:
        return img
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1
    k = ndi.rotate(k, angle_deg, reshape=False, order=1)
    k /= max(1e-6, k.sum())
    if img.ndim == 3:
        return np.stack([ndi.convolve(img[..., c], k, mode="nearest") for c in range(3)], -1)
    return ndi.convolve(img, k, mode="nearest")


def noise(img: np.ndarray, rng: Rng, sigma: float = 0.02) -> np.ndarray:
    return np.clip(img + rng.normal(0, sigma, img.shape).astype(np.float32), 0, 1)


def salt_pepper(img: np.ndarray, rng: Rng, amount: float = 0.002) -> np.ndarray:
    out = img.copy()
    m = rng.random(img.shape[:2])
    out[m < amount / 2] = 0
    out[m > 1 - amount / 2] = 1
    return out


def to_image(arr: np.ndarray) -> Image.Image:
    a = (np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)
    return Image.fromarray(a, "RGB" if a.ndim == 3 else "L")


def from_image(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), np.float32) / 255.0


def jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def rescale(img: Image.Image, factor: float, resample=Image.BILINEAR) -> Image.Image:
    w, h = img.size
    return img.resize((max(1, int(w * factor)), max(1, int(h * factor))), resample)


def fit_long_edge(img: Image.Image, max_edge: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_edge:
        return img
    f = max_edge / max(w, h)
    return img.resize((int(w * f), int(h * f)), Image.LANCZOS)
