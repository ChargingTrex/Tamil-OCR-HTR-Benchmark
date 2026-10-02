"""Screenshots rendered by a real browser engine (Chromium via Playwright).

Each template builds a small HTML page (news card, chat, settings list, subtitle over a
video frame, form, table, notification). The ground truth is the element's rendered
``innerText`` — i.e. exactly what the browser drew, in DOM (= visual) order.
"""

from __future__ import annotations

import base64
import html
import io
import os
import random

import numpy as np

from . import effects as E
from .fonts import FONT_DIR

CHROMIUM = os.environ.get("TAMILBENCH_CHROMIUM", "/opt/pw-browsers/chromium")

FAMILIES = {  # css family -> file
    "TBNotoSans": "NotoSansTamil-VF.ttf", "TBNotoSerif": "NotoSerifTamil-VF.ttf",
    "TBMukta": "MuktaMalar-Regular.ttf", "TBMuktaBold": "MuktaMalar-Bold.ttf",
    "TBHind": "HindMadurai-Regular.ttf", "TBCatamaran": "Catamaran-VF.ttf",
    "TBAnek": "AnekTamil-VF.ttf", "TBNotoUI": "NotoSansTamilUI-VF.ttf",
    "TBMeera": "MeeraInimai-Regular.ttf", "TBTiro": "TiroTamil-Regular.ttf",
}
UI_FAMILIES = ["TBNotoSans", "TBMukta", "TBHind", "TBCatamaran", "TBAnek", "TBNotoUI", "TBMeera"]
NEWS_FAMILIES = ["TBNotoSerif", "TBTiro", "TBNotoSans", "TBMukta", "TBHind"]


def _font_faces() -> str:
    return "\n".join(f"@font-face{{font-family:'{fam}';src:url('{(FONT_DIR / f).as_uri()}');}}"
                     for fam, f in FAMILIES.items())


def _theme(rng: random.Random):
    if rng.random() < 0.35:
        return {"bg": "#121212", "card": "#1e1f22", "fg": "#e8e8e8", "muted": "#9aa0a6",
                "accent": rng.choice(["#8ab4f8", "#81c995", "#f28b82"]), "bubble1": "#005c4b", "bubble2": "#202c33", "dark": True}
    return {"bg": rng.choice(["#ffffff", "#f5f5f5", "#eef1f5"]), "card": "#ffffff", "fg": "#1f1f1f",
            "muted": "#5f6368", "accent": rng.choice(["#1a73e8", "#0b8043", "#c5221f", "#6a1b9a"]),
            "bubble1": "#d9fdd3", "bubble2": "#ffffff", "dark": False}


def _esc(s: str) -> str:
    return html.escape(s, quote=False)


def _frame_data_uri(rng: random.Random, w: int, h: int) -> str:
    g = np.random.default_rng(rng.randrange(1 << 30))
    base = E.tint(E.fbm(h, w, g, scale=120, octaves=4), g.uniform(0, .5, 3), g.uniform(.3, 1, 3))
    img = E.to_image(E.gaussian_blur(base, 3))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build_html(template: str, data: dict, rng: random.Random) -> tuple[str, dict]:
    t = _theme(rng)
    fam = rng.choice(NEWS_FAMILIES if template == "news" else UI_FAMILIES)
    fs = rng.randint(15, 22)
    width = rng.choice([360, 390, 412]) if rng.random() < 0.6 else rng.choice([640, 800, 960])
    css = f"""{_font_faces()}
    *{{box-sizing:border-box}} body{{margin:0;background:{t['bg']};font-family:'{fam}',sans-serif;
    font-size:{fs}px;color:{t['fg']};}} #root{{width:{width}px;padding:14px;background:{t['bg']}}}
    .card{{background:{t['card']};border-radius:10px;padding:12px 14px;box-shadow:0 1px 3px rgba(0,0,0,.2)}}
    .muted{{color:{t['muted']};font-size:.8em}} .accent{{color:{t['accent']}}}"""
    if template == "news":
        body = (f"<div class='card'><div class='muted'>{_esc(data['kicker'])}</div>"
                f"<h2 style='margin:.3em 0;font-size:1.35em;line-height:1.35'>{_esc(data['headline'])}</h2>"
                f"<p style='margin:0;line-height:1.6'>{_esc(data['body'])}</p></div>")
    elif template == "chat":
        rows = []
        for i, (msg, tm) in enumerate(data["messages"]):
            mine = i % 2 == 0
            rows.append(f"<div style='display:flex;justify-content:{'flex-end' if mine else 'flex-start'};margin:6px 0'>"
                        f"<div style='max-width:78%;background:{t['bubble1'] if mine else t['bubble2']};"
                        f"border-radius:12px;padding:7px 10px 5px'><div>{_esc(msg)}</div>"
                        f"<div class='muted' style='text-align:right'>{_esc(tm)}</div></div></div>")
        body = f"<div>{''.join(rows)}</div>"
    elif template == "settings":
        items = "".join(f"<div style='display:flex;align-items:center;padding:11px 4px;border-bottom:1px solid "
                        f"{t['muted']}33'><span style='width:22px;height:22px;border-radius:6px;background:"
                        f"{t['accent']};opacity:.8;margin-right:14px;flex:none'></span><div>{_esc(x)}</div></div>"
                        for x in data["items"])
        body = (f"<div class='card'><div style='font-weight:700;font-size:1.25em;margin-bottom:6px'>"
                f"{_esc(data['title'])}</div>{items}</div>")
    elif template == "subtitle":
        h = int(width * 9 / 16)
        uri = _frame_data_uri(rng, width, h)
        body = (f"<div style='position:relative;width:{width - 28}px;height:{h}px;background:url({uri}) center/cover'>"
                f"<div style='position:absolute;left:5%;right:5%;bottom:7%;text-align:center;color:#fff;"
                f"font-size:{fs + 4}px;line-height:1.4;text-shadow:0 0 3px #000,0 0 3px #000,1px 1px 2px #000'>"
                f"{_esc(data['subtitle'])}</div></div>")
    elif template == "form":
        rows = "".join(f"<div style='margin:8px 0'><div class='muted' style='font-size:.85em'>{_esc(k)}</div>"
                       f"<div style='border:1px solid {t['muted']};border-radius:6px;padding:7px 9px;"
                       f"margin-top:3px'>{_esc(v)}</div></div>" for k, v in data["fields"])
        body = f"<div class='card'><div style='font-weight:700;margin-bottom:4px'>{_esc(data['title'])}</div>{rows}</div>"
    elif template == "table":
        head = "".join(f"<th style='text-align:left;padding:6px 8px;border-bottom:2px solid {t['muted']}'>{_esc(c)}</th>"
                       for c in data["header"])
        rows = "".join("<tr>" + "".join(f"<td style='padding:6px 8px;border-bottom:1px solid {t['muted']}44'>{_esc(c)}</td>"
                                        for c in r) + "</tr>" for r in data["rows"])
        body = f"<div class='card'><table style='border-collapse:collapse;width:100%'><tr>{head}</tr>{rows}</table></div>"
    elif template == "notification":
        body = (f"<div class='card' style='border-radius:16px'><div class='muted'>{_esc(data['app'])} · "
                f"{_esc(data['time'])}</div><div style='font-weight:700;margin:3px 0'>{_esc(data['title'])}</div>"
                f"<div>{_esc(data['body'])}</div></div>")
    else:
        raise ValueError(template)
    page = f"<!doctype html><html lang='ta'><head><meta charset='utf-8'><style>{css}</style></head><body><div id='root'>{body}</div></body></html>"
    return page, {"template": template, "font_family": fam, "font_px": fs, "width": width,
                  "theme": "dark" if t["dark"] else "light"}


class ScreenRenderer:
    """Keeps one headless Chromium alive across many screenshots."""

    def __init__(self, executable: str | None = None):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        exe = executable or (CHROMIUM if os.path.exists(CHROMIUM) else None)
        self._browser = self._pw.chromium.launch(executable_path=exe) if exe else self._pw.chromium.launch()

    def render(self, template: str, data: dict, rng: random.Random):
        page_html, meta = build_html(template, data, rng)
        dpr = rng.choice([1, 1, 2])
        page = self._browser.new_page(viewport={"width": meta["width"] + 40, "height": 900}, device_scale_factor=dpr)
        try:
            page.set_content(page_html, wait_until="load")
            page.evaluate("document.fonts.ready")
            root = page.locator("#root")
            text = root.inner_text()
            png = root.screenshot(type="png")
        finally:
            page.close()
        from PIL import Image
        img = Image.open(io.BytesIO(png)).convert("RGB")
        meta["dpr"] = dpr
        r = rng.random()
        if r < 0.3:   # forwarded / re-compressed screenshot (never below ~13 px text)
            f = rng.uniform(0.5, 0.85) if dpr == 2 else rng.uniform(0.82, 0.95)
            img = E.jpeg_roundtrip(E.rescale(img, f), rng.randint(55, 85))
            meta["recompressed"] = round(f, 2)
        elif r < 0.4:  # low-DPI capture upscaled for display
            f = rng.uniform(0.55, 0.8) if dpr == 2 else rng.uniform(0.75, 0.9)
            img = E.rescale(E.rescale(img, f), 1 / f)
            meta["upscaled_from"] = round(f, 2)
        text = "\n".join(" ".join(ln.split()) for ln in text.splitlines() if ln.strip())
        return img, text, meta

    def close(self):
        self._browser.close()
        self._pw.stop()
