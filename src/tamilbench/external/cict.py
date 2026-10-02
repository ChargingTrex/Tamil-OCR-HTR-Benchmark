"""Import the real palm-leaf subset from the CICT Tirukkural Ground Truth Corpus.

Source: Central Institute of Classical Tamil (CICT), *CICT Tirukkural Ground Truth Corpus*,
CC BY 4.0, one Zenodo DOI per leaf. Expected input is a ``cict-htr`` directory as produced by
https://github.com/ChargingTrex/tamil-ocr (``data/leaves.jsonl``, ``data/lines.jsonl``,
``data/splits.json``, ``images/*.jpg`` and ``out/geometry/*.json`` with measured leaf boxes).

What this does per leaf
  * crops to the measured leaf box (plus a small margin);
  * paints out the digitiser's red caption (chapter name and verse numbers) — it is not part
    of the manuscript and would let a model recite the canonical text instead of reading;
  * attaches the 10 lines of CICT's *diplomatic* transcription (body text + the wrapped final
    syllable, i.e. ``text_reading``), in the manuscript's own line order.

Only leaves in the source's held-out ``val`` and ``test`` splits are used, so models trained
on the CICT ``train`` split can be evaluated fairly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

SUBSET = "palm-leaf-cict"
ATTRIBUTION = ("Central Institute of Classical Tamil (CICT), CICT Tirukkural Ground Truth Corpus, "
               "CC BY 4.0. Data curator: Kannan Krishnan (CICT).")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mask_caption(arr: np.ndarray) -> tuple[np.ndarray, bool]:
    """Paint out red caption text (and its white box) in the top band of a leaf crop."""
    h, w, _ = arr.shape
    band = int(h * 0.3)
    r, g, b = (arr[:band, :, i].astype(int) for i in range(3))
    red = (r > 140) & (g < 125) & (b < 125) & (r - g > 55)
    if red.sum() < 30:
        return arr, False
    ys, xs = np.where(red)
    y0, y1 = max(0, ys.min() - 25), min(h, ys.max() + 25)
    x0, x1 = max(0, xs.min() - 40), min(w, xs.max() + 40)
    region = arr[y0:y1, x0:x1].astype(int)
    white = (region.min(axis=2) > 200)
    rr, gg, bb = region[..., 0], region[..., 1], region[..., 2]
    reddish = (rr > 120) & (rr - gg > 35) & (rr - bb > 35)
    paint = white | reddish
    # background colour: bluish / light pixels outside the leaf in the same band, else neutral
    full = arr[:band].astype(int)
    bgsel = (full[..., 2] > full[..., 0] + 10) & ~red
    bg = np.median(full[bgsel], axis=0) if bgsel.sum() > 50 else np.array([215, 222, 230])
    out = arr.copy()
    sub = out[y0:y1, x0:x1]
    sub[paint] = bg.astype(np.uint8)
    out[y0:y1, x0:x1] = sub
    return out, True


def build_subset(root: Path, out_dir: Path, *, split: str = "test", limit: int | None = None,
                 use_splits: tuple[str, ...] = ("val", "test")) -> list[dict]:
    root = Path(root)
    leaves = {L["specimen"]: L for L in _jsonl(root / "data" / "leaves.jsonl")}
    lines = _jsonl(root / "data" / "lines.jsonl")
    assignment = json.loads((root / "data" / "splits.json").read_text())["assignment"]
    chosen = sorted(s for s, v in assignment.items() if v in use_splits)
    if limit:
        chosen = chosen[:limit]
    rows = []
    for spec_id in chosen:
        leaf = leaves[spec_id]
        geo = json.loads((root / "out" / "geometry" / f"{spec_id}.json").read_text())
        img = Image.open(root / "images" / leaf["image_file"]).convert("RGB")
        b = geo["leaf_box"]
        pad = 14
        box = (max(0, b["x"] - pad), max(0, b["y"] - pad),
               min(img.width, b["x"] + b["w"] + pad), min(img.height, b["y"] + b["h"] + pad))
        arr, masked = mask_caption(np.asarray(img.crop(box)))
        crop = Image.fromarray(arr)
        own = sorted((ln for ln in lines if ln["specimen"] == spec_id), key=lambda x: x["line_index"])
        text = "\n".join(ln["text_reading"] for ln in own)
        full = "\n".join(ln["text_full"] for ln in own)
        sid = f"{SUBSET}-{spec_id.split('-')[-1]}"
        rel = Path("images") / SUBSET / f"{sid}.jpg"
        path = Path(out_dir) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(path, "JPEG", quality=92, optimize=True)
        data = path.read_bytes()
        rows.append({
            "id": sid, "subset": SUBSET, "split": split, "task": "recognition",
            "image": rel.as_posix(), "sha256": hashlib.sha256(data).hexdigest(),
            "width": crop.width, "height": crop.height, "bytes": len(data),
            "text": text, "text_full": full, "script": "tamil-pre-reform", "medium": "palm-leaf",
            "granularity": "page", "provenance": "real", "lexical": "corpus",
            "text_source": f"cict:{spec_id}",
            "render": {"caption_masked": masked, "crop_box": list(box), "source_split": assignment[spec_id]},
            "source": {"specimen": spec_id, "chapter": leaf.get("chapter"), "chapter_title": leaf.get("chapter_title"),
                       "kurals": leaf.get("kurals"), "doi": leaf.get("doi"), "zenodo_record": leaf.get("zenodo_record"),
                       "corpus": leaf.get("source")},
            "license": "CC-BY-4.0", "attribution": ATTRIBUTION,
        })
    return rows
