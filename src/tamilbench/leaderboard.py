"""Aggregate scored runs into ``leaderboard.json`` (consumed by the static site), the
per-script CSV and the README table. Only complete runs on the public ``test`` split are
ranked; human readers are shown as references, never ranked.

The site has two pages built from the same data: ``index.html`` (the leaderboard) and
``scripts.html`` (scores by script stage, for analysis)."""

from __future__ import annotations

import csv
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import BENCHMARK_VERSION, CANARY
from . import subsets as S
from .models import load_registry
from .runner import load_manifest, read_predictions
from .scoring import AGGREGATION, CI_METHOD, COMPOSITE_WEIGHTS, item_result, score_run
from .taxonomy import (ERAS, LINEAGE_ORDER, MEDIA, SCORED_TRACKS, SCRIPT_ERA, SCRIPTS, TRACKS, Era,
                       Track, era_of)

README_START, README_END = "<!-- LEADERBOARD:START -->", "<!-- LEADERBOARD:END -->"
EXAMPLES_PER_SUBSET = 3
PAGES = {"index.html": "_page.src.html", "scripts.html": "_scripts.src.html"}


def _subset_summary(v: dict | None) -> dict | None:
    if v is None:
        return None
    keep = ("score", "ci95", "n", "missing", "cer", "aer", "wer", "exact_match", "macro_f1", "accuracy",
            "invalid_rate", "chrf_pp", "slices", "slices_by_era", "by_script", "by_era", "per_class",
            "hallucination_rate", "by_control", "by_medium", "unit_tests", "recitation_index",
            "recitation_by_script", "recitation_by_era")
    out = {k: (round(v[k], 4) if isinstance(v[k], float) else v[k]) for k in keep if k in v}
    if "diagnostics" in v:
        out["diagnostics"] = {k: v["diagnostics"].get(k) for k in ("order_gap", "bag_cer", "cer_nospace", "ned")}
    return out


def _entry(reg: dict, scores: dict | None) -> dict:
    e = {k: reg.get(k) for k in ("id", "name", "org", "type", "released", "provider", "model")}
    e["verify_id"] = bool(reg.get("verify_id"))
    if scores is None:
        e["status"] = "pending"
        return e
    eras = scores.get("eras") or {}
    e.update({
        "status": "evaluated" if scores.get("complete") else "partial",
        "overall": scores["overall"], "overall_ci95": scores.get("overall_ci95"),
        "recognition_avg": scores["recognition_avg"], "recognition_avg_ci95": scores.get("recognition_avg_ci95"),
        "modern_script": (eras.get("modern") or {}).get("reading"),
        "older_scripts": (eras.get("older") or {}).get("reading"),
        "eras": eras, "era_gap": scores.get("era_gap"), "scripts": scores.get("scripts"),
        "tracks": scores["tracks"],
        "subsets": {k: _subset_summary(v) for k, v in scores["subsets"].items()},
        "prior_reliance": scores.get("prior_reliance"),
        "hallucination": scores.get("hallucination"),
        "unit_tests": scores.get("unit_tests"),
        "failure_modes": scores.get("failure_modes"),
        "failure_modes_by_era": scores.get("failure_modes_by_era"),
        "n_refusals": scores.get("n_refusals", 0), "n_errors": scores.get("n_errors", 0),
        "input_tokens": scores.get("input_tokens"), "output_tokens": scores.get("output_tokens"),
        "mean_latency_s": scores.get("mean_latency_s"), "params": scores["model"].get("params", {}),
        "kind": scores["model"].get("kind"), "prompt_version": scores.get("prompt_version"),
        "scored_at": scores.get("scored_at"), "split": scores.get("split"),
    })
    return e


def _separability(models: list[dict], results_root: Path, data_dir: Path) -> None:
    """For each ranked model, the paired bootstrap difference to the next model down, on the
    ranking's own metric. ``separable`` is False when the 95% interval includes zero: the
    two ranks are a statistical tie."""
    from .compare import compare
    rows = load_manifest(data_dir / "manifest-test.jsonl")
    cache: dict[str, tuple[dict, set]] = {}

    def run_of(mid):
        if mid not in cache:
            d = results_root / mid / f"{BENCHMARK_VERSION}-test"
            preds = {k: v.get("text") for k, v in read_predictions(d / "predictions.jsonl").items()}
            meta = json.loads((d / "run.json").read_text())
            cache[mid] = (preds, set(meta["model"].get("supports") or []))
        return cache[mid]

    for key, rank in (("overall", "rank_overall"), ("recognition_avg", "rank_recognition")):
        ranked = sorted((m for m in models if m.get(rank)), key=lambda m: m[rank])
        for a, b in zip(ranked, ranked[1:]):
            (pa, sa), (pb, sb) = run_of(a["id"]), run_of(b["id"])
            res = compare(rows, pa, pb, supported_a=sa or None, supported_b=sb or None)
            if res.get(key):
                a.setdefault("vs_next", {})[key] = {"model": b["id"], **res[key]}


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    rx, ry = np.argsort(np.argsort(x)).astype(float), np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return round(float(np.corrcoef(rx, ry)[0, 1]), 3)


def proxy_validity(models: list[dict], pairs=(("palm-leaf-synth", "palm-leaf-cict"),)) -> list[dict]:
    """Do synthetic proxies rank systems the way real artefacts do? Rank correlation across
    evaluated systems between a synthetic subset and its real counterpart, and the mean gap."""
    out = []
    for synth, real in pairs:
        pts = [(m["subsets"][synth]["score"], m["subsets"][real]["score"]) for m in models
               if m.get("status") == "evaluated" and (m.get("subsets") or {}).get(synth) and (m["subsets"].get(real))]
        out.append({"synthetic": synth, "real": real, "n_systems": len(pts),
                    "spearman": _spearman([p[0] for p in pts], [p[1] for p in pts]),
                    "mean_gap": round(float(np.mean([p[0] - p[1] for p in pts])), 2) if pts else None})
    return out


def signal_to_noise(models: list[dict], min_systems: int = 3) -> dict | None:
    """Signal-to-noise per subset, script stage and headline score (after Heineman et al.
    2025): signal = standard deviation of the score across evaluated systems; noise = the
    mean bootstrap standard error of one system's score (95 % interval width / 3.92), i.e.
    the sampling noise of the item set. A ratio near or below 1 means the score cannot
    separate the systems that have been evaluated."""
    ev = [m for m in models if m.get("status") == "evaluated"]
    if len(ev) < min_systems:
        return None

    def snr(points):
        pts = [(v, ci) for v, ci in points if v is not None and ci]
        if len(pts) < min_systems:
            return None
        signal = float(np.std([v for v, _ in pts], ddof=1))
        noise = float(np.mean([(ci[1] - ci[0]) / 3.92 for _, ci in pts]))
        return {"signal": round(signal, 2), "noise": round(noise, 2),
                "snr": round(signal / noise, 2) if noise > 0 else None, "n_systems": len(pts)}

    subsets = {}
    for spec in S.SUBSETS:
        got = snr([((m["subsets"].get(spec.id) or {}).get("score"), (m["subsets"].get(spec.id) or {}).get("ci95"))
                   for m in ev])
        if got:
            subsets[spec.id] = got
    scripts = {}
    for sc in LINEAGE_ORDER:
        got = snr([(((m.get("scripts") or {}).get(sc.value) or {}).get("score"),
                    ((m.get("scripts") or {}).get(sc.value) or {}).get("ci95")) for m in ev])
        if got:
            scripts[sc.value] = got
    headline = {}
    for key in ("recognition_avg", "overall"):
        got = snr([(m.get(key), m.get(key + "_ci95")) for m in ev])
        if got:
            headline[key] = got
    for era in Era:
        got = snr([(((m.get("eras") or {}).get(era.value) or {}).get("reading"),
                    ((m.get("eras") or {}).get(era.value) or {}).get("reading_ci95")) for m in ev])
        if got:
            headline[f"{era.value}_reading"] = got
    return {"n_systems": len(ev), "subsets": subsets, "scripts": scripts, "headline": headline,
            "method": "signal = SD of scores across systems; noise = mean bootstrap SE (95% CI width / 3.92)"}


def _sort_key(m: dict):
    ev = m["status"] == "evaluated"
    ov = m.get("overall")
    rec = m.get("recognition_avg")
    return (not ev, -(ov if ov is not None else -1), -(rec if rec is not None else -1))


def _examples(data_dir: Path, results_root: Path, site_dir: Path, model_ids: list[str]) -> list[dict]:
    """Copy a few test images per subset into ``site_dir/assets/examples`` with every evaluated
    model's answer and its official per-item result."""
    rows = load_manifest(data_dir / "manifest-test.jsonl")
    preds = {mid: read_predictions(results_root / mid / f"{BENCHMARK_VERSION}-test" / "predictions.jsonl")
             for mid in model_ids}
    asset_dir = Path(site_dir) / "assets" / "examples"
    if asset_dir.exists():
        shutil.rmtree(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    picked: dict[str, list[dict]] = {}
    for r in rows:
        picked.setdefault(r["subset"], []).append(r)
    chosen = []
    for sid, group in picked.items():
        if sid == "blank-controls":   # one blank and one effaced surface
            chosen += [next(r for r in group if r.get("control") == c) for c in ("blank", "effaced")
                       if any(r.get("control") == c for r in group)]
            continue
        if sid == "perturbed":        # one modern and one older item
            chosen += [next(r for r in group if era_of(r["script"]) == e) for e in Era
                       if any(era_of(r["script"]) == e for r in group)]
            continue
        # real text first (the first example of a subset should be typical), then one
        # nonce or random-word item where the subset has them, to show the controls
        real = [r for r in group if r.get("lexical") in (None, "corpus", "numerals")]
        other = [r for r in group if r not in real]
        take = real[:EXAMPLES_PER_SUBSET - 1] + other[:1]
        for r in real[EXAMPLES_PER_SUBSET - 1:]:
            if len(take) >= EXAMPLES_PER_SUBSET:
                break
            take.append(r)
        chosen += take
    out = []
    for r in chosen:
        spec = S.get(r["subset"])
        pspec = S.presentation(r)
        src = data_dir / r["image"]
        dst = asset_dir / Path(r["image"]).name
        shutil.copyfile(src, dst)
        out.append({
            "id": r["id"], "subset": r["subset"], "image": f"assets/examples/{dst.name}",
            "reference": r.get(pspec.target), "script": r["script"], "medium": r["medium"],
            "provenance": r["provenance"], "lexical": r.get("lexical"), "as_subset": r.get("as_subset"),
            "control": r.get("control"), "text_canonical": r.get("text_canonical"),
            "text_native": r.get("text_native"), "text_diplomatic": r.get("text_diplomatic"),
            "attribution": r.get("attribution"), "source": r.get("source"),
            "predictions": {mid: item_result(spec, r, (preds[mid].get(r["id"]) or {}).get("text"))
                            for mid in model_ids if r["id"] in preds[mid]},
        })
    return out


def _references(results_root: Path, data_dir: Path, models: list[dict]) -> list[dict]:
    """Human readers (runs imported with ``--kind human``), on any split, with every
    evaluated model re-scored on the same items so the comparison is like for like."""
    refs = []
    for p in sorted(Path(results_root).glob(f"*/{BENCHMARK_VERSION}-*/scores.json")):
        s = json.loads(p.read_text())
        if s["model"].get("kind") != "human":
            continue
        split = s.get("split") or p.parent.name.split("-", 1)[1]
        entry = _entry({"id": p.parent.parent.name, "name": s["model"].get("model"), "org": "Human reader",
                        "type": "human", "provider": "human", "model": s["model"].get("model")}, s)
        entry["status"] = "reference"
        entry["reader"] = (s["model"].get("params") or {}).get("reader")
        rows = load_manifest(data_dir / f"manifest-{split}.jsonl")
        answered = {k for k, v in read_predictions(p.parent / "predictions.jsonl").items()}
        rows = [r for r in rows if r["id"] in answered]
        same_items = {}
        for m in models:
            if m.get("status") != "evaluated":
                continue
            d = Path(results_root) / m["id"] / f"{BENCHMARK_VERSION}-test"
            preds = {k: v.get("text") for k, v in read_predictions(d / "predictions.jsonl").items()}
            supported = set(json.loads((d / "run.json").read_text())["model"].get("supports") or []) or None
            sc = score_run(rows, preds, supported_tasks=supported, n_boot=200, keep_per_sample=False)
            same_items[m["id"]] = {"recognition_avg": sc["recognition_avg"],
                                   "modern_script": (sc["eras"].get("modern") or {}).get("reading"),
                                   "older_scripts": (sc["eras"].get("older") or {}).get("reading"),
                                   "scripts": {k: v["score"] for k, v in sc["scripts"].items()}}
        entry.update({"split": split, "n_items": len(rows), "models_on_same_items": same_items})
        refs.append(entry)
    return refs


def _robustness(results_root: Path, model_id: str) -> dict | None:
    for split in ("test", "lite"):
        p = Path(results_root) / model_id / f"{BENCHMARK_VERSION}-{split}-robustness.json"
        if p.exists():
            return json.loads(p.read_text())
    return None


def write_script_csv(lb: dict, path: Path) -> Path:
    """Every evaluated system's score per script stage, per (script, track) cell and per
    (script, subset) slice, in long format, for analysis outside the site."""
    path = Path(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_id", "model", "era", "script", "level", "key", "score", "ci95_low", "ci95_high", "n"])
        for m in lb["models"] + lb.get("references", []):
            if m.get("status") not in ("evaluated", "reference"):
                continue
            for e, v in (m.get("eras") or {}).items():
                ci = v.get("reading_ci95") or [None, None]
                w.writerow([m["id"], m["name"], e, "", "era", "reading", v.get("reading"), ci[0], ci[1], ""])
            for sc, v in (m.get("scripts") or {}).items():
                ci = v.get("ci95") or [None, None]
                w.writerow([m["id"], m["name"], v.get("era"), sc, "script", sc, v.get("score"), ci[0], ci[1], v.get("n")])
                for t, x in (v.get("tracks") or {}).items():
                    w.writerow([m["id"], m["name"], v.get("era"), sc, "track", t, x, "", "", ""])
                for sid, x in (v.get("subsets") or {}).items():
                    if x:
                        ci = x.get("ci95") or [None, None]
                        w.writerow([m["id"], m["name"], v.get("era"), sc, "subset", sid, x.get("score"), ci[0], ci[1],
                                    x.get("n")])
    return path


def build(results_root: Path, data_dir: Path, out_json: Path, *, readme: Path | None = None,
          include_debug: bool = False) -> dict:
    registry = load_registry()
    found: dict[str, dict] = {}
    for p in sorted(Path(results_root).glob(f"*/{BENCHMARK_VERSION}-test/scores.json")):
        s = json.loads(p.read_text())
        # debug adapters, interactive (chat-client) runs and human readers are never ranked
        if s["model"].get("kind") in ("debug", "interactive", "human") and not include_debug:
            continue
        found[p.parent.parent.name] = s
    models = []
    for reg in registry:
        models.append(_entry(reg, found.pop(reg["id"], None)))
    for mid, s in found.items():
        models.append(_entry({"id": mid, "name": s["model"].get("model", mid), "org": s["model"].get("provider"),
                              "type": s["model"].get("kind"), "provider": s["model"].get("provider"),
                              "model": s["model"].get("model")}, s))
    models.sort(key=_sort_key)
    rank_o = rank_r = 0
    for m in models:
        if m["status"] == "evaluated" and m.get("overall") is not None:
            rank_o += 1
            m["rank_overall"] = rank_o
    for m in sorted([m for m in models if m["status"] == "evaluated" and m.get("recognition_avg") is not None],
                    key=lambda m: -m["recognition_avg"]):
        rank_r += 1
        m["rank_recognition"] = rank_r
    _separability(models, Path(results_root), data_dir)
    for m in models:
        if m["status"] in ("evaluated", "partial"):
            rob = _robustness(Path(results_root), m["id"])
            if rob:
                m["robustness"] = rob

    meta = json.loads((data_dir / "subsets.json").read_text())
    coverage: dict[str, dict[str, dict[str, int]]] = {}
    for r in load_manifest(data_dir / "manifest-test.jsonl"):
        if not S.get(r["subset"]).ranked:
            continue
        cell = coverage.setdefault(r["script"], {}).setdefault(r["medium"], {"synthetic": 0, "real": 0})
        cell[r["provenance"]] += 1
    evaluated_ids = [m["id"] for m in models if m["status"] in ("evaluated", "partial")]
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    lb = {
        "benchmark": {"name": "Tamil OCR / HTR Benchmark", "version": BENCHMARK_VERSION,
                      "counts": meta.get("counts", {}), "subsets": meta["subsets"]},
        "tracks": [{"id": t.value, "name": TRACKS[t].name, "tamil": TRACKS[t].tamil,
                    "description": TRACKS[t].description, "subsets": S.TRACK_SUBSETS[t]} for t in SCORED_TRACKS],
        "controls": {"id": Track.DIAGNOSTICS.value, "name": TRACKS[Track.DIAGNOSTICS].name,
                     "description": TRACKS[Track.DIAGNOSTICS].description, "subsets": S.CONTROL_SUBSETS},
        "eras": {e.value: {"name": ERAS[e].name, "tamil": ERAS[e].tamil, "description": ERAS[e].description,
                           "scripts": [s.value for s in LINEAGE_ORDER if SCRIPT_ERA[s] == e],
                           "weights": COMPOSITE_WEIGHTS[e]} for e in Era},
        "lineage": [s.value for s in LINEAGE_ORDER],
        "aggregation": AGGREGATION,
        "scripts": {s.value: {"name": i.name, "tamil": i.tamil, "period": i.period, "description": i.description,
                              "era": SCRIPT_ERA[s].value if SCRIPT_ERA[s] else None}
                    for s, i in SCRIPTS.items()},
        "media": {m.value: {"name": i.name, "tamil": i.tamil, "description": i.description} for m, i in MEDIA.items()},
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": models,
        "references": _references(Path(results_root), data_dir, models),
        "coverage": coverage,
        "diagnostics": {"proxy_validity": proxy_validity(models), "signal_to_noise": signal_to_noise(models),
                        "ci_method": CI_METHOD, "canary": CANARY},
        "examples": _examples(data_dir, Path(results_root), out_json.parent.parent, evaluated_ids),
    }
    out_json.write_text(json.dumps(lb, ensure_ascii=False, indent=1), encoding="utf-8")
    write_script_csv(lb, out_json.parent / "scores-by-script.csv")
    write_site(out_json.parent.parent)
    if readme and Path(readme).exists():
        update_readme(Path(readme), lb)
    return lb


SITE_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="{description}">
</head>
<body>
"""
DESCRIPTIONS = {
    "index.html": "Leaderboard of the Tamil OCR / HTR Benchmark: models read Tamil from Tamil-Brahmi inscriptions "
                  "and palm leaves to screens.",
    "scripts.html": "Scores by script stage on the Tamil OCR / HTR Benchmark: modern Tamil, pre-reform Tamil, "
                    "Grantha–Tamil, Grantha and Tamil-Brahmi.",
}


def page_fragment(site_dir: Path, source: str = "_page.src.html") -> str:
    """A page body: the source fragment with the shared stylesheet and the inline masthead
    font face filled in."""
    site_dir = Path(site_dir)
    frag = (site_dir / source).read_text(encoding="utf-8")
    base = site_dir / "_base.css"
    face = site_dir / "_lohit-face.css"
    frag = frag.replace("/*BASE-CSS*/", base.read_text(encoding="utf-8").strip() if base.exists() else "")
    return frag.replace("/*LOHIT-FACE*/", face.read_text(encoding="utf-8").strip() if face.exists() else "")


def write_site(site_dir: Path) -> list[Path]:
    """Wrap each page fragment into a GitHub Pages HTML file."""
    out = []
    for page, source in PAGES.items():
        if not (Path(site_dir) / source).exists():
            continue
        path = Path(site_dir) / page
        path.write_text(SITE_HEAD.format(description=DESCRIPTIONS[page]) + page_fragment(site_dir, source)
                        + "\n</body>\n</html>\n", encoding="utf-8")
        out.append(path)
    return out


def write_standalone(site_dir: Path, lb_json: Path, out: Path, max_width: int = 900, *,
                     page: str = "index.html", root_link: str = "./", full_document: bool = False) -> Path:
    """A single self-contained page (data and example images inlined) — used for previews
    outside GitHub Pages. Links to the main page become ``root_link``; ``full_document``
    adds the doctype and head (for a page served beside the main one)."""
    import base64
    import io

    from PIL import Image
    lb = json.loads(Path(lb_json).read_text(encoding="utf-8"))
    if page == "index.html":
        for ex in lb.get("examples", []):
            img = Image.open(Path(site_dir) / ex["image"]).convert("RGB")
            if img.width > max_width:
                img = img.resize((max_width, int(img.height * max_width / img.width)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=80, optimize=True)
            ex["image"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    else:
        lb["examples"] = []
    frag = page_fragment(site_dir, PAGES[page]).replace('href="index.html', f'href="{root_link}')
    data = json.dumps(lb, ensure_ascii=False).replace("</", "<\\/")
    inline = f"<script>window.__LEADERBOARD__ = {data};</script>\n"
    marker = "<!-- INLINE-DATA -->"
    frag = frag.replace(marker, inline) if marker in frag else inline + frag
    if full_document:
        frag = SITE_HEAD.format(description=DESCRIPTIONS[page]) + frag + "\n</body>\n</html>\n"
    Path(out).write_text(frag, encoding="utf-8")
    return Path(out)


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.1f}"


def markdown_table(lb: dict) -> str:
    tracks = list(SCORED_TRACKS)
    head = ["#", "Model", "Overall", "OCR/HTR", "Modern script", "Older scripts"] + [TRACKS[t].name for t in tracks]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---:"] + [":---"] + ["---:"] * (len(head) - 2)) + "|"]
    ev = [m for m in lb["models"] if m["status"] == "evaluated"]
    for m in ev:
        rank = m.get("rank_overall") or f"({m.get('rank_recognition', '')})"
        vs = (m.get("vs_next") or {}).get("overall" if m.get("rank_overall") else "recognition_avg")
        if vs and not vs["separable"]:
            rank = f"{rank} ≈"
        cells = [str(rank), f"**{m['name']}** <br><sub>{m.get('org') or ''}</sub>", _fmt(m["overall"]),
                 _fmt(m["recognition_avg"]), _fmt(m.get("modern_script")), _fmt(m.get("older_scripts"))]
        cells += [_fmt(m["tracks"].get(t.value)) for t in tracks]
        lines.append("| " + " | ".join(cells) + " |")
    out = "\n".join(lines) if ev else "_No complete runs yet._"
    if ev:
        order = [s for s in lb.get("lineage", []) if any(s in (m.get("scripts") or {}) for m in ev)][::-1]
        names = {k: v["name"] for k, v in lb["scripts"].items()}
        out += ("\n\n**Reading by script stage** (newest to oldest; every older stage counts equally in "
                "*Older scripts*):\n\n| Model | " + " | ".join(names[s] for s in order) + " |\n|:---|"
                + "---:|" * len(order) + "\n")
        out += "\n".join(f"| {m['name']} | " + " | ".join(_fmt(((m.get('scripts') or {}).get(s) or {}).get('score'))
                                                           for s in order) + " |" for m in ev)
    pending = [m for m in lb["models"] if m["status"] != "evaluated"]
    if pending:
        out += ("\n\n**Awaiting evaluation** (adapters ready — add an API key or endpoint and run "
                "`tamilbench run --model <id>`): " + ", ".join(f"`{m['id']}`" for m in pending))
    out += (f"\n\n<sub>Scores are 0–100 (higher is better): 100·(1−CER) for reading, macro-F1 for identification, "
            f"chrF++ for translation. The modern script and the older scripts count equally: OCR/HTR = ½ Modern "
            f"script + ½ Older scripts, where Older scripts is the mean of pre-reform Tamil, Grantha–Tamil, Grantha "
            f"and Tamil-Brahmi; Overall = ½ modern composite + ½ older composite (¾ reading, ¼ identification and "
            f"translation in each half). Systems that only read are ranked on OCR/HTR (in parentheses). Track "
            f"columns break the same scores down by medium. ≈ marks a rank not statistically separable from the "
            f"next (paired cluster bootstrap, 95 %). Per-script detail: `leaderboard/scripts.html`. Benchmark "
            f"{BENCHMARK_VERSION}, generated {lb['generated_at'][:10]}.</sub>")
    return out


def update_readme(path: Path, lb: dict) -> None:
    text = path.read_text(encoding="utf-8")
    if README_START not in text:
        return
    block = f"{README_START}\n{markdown_table(lb)}\n{README_END}"
    text = re.sub(re.escape(README_START) + r".*?" + re.escape(README_END), lambda _: block, text, flags=re.S)
    path.write_text(text, encoding="utf-8")
