"""Aggregate scored runs into ``leaderboard.json`` (consumed by the static site) and the
README table. Only complete runs on the public ``test`` split are ranked."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import BENCHMARK_VERSION, CANARY
from . import subsets as S
from .models import load_registry
from .metrics import classification as cls_m
from .metrics import recognition as rec_m
from .metrics import translation as tr_m
from .runner import load_manifest, read_predictions
from .scoring import CI_METHOD
from .taxonomy import MEDIA, SCRIPTS, TRACKS, Task, Track

README_START, README_END = "<!-- LEADERBOARD:START -->", "<!-- LEADERBOARD:END -->"
EXAMPLES_PER_SUBSET = 3


def _subset_summary(v: dict | None) -> dict | None:
    if v is None:
        return None
    keep = ("score", "ci95", "n", "missing", "cer", "aer", "wer", "exact_match", "macro_f1", "accuracy",
            "invalid_rate", "chrf_pp", "slices")
    return {k: (round(v[k], 4) if isinstance(v[k], float) else v[k]) for k in keep if k in v}


def _entry(reg: dict, scores: dict | None) -> dict:
    e = {k: reg.get(k) for k in ("id", "name", "org", "type", "released", "provider", "model")}
    e["verify_id"] = bool(reg.get("verify_id"))
    if scores is None:
        e["status"] = "pending"
        return e
    e.update({
        "status": "evaluated" if scores.get("complete") else "partial",
        "overall": scores["overall"], "recognition_avg": scores["recognition_avg"],
        "tracks": scores["tracks"],
        "subsets": {k: _subset_summary(v) for k, v in scores["subsets"].items()},
        "prior_reliance": scores.get("prior_reliance"),
        "failure_modes": scores.get("failure_modes"),
        "n_refusals": scores.get("n_refusals", 0), "n_errors": scores.get("n_errors", 0),
        "input_tokens": scores.get("input_tokens"), "output_tokens": scores.get("output_tokens"),
        "mean_latency_s": scores.get("mean_latency_s"), "params": scores["model"].get("params", {}),
        "kind": scores["model"].get("kind"), "prompt_version": scores.get("prompt_version"),
        "scored_at": scores.get("scored_at"),
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
    for group in picked.values():
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
        src = data_dir / r["image"]
        dst = asset_dir / Path(r["image"]).name
        shutil.copyfile(src, dst)
        out.append({
            "id": r["id"], "subset": r["subset"], "image": f"assets/examples/{dst.name}",
            "reference": r.get(spec.target), "script": r["script"], "medium": r["medium"],
            "provenance": r["provenance"], "lexical": r.get("lexical"),
            "text_native": r.get("text_native"), "text_diplomatic": r.get("text_diplomatic"),
            "attribution": r.get("attribution"), "source": r.get("source"),
            "predictions": {mid: _item_result(spec, r, (preds[mid].get(r["id"]) or {}).get("text"))
                            for mid in model_ids if r["id"] in preds[mid]},
        })
    return out


def _item_result(spec, row: dict, text: str | None) -> dict:
    """Official per-item score, computed with the same code as the subset score."""
    ref = row.get(spec.target) or ""
    if spec.task == Task.RECOGNITION:
        st = rec_m.sample_stats(ref, text or "", rec_m.TextPolicy.from_dict(spec.policy))
        return {"text": text, "cer": round(st["char_edits"] / max(1, st["chars"]), 3)}
    if spec.task in (Task.SCRIPT_ID, Task.MEDIUM_ID):
        lab = cls_m.parse_label(text or "", list(spec.labels))
        return {"text": text, "label": lab, "ok": lab == ref}
    return {"text": text, "chrf": round(tr_m.f_score(tr_m.sentence_stats(text or "", ref)), 1)}


def build(results_root: Path, data_dir: Path, out_json: Path, *, readme: Path | None = None,
          include_debug: bool = False) -> dict:
    registry = load_registry()
    found: dict[str, dict] = {}
    for p in sorted(Path(results_root).glob(f"*/{BENCHMARK_VERSION}-test/scores.json")):
        s = json.loads(p.read_text())
        if s["model"].get("kind") == "debug" and not include_debug:
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

    meta = json.loads((data_dir / "subsets.json").read_text())
    coverage: dict[str, dict[str, dict[str, int]]] = {}
    for r in load_manifest(data_dir / "manifest-test.jsonl"):
        cell = coverage.setdefault(r["script"], {}).setdefault(r["medium"], {"synthetic": 0, "real": 0})
        cell[r["provenance"]] += 1
    evaluated_ids = [m["id"] for m in models if m["status"] in ("evaluated", "partial")]
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    lb = {
        "benchmark": {"name": "Tamil OCR / HTR Benchmark", "version": BENCHMARK_VERSION,
                      "counts": meta.get("counts", {}), "subsets": meta["subsets"]},
        "tracks": [{"id": t.value, "name": TRACKS[t].name, "tamil": TRACKS[t].tamil,
                    "description": TRACKS[t].description, "subsets": S.TRACK_SUBSETS[t]} for t in Track],
        "scripts": {s.value: {"name": i.name, "tamil": i.tamil, "period": i.period, "description": i.description}
                    for s, i in SCRIPTS.items()},
        "media": {m.value: {"name": i.name, "tamil": i.tamil, "description": i.description} for m, i in MEDIA.items()},
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": models,
        "coverage": coverage,
        "diagnostics": {"proxy_validity": proxy_validity(models), "ci_method": CI_METHOD,
                        "canary": CANARY},
        "examples": _examples(data_dir, Path(results_root), out_json.parent.parent, evaluated_ids),
    }
    out_json.write_text(json.dumps(lb, ensure_ascii=False, indent=1), encoding="utf-8")
    write_site(out_json.parent.parent)
    if readme and Path(readme).exists():
        update_readme(Path(readme), lb)
    return lb


SITE_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="Leaderboard of the Tamil OCR / HTR Benchmark: models read Tamil from Tamil-Brahmi inscriptions and palm leaves to screens.">
</head>
<body>
"""


def page_fragment(site_dir: Path) -> str:
    """The page body: ``_page.src.html`` with the inline masthead font face filled in."""
    site_dir = Path(site_dir)
    frag = (site_dir / "_page.src.html").read_text(encoding="utf-8")
    face = site_dir / "_lohit-face.css"
    return frag.replace("/*LOHIT-FACE*/", face.read_text(encoding="utf-8").strip() if face.exists() else "")


def write_site(site_dir: Path) -> Path:
    """Wrap the page fragment into the GitHub Pages ``index.html``."""
    frag = page_fragment(site_dir)
    out = Path(site_dir) / "index.html"
    out.write_text(SITE_HEAD + frag + "\n</body>\n</html>\n", encoding="utf-8")
    return out


def write_standalone(site_dir: Path, lb_json: Path, out: Path, max_width: int = 900) -> Path:
    """A single self-contained HTML file (data and example images inlined) — used for
    previews outside GitHub Pages."""
    import base64
    import io

    from PIL import Image
    lb = json.loads(Path(lb_json).read_text(encoding="utf-8"))
    for ex in lb.get("examples", []):
        img = Image.open(Path(site_dir) / ex["image"]).convert("RGB")
        if img.width > max_width:
            img = img.resize((max_width, int(img.height * max_width / img.width)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=80, optimize=True)
        ex["image"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    frag = page_fragment(site_dir)
    data = json.dumps(lb, ensure_ascii=False).replace("</", "<\\/")
    inline = f"<script>window.__LEADERBOARD__ = {data};</script>\n"
    marker = "<!-- INLINE-DATA -->"
    frag = frag.replace(marker, inline) if marker in frag else inline + frag
    Path(out).write_text(frag, encoding="utf-8")
    return Path(out)


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.1f}"


def markdown_table(lb: dict) -> str:
    tracks = [t for t in Track]
    head = ["#", "Model", "Overall", "OCR/HTR avg"] + [TRACKS[t].name for t in tracks]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---:"] + [":---"] + ["---:"] * (len(head) - 2)) + "|"]
    ev = [m for m in lb["models"] if m["status"] == "evaluated"]
    for m in ev:
        rank = m.get("rank_overall") or f"({m.get('rank_recognition', '')})"
        vs = (m.get("vs_next") or {}).get("overall" if m.get("rank_overall") else "recognition_avg")
        if vs and not vs["separable"]:
            rank = f"{rank} ≈"
        cells = [str(rank), f"**{m['name']}** <br><sub>{m.get('org') or ''}</sub>", _fmt(m["overall"]),
                 _fmt(m["recognition_avg"])] + [_fmt(m["tracks"].get(t.value)) for t in tracks]
        lines.append("| " + " | ".join(cells) + " |")
    pending = [m for m in lb["models"] if m["status"] != "evaluated"]
    out = "\n".join(lines) if ev else "_No complete runs yet._"
    if pending:
        out += ("\n\n**Awaiting evaluation** (adapters ready — add an API key or endpoint and run "
                "`tamilbench run --model <id>`): " + ", ".join(f"`{m['id']}`" for m in pending))
    out += (f"\n\n<sub>Scores are 0–100 (higher is better): 100·(1−CER) for reading tasks, macro-F1 for "
            f"identification, chrF++ for translation. Overall = mean of the 8 tracks; OCR/HTR avg = mean of the 6 "
            f"reading tracks, so OCR engines that cannot classify or translate are ranked there (in parentheses). "
            f"≈ marks a rank not statistically separable from the next (paired cluster bootstrap, 95 %). "
            f"Benchmark {BENCHMARK_VERSION}, generated {lb['generated_at'][:10]}.</sub>")
    return out


def update_readme(path: Path, lb: dict) -> None:
    text = path.read_text(encoding="utf-8")
    if README_START not in text:
        return
    block = f"{README_START}\n{markdown_table(lb)}\n{README_END}"
    text = re.sub(re.escape(README_START) + r".*?" + re.escape(README_END), lambda _: block, text, flags=re.S)
    path.write_text(text, encoding="utf-8")
