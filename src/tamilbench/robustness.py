"""Prompt robustness and run-to-run stability — diagnostics, never ranked.

* **Prompt robustness** (Mizrahi et al. 2024, "State of What Art?"): the same system is run
  with each paraphrase of the prompts (``prompts.PROMPT_VARIANTS``); the spread of its
  scores shows how much of a ranking is wording rather than reading.
* **Stability** (Levchenko 2025): the canonical prompt is run again; the share of items
  whose answer changes and the score spread show how repeatable a single run is.

Runs are written beside the main run as ``<version>-<split>-p1``, ``-p2``, ``-r1`` … and the
summary as ``<version>-<split>-robustness.json`` in the model's results folder.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import BENCHMARK_VERSION
from . import runner
from .prompts import N_VARIANTS
from .scoring import score_run

KEYS = ("overall", "recognition_avg")


def tag_for(variant: int, repeat: int) -> str | None:
    if variant == 0 and repeat == 0:
        return None
    return f"p{variant}" + (f"r{repeat}" if repeat else "") if variant else f"r{repeat}"


def _headline(scores: dict) -> dict:
    out = {k: scores.get(k) for k in KEYS}
    for era, v in (scores.get("eras") or {}).items():
        out[f"{era}_reading"] = v.get("reading")
    for sid, v in scores["subsets"].items():
        if v is not None:
            out[f"subset:{sid}"] = round(v["score"], 2)
    return out


def _spread(values: list[float | None]) -> dict | None:
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None
    return {"values": [round(v, 2) for v in vals], "range": round(max(vals) - min(vals), 2),
            "sd": round(float(np.std(vals, ddof=1)), 2)}


def summarize(model_dir: Path, data_dir: Path, *, split: str = "lite", n_boot: int = 200) -> dict | None:
    """Collect every robustness run of one model on ``split`` and write the summary."""
    model_dir = Path(model_dir)
    rows = runner.load_manifest(runner.manifest_path(data_dir, split))
    runs: dict[tuple[int, int], dict] = {}
    texts: dict[tuple[int, int], dict] = {}
    for d in sorted(model_dir.glob(f"{BENCHMARK_VERSION}-{split}*")):
        if not d.is_dir() or not (d / "run.json").exists():
            continue
        meta = json.loads((d / "run.json").read_text())
        tag = meta.get("tag")
        variant = int(meta.get("prompt_variant") or 0)
        repeat = int(tag.split("r", 1)[1]) if tag and "r" in tag else 0
        preds = {k: v.get("text") for k, v in runner.read_predictions(d / "predictions.jsonl").items()}
        ids = set(preds)
        sub_rows = [r for r in rows if r["id"] in ids]
        if not sub_rows:
            continue
        supported = set(meta["model"].get("supports") or []) or None
        runs[(variant, repeat)] = _headline(score_run(sub_rows, preds, supported_tasks=supported, n_boot=n_boot,
                                                       keep_per_sample=False))
        texts[(variant, repeat)] = preds
    if len(runs) < 2:
        return None
    keys = sorted({k for v in runs.values() for k in v})
    variants = sorted(v for v, r in runs if r == 0)
    repeats = sorted(r for v, r in runs if v == 0)
    out = {"model": model_dir.name, "split": split, "variants": variants, "repeats": repeats,
           "prompt_spread": {k: _spread([runs[(v, 0)].get(k) for v in variants]) for k in keys} if len(variants) > 1 else None,
           "repeat_spread": {k: _spread([runs[(0, r)].get(k) for r in repeats]) for k in keys} if len(repeats) > 1 else None}
    if len(repeats) > 1:
        base = texts[(0, 0)]
        changed = [sum(1 for i in base if i in texts[(0, r)] and (texts[(0, r)][i] or "") != (base[i] or "")) /
                   max(1, sum(1 for i in base if i in texts[(0, r)])) for r in repeats[1:]]
        out["answers_changed_between_repeats"] = round(float(np.mean(changed)), 4)
    path = model_dir / f"{BENCHMARK_VERSION}-{split}-robustness.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def run(model: str, data_dir: Path, *, split: str = "lite", results_root: Path = Path("results"),
        variants: int = N_VARIANTS, repeats: int = 1, subset_ids: list[str] | None = None,
        limit: int | None = None, concurrency: int = 4, progress: bool = True, n_boot: int = 200,
        **adapter_params) -> dict | None:
    """Run ``model`` with every prompt variant (and ``repeats`` extra runs of the canonical
    prompt), then summarise. The main run of the split is reused if it exists."""
    from .models import create
    slug = None
    plan = [(v, 0) for v in range(variants)] + [(0, r) for r in range(1, repeats + 1)]
    for v, r in plan:
        adapter = create(model, **adapter_params)
        slug = adapter.slug
        runner.run(adapter, data_dir, split=split, results_root=results_root, subset_ids=subset_ids,
                   limit=limit, concurrency=concurrency, progress=progress, prompt_variant=v,
                   tag=tag_for(v, r), resume=True)
    return summarize(Path(results_root) / slug, data_dir, split=split, n_boot=n_boot)
