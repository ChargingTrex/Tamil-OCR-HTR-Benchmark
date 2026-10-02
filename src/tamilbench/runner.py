"""Run a model over the benchmark, then score it.

Results layout (one directory per model and benchmark split)::

    results/<model-id>/<version>-<split>/
        predictions.jsonl   one line per sample (appended as answers arrive; resumable)
        run.json            who/what/when: adapter, parameters, prompt version, totals
        scores.json         produced by `tamilbench score`
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from . import BENCHMARK_VERSION, __version__
from . import subsets as S
from .prompts import PROMPT_VERSION, PROMPTS, SYSTEM
from .scoring import score_run

MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def manifest_path(data_dir: Path, split: str) -> Path:
    return Path(data_dir) / f"manifest-{split}.jsonl"


def results_dir(root: Path, slug: str, split: str) -> Path:
    return Path(root) / slug / f"{BENCHMARK_VERSION}-{split}"


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_predictions(path: Path) -> dict[str, dict]:
    """Latest record per sample id wins (re-runs append)."""
    out: dict[str, dict] = {}
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["id"]] = rec
    return out


def run(adapter, data_dir: Path, *, split: str = "test", results_root: Path = Path("results"),
        subset_ids: list[str] | None = None, limit: int | None = None, concurrency: int = 4,
        resume: bool = True, progress: bool = True) -> Path:
    mpath = manifest_path(data_dir, split)
    rows = load_manifest(mpath)
    out = results_dir(results_root, adapter.slug, split)
    out.mkdir(parents=True, exist_ok=True)
    pred_file = out / "predictions.jsonl"
    done = {k for k, v in read_predictions(pred_file).items()
            if resume and (v.get("text") is not None or v.get("refusal"))}

    per_subset: dict[str, int] = defaultdict(int)
    todo = []
    for r in rows:
        spec = S.get(r["subset"])
        if subset_ids and r["subset"] not in subset_ids:
            continue
        if spec.task.value not in adapter.supports:
            continue
        if limit and per_subset[r["subset"]] >= limit:
            continue
        per_subset[r["subset"]] += 1
        if r["id"] not in done:
            todo.append(r)

    meta_file = out / "run.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() and resume else {}
    meta.update({
        "model": adapter.metadata(), "benchmark_version": BENCHMARK_VERSION, "split": split,
        "manifest_sha256": hashlib.sha256(mpath.read_bytes()).hexdigest(),
        "prompt_version": PROMPT_VERSION, "tamilbench_version": __version__, "git_commit": _git_commit(),
        "python": platform.python_version(), "platform": platform.platform(),
        "started_at": meta.get("started_at") or _now(),
    })
    meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    lock = threading.Lock()
    data_dir = Path(data_dir)

    def job(r: dict) -> dict:
        spec = S.get(r["subset"])
        img_path = data_dir / r["image"]
        sample = {**r, "_target_field": spec.target}
        pred = adapter.predict(img_path.read_bytes(), MIME.get(img_path.suffix.lower(), "image/jpeg"),
                               PROMPTS[spec.prompt], SYSTEM, sample)
        return {"id": r["id"], "subset": r["subset"], **pred.to_json()}

    t0 = time.time()
    n_done = 0
    with open(pred_file, "a", encoding="utf-8") as f, ThreadPoolExecutor(max(1, concurrency)) as ex:
        futures = [ex.submit(job, r) for r in todo]
        for fut in as_completed(futures):
            rec = fut.result()
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n_done += 1
            if progress and (n_done % 25 == 0 or n_done == len(todo)):
                rate = n_done / max(1e-6, time.time() - t0)
                print(f"  {adapter.slug}: {n_done}/{len(todo)} ({rate:.2f}/s)", flush=True)
    adapter.close()

    preds = read_predictions(pred_file)
    meta.update({
        "finished_at": _now(), "n_predictions": len(preds),
        "n_errors": sum(1 for p in preds.values() if p.get("error")),
        "n_refusals": sum(1 for p in preds.values() if p.get("refusal")),
        "input_tokens": sum(p.get("input_tokens") or 0 for p in preds.values()),
        "output_tokens": sum(p.get("output_tokens") or 0 for p in preds.values()),
        "mean_latency_s": round(sum(p.get("latency_s") or 0 for p in preds.values()) / max(1, len(preds)), 3),
        "subsets_run": sorted({p["subset"] for p in preds.values()}),
    })
    meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return out


def score(result_dir: Path, data_dir: Path, *, split: str | None = None, n_boot: int = 1000) -> dict:
    result_dir = Path(result_dir)
    meta = json.loads((result_dir / "run.json").read_text())
    split = split or meta["split"]
    rows = load_manifest(manifest_path(data_dir, split))
    preds_raw = read_predictions(result_dir / "predictions.jsonl")
    preds = {k: v.get("text") for k, v in preds_raw.items()}
    supported = set(meta["model"].get("supports") or [])
    scores = score_run(rows, preds, supported_tasks=supported or None, n_boot=n_boot, keep_per_sample=False)
    complete = all(r["id"] in preds_raw for r in rows if S.get(r["subset"]).task.value in supported)
    scores.update({
        "model": meta["model"], "benchmark_version": meta["benchmark_version"], "split": split,
        "prompt_version": meta.get("prompt_version"), "complete": complete,
        "n_refusals": sum(1 for p in preds_raw.values() if p.get("refusal")),
        "n_errors": sum(1 for p in preds_raw.values() if p.get("error")),
        "input_tokens": meta.get("input_tokens"), "output_tokens": meta.get("output_tokens"),
        "mean_latency_s": meta.get("mean_latency_s"), "scored_at": _now(),
    })
    (result_dir / "scores.json").write_text(json.dumps(scores, ensure_ascii=False, indent=2))
    return scores


def import_predictions(file: Path, *, model_id: str, name: str | None, supports: list[str], split: str,
                       results_root: Path = Path("results"), notes: str | None = None) -> Path:
    """Bring in predictions produced outside this harness (JSONL with ``id`` and ``text``)."""
    out = results_dir(results_root, model_id, split)
    out.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(line) for line in Path(file).read_text(encoding="utf-8").splitlines() if line.strip()]
    with open(out / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps({"id": r["id"], "subset": r.get("subset") or r["id"].rsplit("-", 1)[0],
                                "text": r.get("text")}, ensure_ascii=False) + "\n")
    meta = {"model": {"id": model_id, "provider": "import", "model": name or model_id, "kind": "imported",
                      "supports": sorted(supports), "params": {"notes": notes} if notes else {}},
            "benchmark_version": BENCHMARK_VERSION, "split": split, "prompt_version": None,
            "imported_at": _now(), "n_predictions": len(recs)}
    (out / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return out
