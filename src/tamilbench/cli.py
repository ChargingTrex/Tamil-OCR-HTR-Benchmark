"""``tamilbench`` command-line interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from . import BENCHMARK_VERSION, __version__

DATA = Path("data") / BENCHMARK_VERSION
RESULTS = Path("results")
TESSDATA_BEST_URL = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/{path}"


def cmd_build(a):
    from . import build
    counts = {}
    if a.count:
        from .subsets import SUBSETS
        counts = {s.id: a.count for s in SUBSETS}
    rows = build.build(Path(a.out), seed=a.seed, split=a.split, subset_ids=a.subsets, counts=counts,
                       workers=a.workers, cict_root=Path(a.cict_root) if a.cict_root else None)
    build.write_manifests(Path(a.out), rows, split=a.split, seed=a.seed)
    print(f"built {len(rows)} samples into {a.out}")


def cmd_run(a):
    from . import runner
    from .models import create
    overrides = dict(kv.split("=", 1) for kv in a.param or [])
    from .models import _coerce
    overrides = {k: _coerce(v) for k, v in overrides.items()}
    adapter = create(a.model, **overrides)
    out = runner.run(adapter, Path(a.data), split=a.split, results_root=Path(a.results), subset_ids=a.subsets,
                     limit=a.limit, concurrency=a.concurrency, resume=not a.no_resume)
    print(f"predictions in {out}")
    if not a.no_score:
        s = runner.score(out, Path(a.data), n_boot=a.n_boot)
        _print_scores(s)


def cmd_score(a):
    from . import runner
    paths = [Path(p) for p in a.results_dirs] if a.results_dirs else sorted(Path(a.results).glob(f"*/{BENCHMARK_VERSION}-{a.split}"))
    for p in paths:
        if (p / "predictions.jsonl").exists():
            s = runner.score(p, Path(a.data), n_boot=a.n_boot)
            print(f"== {p}")
            _print_scores(s)


def cmd_compare(a):
    from . import runner
    from .compare import compare

    def load(model):
        d = Path(model) if Path(model).is_dir() else Path(a.results) / model / f"{BENCHMARK_VERSION}-{a.split}"
        preds = {k: v.get("text") for k, v in runner.read_predictions(d / "predictions.jsonl").items()}
        meta = json.loads((d / "run.json").read_text())
        return preds, set(meta["model"].get("supports") or []) or None

    (pa, sa), (pb, sb) = load(a.a), load(a.b)
    rows = runner.load_manifest(runner.manifest_path(Path(a.data), a.split))
    res = compare(rows, pa, pb, supported_a=sa, supported_b=sb, n_boot=a.n_boot)
    if a.json:
        Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=2))

    def line(name, r):
        if not r:
            return f"{name:22s}   —"
        mark = "" if r["separable"] else "  (tie)"
        return f"{name:22s} {r['diff']:+7.2f}  [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]  p={r['p']:.3f}{mark}"
    print(f"{a.a} − {a.b}  (paired cluster bootstrap, {a.n_boot} replicates)")
    print(line("overall", res["overall"]))
    print(line("OCR/HTR average", res["recognition_avg"]))
    for k, v in res["tracks"].items():
        print(line("  " + k, v))
    for k, v in res["subsets"].items():
        print(line("    " + k, v))

def _print_scores(s: dict):
    def f(v):
        return "  —  " if v is None else f"{v:5.1f}"
    print(f"overall {f(s['overall'])} | recognition avg {f(s['recognition_avg'])}")
    print("tracks: " + "  ".join(f"{k}={f(v)}" for k, v in s["tracks"].items()))
    for sid, v in s["subsets"].items():
        if v is None:
            print(f"  {sid:20s}   (not supported)")
        else:
            extra = f"CER {v['cer']:.3f}" if "cer" in v else (f"macro-F1 {v['macro_f1']:.3f}" if "macro_f1" in v else f"chrF++ {v['chrf_pp']:.1f}")
            print(f"  {sid:20s} {v['score']:6.2f}  [{v['ci95'][0]:.1f}, {v['ci95'][1]:.1f}]  {extra}  n={v['n']}")


def cmd_leaderboard(a):
    from . import leaderboard
    lb = leaderboard.build(Path(a.results), Path(a.data), Path(a.out), readme=Path(a.readme) if a.readme else None,
                           include_debug=a.include_debug)
    print(f"leaderboard: {sum(1 for m in lb['models'] if m['status'] == 'evaluated')} evaluated, "
          f"{sum(1 for m in lb['models'] if m['status'] != 'evaluated')} awaiting evaluation -> {a.out}")


def cmd_import(a):
    from . import runner
    out = runner.import_predictions(Path(a.file), model_id=a.model, name=a.name, supports=a.supports,
                                    split=a.split, results_root=Path(a.results), notes=a.notes)
    s = runner.score(out, Path(a.data))
    _print_scores(s)


def cmd_fetch_tessdata(a):
    import httpx
    dest = Path(a.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    for path, name in (("tam.traineddata", "tam.traineddata"), ("script/Tamil.traineddata", "Tamil.traineddata"),
                       ("eng.traineddata", "eng.traineddata")):
        target = dest / name
        if target.exists():
            print(f"have {target}")
            continue
        r = httpx.get(TESSDATA_BEST_URL.format(path=path), follow_redirects=True, timeout=300)
        r.raise_for_status()
        target.write_bytes(r.content)
        print(f"fetched {target} ({len(r.content) // 1024} KB)")


def cmd_validate(a):
    from .runner import load_manifest
    from .subsets import get
    data = Path(a.data)
    bad = 0
    for split in ("test", "lite"):
        mp = data / f"manifest-{split}.jsonl"
        if not mp.exists():
            continue
        rows = load_manifest(mp)
        for r in rows:
            p = data / r["image"]
            spec = get(r["subset"])
            if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != r["sha256"]:
                print(f"checksum mismatch: {r['id']}")
                bad += 1
            if not r.get(spec.target):
                print(f"empty reference: {r['id']}")
                bad += 1
        print(f"{split}: {len(rows)} samples checked")
    if bad:
        sys.exit(f"{bad} problems")
    print("ok")


def cmd_list_models(a):
    from .models import load_registry
    for e in load_registry():
        flag = " (verify id)" if e.get("verify_id") else ""
        print(f"{e['id']:32s} {e['provider']:12s} {e['model']}{flag}")


def cmd_list_subsets(a):
    from .subsets import SUBSETS
    for s in SUBSETS:
        print(f"{s.id:20s} {s.task.value:12s} {s.track.value:15s} {s.provenance.value:9s} n={s.count:<4d} {s.title}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tamilbench", description="Tamil OCR / HTR / epigraphy benchmark")
    p.add_argument("--version", action="version", version=f"tamilbench {__version__} (benchmark {BENCHMARK_VERSION})")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="render the synthetic subsets and import the real ones")
    b.add_argument("--out", default=str(DATA))
    b.add_argument("--split", default="test", help="test (public) or private (use a secret --seed)")
    b.add_argument("--seed", type=int, default=20261002)
    b.add_argument("--subsets", nargs="*")
    b.add_argument("--count", type=int, help="override every subset's sample count")
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--cict-root", help="path to a cict-htr directory (for palm-leaf-cict)")
    b.set_defaults(fn=cmd_build)

    r = sub.add_parser("run", help="evaluate a model (and score it)")
    r.add_argument("--model", required=True, help="registry id (see list-models) or provider:model[?k=v]")
    r.add_argument("--split", default="test", choices=["test", "lite", "private"])
    r.add_argument("--data", default=str(DATA))
    r.add_argument("--results", default=str(RESULTS))
    r.add_argument("--subsets", nargs="*")
    r.add_argument("--limit", type=int, help="max samples per subset (quick checks)")
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--param", action="append", help="adapter parameter override, key=value")
    r.add_argument("--no-resume", action="store_true")
    r.add_argument("--no-score", action="store_true")
    r.add_argument("--n-boot", type=int, default=1000)
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("score", help="(re)score predictions")
    s.add_argument("results_dirs", nargs="*")
    s.add_argument("--results", default=str(RESULTS))
    s.add_argument("--split", default="test")
    s.add_argument("--data", default=str(DATA))
    s.add_argument("--n-boot", type=int, default=1000)
    s.set_defaults(fn=cmd_score)

    c = sub.add_parser("compare", help="paired significance test between two runs (A − B)")
    c.add_argument("a", help="model id under --results, or a results directory")
    c.add_argument("b")
    c.add_argument("--results", default=str(RESULTS))
    c.add_argument("--data", default=str(DATA))
    c.add_argument("--split", default="test")
    c.add_argument("--n-boot", type=int, default=1000)
    c.add_argument("--json", help="also write the full comparison to this file")
    c.set_defaults(fn=cmd_compare)

    lb = sub.add_parser("leaderboard", help="aggregate scores into leaderboard.json and the README table")
    lb.add_argument("--results", default=str(RESULTS))
    lb.add_argument("--data", default=str(DATA))
    lb.add_argument("--out", default="leaderboard/data/leaderboard.json")
    lb.add_argument("--readme", default="README.md")
    lb.add_argument("--include-debug", action="store_true")
    lb.set_defaults(fn=cmd_leaderboard)

    im = sub.add_parser("import-predictions", help="score predictions produced outside this harness")
    im.add_argument("--model", required=True, help="results id, e.g. sarvam-vision-2-1")
    im.add_argument("--name")
    im.add_argument("--file", required=True, help="JSONL with id and text per line")
    im.add_argument("--supports", nargs="+", default=["recognition"])
    im.add_argument("--split", default="test")
    im.add_argument("--data", default=str(DATA))
    im.add_argument("--results", default=str(RESULTS))
    im.add_argument("--notes")
    im.set_defaults(fn=cmd_import)

    t = sub.add_parser("fetch-tessdata", help="download Tesseract tessdata_best models for Tamil")
    t.add_argument("--dest", default="~/.cache/tamilbench/tessdata_best")
    t.set_defaults(fn=cmd_fetch_tessdata)

    v = sub.add_parser("validate", help="verify image checksums and references")
    v.add_argument("--data", default=str(DATA))
    v.set_defaults(fn=cmd_validate)

    lm = sub.add_parser("list-models")
    lm.set_defaults(fn=cmd_list_models)
    ls = sub.add_parser("list-subsets")
    ls.set_defaults(fn=cmd_list_subsets)

    a = p.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
