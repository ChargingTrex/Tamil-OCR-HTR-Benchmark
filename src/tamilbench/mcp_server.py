"""MCP server for the Tamil OCR / HTR benchmark. Start it with ``tamilbench mcp``.

It lets Claude, or any other MCP client:

* browse the subsets and items and see each test image with the exact prompt a model gets;
* take the benchmark itself: fetch the next unanswered item, answer it, and have the
  answer scored by the benchmark's own code. These *interactive* runs are kept off the
  ranked leaderboard, because a chat client adds its own system prompt and tools;
* run any registered system (Claude, GPT, Gemini, open-weights models, OCR engines)
  through its adapter, import outputs produced elsewhere, score runs, compare two runs
  with paired statistics, and read diagnostics and the leaderboard.

Model API keys are read from the server's environment, never from tool arguments.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import shutil
import sys
import threading
import time
import uuid
from contextlib import redirect_stdout
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

import anyio
from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import Field

from . import BENCHMARK_VERSION, CANARY, __version__
from . import subsets as S
from .prompts import PROMPT_VERSION, PROMPTS, SYSTEM
from .runner import load_manifest, manifest_path, read_predictions, results_dir
from .scoring import item_result
from .taxonomy import ERAS, LINEAGE_ORDER, SCRIPTS, TRACKS, Era, Script, Task, Track

_REPO = Path(__file__).resolve().parents[2]
_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,99}$")
_MAX_IMPORT_BYTES = 50_000_000
TASKS_ALL = [t.value for t in Task]

INSTRUCTIONS = """\
Tamil OCR / HTR Benchmark (tamilbench): test images of Tamil writing from Tamil-Brahmi
inscriptions and palm leaves to screens, in 18 ranked subsets, 8 tracks and 2 control
subsets. The modern script and the older scripts count equally: OCR/HTR = ½ modern-script
reading + ½ older-script reading; tamilbench_get_script_scores shows every script stage.

Workflows:
1. Explore: tamilbench_list_subsets -> tamilbench_get_subset -> tamilbench_list_items ->
   tamilbench_get_item (shows the image and the exact prompt).
2. Take the benchmark yourself: tamilbench_next_item(run_name) -> answer from the image
   alone, exactly as the prompt asks -> tamilbench_submit_answer -> repeat ->
   tamilbench_score_run. Do not ask for references while taking the test.
3. Test another model: tamilbench_list_models (shows which have credentials) ->
   tamilbench_run_model -> tamilbench_run_status -> tamilbench_score_run,
   tamilbench_compare_runs, tamilbench_update_leaderboard. Outputs produced elsewhere:
   tamilbench_import_predictions.
Use split "lite" for quick checks; only "test" is ranked.
"""

mcp = MCPServer(name="tamilbench_mcp", title="Tamil OCR / HTR Benchmark", version=__version__,
                instructions=INSTRUCTIONS)


class Split(str, Enum):
    TEST = "test"
    LITE = "lite"


class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"


class ModelStatus(str, Enum):
    ALL = "all"
    EVALUATED = "evaluated"
    PENDING = "pending"


Fmt = Annotated[ResponseFormat, Field(description="'markdown' for reading, 'json' for programmatic use")]
SplitArg = Annotated[Split, Field(description="'test' (the ranked split) or 'lite' (about 20 items per subset, "
                                              "for quick checks)")]
RunName = Annotated[str, Field(description="Run identifier (the results/ folder name): letters, digits, '.', '_' "
                                           "or '-', e.g. 'claude-desktop-2026-10' or 'tesseract-tam-best'",
                                min_length=2, max_length=100)]
MaxSide = Annotated[int | None, Field(description="Downscale the image so its longer side is at most this many "
                                                  "pixels (saves context); omit to get the original, which is what "
                                                  "official runs use", ge=256, le=4096)]

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
WRITES_LOCAL = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False)


# ------------------------------------------------------------------------------- paths & data

def data_dir() -> Path:
    env = os.environ.get("TAMILBENCH_DATA")
    if env:
        return Path(env)
    for cand in (_REPO / "data" / BENCHMARK_VERSION, Path.cwd() / "data" / BENCHMARK_VERSION):
        if (cand / "manifest-test.jsonl").exists():
            return cand
    return _REPO / "data" / BENCHMARK_VERSION


def results_root() -> Path:
    env = os.environ.get("TAMILBENCH_RESULTS")
    return Path(env) if env else _REPO / "results"


@lru_cache(maxsize=4)
def _manifest(split: str) -> tuple[dict, ...]:
    path = manifest_path(data_dir(), split)
    if not path.exists():
        raise ToolError(f"No manifest for split '{split}' at {path}. Build the data with `tamilbench build`, "
                        "or point TAMILBENCH_DATA at a data/v1 directory.")
    return tuple(load_manifest(path))


@lru_cache(maxsize=4)
def _by_id(split: str) -> dict[str, dict]:
    return {r["id"]: r for r in _manifest(split)}


def _row(item_id: str, split: str) -> dict:
    row = _by_id(split).get(item_id)
    if row is None:
        other = "lite" if split == "test" else "test"
        hint = (f" It is in split '{other}'." if item_id in _by_id(other)
                else " Use tamilbench_list_items to find valid ids.")
        raise ToolError(f"Unknown item '{item_id}' in split '{split}'.{hint}")
    return row


def _spec(subset_id: str) -> S.SubsetSpec:
    try:
        return S.get(subset_id)
    except KeyError:
        raise ToolError(f"Unknown subset '{subset_id}'. Valid ids: {', '.join(S.BY_ID)}.") from None


def _run_dir(run_name: str, split: str) -> Path:
    if not _RUN_NAME.match(run_name):
        raise ToolError("run_name must be 2-100 characters of letters, digits, '.', '_' or '-', starting with a "
                        "letter or digit.")
    return results_dir(results_root(), run_name, split)


def _run_meta(run_dir: Path) -> dict | None:
    p = run_dir / "run.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _image(row: dict, max_side: int | None) -> Image:
    path = data_dir() / row["image"]
    if not path.exists():
        raise ToolError(f"Image file missing: {row['image']}. Run `tamilbench validate`.")
    fmt = "png" if path.suffix.lower() == ".png" else "jpeg"
    if not max_side or max(row.get("width", 0), row.get("height", 0)) <= max_side:
        return Image(data=path.read_bytes(), format=fmt)
    from PIL import Image as PILImage
    im = PILImage.open(path)
    im.thumbnail((max_side, max_side), PILImage.LANCZOS)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "PNG" if fmt == "png" else "JPEG", quality=90)
    return Image(data=buf.getvalue(), format=fmt)


def _policy_text(spec: S.SubsetSpec) -> str:
    if spec.scoring == "abstention":
        return "100 − % of items answered with any letters (the right answer is no text); not ranked"
    if spec.id == "perturbed":
        return "100·(1−CER) with the policy of the subset each item imitates, plus the prior-pull index; not ranked"
    if spec.task == Task.SCRIPT_ID or spec.task == Task.MEDIUM_ID:
        return "macro-F1 over the label set"
    if spec.task == Task.TRANSLATION:
        return "corpus chrF++"
    p = spec.policy
    parts = ["spaces removed" if p.get("spaces") == "remove" else "spaces collapsed"]
    if p.get("fold_pulli"):
        parts.append("puḷḷi folded")
    if p.get("fold_vowel_length"):
        parts.append("e/ē o/ō folded")
    if p.get("strip_tamil_numerals"):
        parts.append("Tamil numerals stripped")
    if p.get("script") == "iast":
        parts.append("IAST folding")
    return "100·(1−CER); " + ", ".join(parts)


def _dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _fmt(v: float | None, nd: int = 1) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


# ------------------------------------------------------------------------------- browsing

@mcp.tool(name="tamilbench_list_subsets", annotations=READ_ONLY, structured_output=False)
def tamilbench_list_subsets(
    track: Annotated[Track | None, Field(description="Only subsets in this track")] = None,
    task: Annotated[Task | None, Field(description="Only subsets of this task")] = None,
    response_format: Fmt = ResponseFormat.MARKDOWN,
) -> str:
    """List the benchmark's 18 ranked subsets and 2 control subsets with their track, task,
    item count, provenance and scoring policy. Use tamilbench_get_subset for the full
    description and the prompt."""
    counts = {}
    for r in _manifest("test"):
        counts[r["subset"]] = counts.get(r["subset"], 0) + 1
    rows = [{"id": s.id, "title": s.title, "track": s.track.value, "task": s.task.value,
             "n_test": counts.get(s.id, 0), "provenance": s.provenance.value, "prompt": s.prompt,
             "scoring": _policy_text(s)}
            for s in S.SUBSETS if (track is None or s.track == track) and (task is None or s.task == task)]
    if response_format == ResponseFormat.JSON:
        return _dump({"count": len(rows), "subsets": rows})
    lines = [f"# Subsets ({len(rows)})", "", "| id | title | track | task | n | provenance | scoring |",
             "|---|---|---|---|---:|---|---|"]
    lines += [f"| `{r['id']}` | {r['title']} | {TRACKS[Track(r['track'])].name} | {r['task']} | {r['n_test']} | "
              f"{r['provenance']} | {r['scoring']} |" for r in rows]
    return "\n".join(lines)


@mcp.tool(name="tamilbench_get_subset", annotations=READ_ONLY, structured_output=False)
def tamilbench_get_subset(
    subset_id: Annotated[str, Field(description="Subset id, e.g. 'palm-leaf-cict' or 'grantha-tamil'")],
    response_format: Fmt = ResponseFormat.MARKDOWN,
) -> str:
    """Describe one subset: what it tests, how it is scored, the exact prompt models see,
    its labels (identification tasks), counts per split, attribution and sample item ids."""
    spec = _spec(subset_id)
    test_ids = [r["id"] for r in _manifest("test") if r["subset"] == spec.id]
    lite_n = sum(1 for r in _manifest("lite") if r["subset"] == spec.id)
    info = {"id": spec.id, "title": spec.title, "track": spec.track.value, "task": spec.task.value,
            "description": spec.description, "provenance": spec.provenance.value,
            "scripts": [s.value for s in spec.scripts], "media": [m.value for m in spec.media],
            "granularity": [g.value for g in spec.granularity], "n_test": len(test_ids), "n_lite": lite_n,
            "scoring": _policy_text(spec), "policy": spec.policy, "labels": list(spec.labels),
            "prompt_id": spec.prompt, "prompt": PROMPTS[spec.prompt], "system_prompt": SYSTEM,
            "ranked": spec.ranked, "attribution": spec.attribution or None, "sample_ids": test_ids[:5]}
    if not spec.ranked:
        imitated = sorted({r.get("as_subset") for r in _manifest("test") if r["subset"] == spec.id} - {None})
        info.update({"prompt_id": None, "prompt": None, "imitates": imitated})
    if response_format == ResponseFormat.JSON:
        return _dump(info)
    lines = [f"# {spec.title} (`{spec.id}`)", "", spec.description, "",
             f"- **Track:** {TRACKS[spec.track].name} · **task:** {spec.task.value} · **provenance:** "
             f"{spec.provenance.value}",
             f"- **Items:** {len(test_ids)} in test, {lite_n} in lite · granularity "
             f"{', '.join(info['granularity'])}",
             f"- **Scripts:** {', '.join(info['scripts'])} · **media:** {', '.join(info['media'])}",
             f"- **Scoring:** {info['scoring']}"]
    if spec.labels:
        lines.append(f"- **Labels:** {', '.join(spec.labels)}")
    if spec.attribution:
        lines.append(f"- **Source and credit:** {spec.attribution}")
    if spec.ranked:
        lines += ["", f"## Prompt (`{spec.prompt}`)", "", "```text", PROMPTS[spec.prompt], "```"]
    else:
        lines += ["", "## Prompt", "", "Control subset, not ranked: each item is shown with the prompt of the subset it "
                  "imitates, so a system cannot tell a control from a test item. Imitated subsets: "
                  + ", ".join(f"`{x}`" for x in info["imitates"]) + "."]
    lines += ["", f"Sample ids: {', '.join(info['sample_ids'])}"]
    return "\n".join(lines)


@mcp.tool(name="tamilbench_list_items", annotations=READ_ONLY, structured_output=False)
def tamilbench_list_items(
    subset_id: Annotated[str, Field(description="Subset id, e.g. 'tamil-brahmi'")],
    split: SplitArg = Split.TEST,
    offset: Annotated[int, Field(description="Items to skip, for paging", ge=0)] = 0,
    limit: Annotated[int, Field(description="Items to return", ge=1, le=100)] = 20,
    include_reference: Annotated[bool, Field(description="Also return the reference answers. Leave false when "
                                                         "the items will be answered by a model in this session")]
    = False,
    response_format: Fmt = ResponseFormat.MARKDOWN,
) -> str:
    """List the items of a subset with their medium, granularity, lexical kind (real text,
    random words or nonce words) and image size, one page at a time."""
    spec = _spec(subset_id)
    rows = [r for r in _manifest(split.value) if r["subset"] == spec.id]
    page = rows[offset:offset + limit]
    items = []
    for r in page:
        it = {"id": r["id"], "medium": r.get("medium"), "script": r.get("script"),
              "granularity": r.get("granularity"), "lexical": r.get("lexical"),
              "provenance": r.get("provenance"), "size": [r.get("width"), r.get("height")]}
        if include_reference:
            it["reference"] = r.get(S.presentation(r).target)
        items.append(it)
    more = offset + len(page) < len(rows)
    out = {"subset": spec.id, "split": split.value, "total": len(rows), "count": len(items), "offset": offset,
           "has_more": more, "next_offset": offset + len(page) if more else None, "items": items}
    if response_format == ResponseFormat.JSON:
        return _dump(out)
    lines = [f"# {spec.title}: items {offset + 1}–{offset + len(page)} of {len(rows)} ({split.value})", ""]
    for it in items:
        ref = f" — reference: {it['reference']!r}" if include_reference else ""
        lines.append(f"- `{it['id']}` · {it['medium']} · {it['granularity']} · {it['lexical']} · "
                     f"{it['size'][0]}×{it['size'][1]}{ref}")
    if more:
        lines += ["", f"More items: call again with offset={out['next_offset']}."]
    return "\n".join(lines)


def _item_text(row: dict, split: str, *, include_reference: bool, run_name: str | None = None) -> str:
    # Control items are presented as the subset they imitate, with that subset's prompt.
    spec = S.presentation(row)
    lines = [f"Item `{row['id']}` · {spec.title} · split {split}",
             f"Task: {spec.task.value} · medium {row.get('medium')} · granularity {row.get('granularity')} · "
             f"image {row.get('width')}×{row.get('height')}",
             "", "System prompt:", SYSTEM, "", "Prompt:", PROMPTS[spec.prompt]]
    if spec.attribution:
        lines += ["", f"Source and credit: {spec.attribution}"]
    if include_reference:
        lines += ["", f"Reference: {row.get(spec.target)!r}"]
    if run_name:
        lines += ["", f"Answer with tamilbench_submit_answer(run_name='{run_name}', item_id='{row['id']}', "
                      f"split='{split}', answer=<only what the prompt asks for>)."]
    return "\n".join(lines)


@mcp.tool(name="tamilbench_get_item", annotations=READ_ONLY, structured_output=False)
def tamilbench_get_item(
    item_id: Annotated[str, Field(description="Item id, e.g. 'palm-leaf-cict-009' or 'scene-0004'")],
    split: SplitArg = Split.TEST,
    include_reference: Annotated[bool, Field(description="Also show the reference answer. Leave false if you "
                                                         "are going to answer the item yourself")] = False,
    max_side: MaxSide = None,
) -> list[str | Image]:
    """Show one test item: the image together with the exact system prompt and task prompt
    that every evaluated model receives, plus its metadata."""
    row = _row(item_id, split.value)
    return [_item_text(row, split.value, include_reference=include_reference), _image(row, max_side)]


# ------------------------------------------------------------------------------- taking the test

def _ensure_interactive_run(run_dir: Path, run_name: str, split: str, model_name: str | None) -> dict:
    meta = _run_meta(run_dir)
    if meta is not None:
        if meta.get("model", {}).get("kind") != "interactive":
            raise ToolError(f"'{run_name}' already holds a {meta.get('model', {}).get('kind', 'model')} run, so "
                            "answers cannot be added to it. Choose another run_name.")
        return meta
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model": {"id": run_name, "provider": "mcp", "model": model_name or run_name, "kind": "interactive",
                      "supports": TASKS_ALL, "params": {"client": "MCP"}},
            "benchmark_version": BENCHMARK_VERSION, "split": split, "prompt_version": PROMPT_VERSION,
            "tamilbench_version": __version__, "started_at": _now(),
            "notes": "Answers submitted through the MCP server by an interactive client; not ranked."}
    (run_dir / "run.json").write_text(_dump(meta), encoding="utf-8")
    return meta


@mcp.tool(name="tamilbench_next_item", annotations=READ_ONLY, structured_output=False)
def tamilbench_next_item(
    run_name: RunName,
    split: SplitArg = Split.LITE,
    subset_id: Annotated[str | None, Field(description="Only items of this subset")] = None,
    max_side: MaxSide = None,
) -> list[str | Image]:
    """Return the next item that the run has not answered yet: its image and the exact
    prompt. Answer from the image alone and submit with tamilbench_submit_answer; when
    every item is answered, this says so and points to tamilbench_score_run."""
    if subset_id:
        _spec(subset_id)
    run_dir = _run_dir(run_name, split.value)
    done = read_predictions(run_dir / "predictions.jsonl")
    rows = [r for r in _manifest(split.value) if not subset_id or r["subset"] == subset_id]
    pending = [r for r in rows if r["id"] not in done]
    if not pending:
        return [f"All {len(rows)} items{f' of {subset_id}' if subset_id else ''} in split '{split.value}' are "
                f"answered for run '{run_name}'. Score them with tamilbench_score_run(run_name='{run_name}', "
                f"split='{split.value}')."]
    row = pending[0]
    head = f"Progress: {len(rows) - len(pending)} of {len(rows)} answered.\n\n"
    return [head + _item_text(row, split.value, include_reference=False, run_name=run_name),
            _image(row, max_side)]


@mcp.tool(name="tamilbench_submit_answer", annotations=WRITES_LOCAL, structured_output=False)
def tamilbench_submit_answer(
    run_name: RunName,
    item_id: Annotated[str, Field(description="The item being answered")],
    answer: Annotated[str, Field(description="Only what the prompt asks for: the transcription, the label or the "
                                             "translation, with no commentary", max_length=20000)],
    split: SplitArg = Split.LITE,
    model_name: Annotated[str | None, Field(description="Who is answering, e.g. 'Claude (Claude Desktop)'; "
                                                        "recorded in run.json", max_length=100)] = None,
    show_reference: Annotated[bool, Field(description="Reveal the reference after scoring. Leave false while "
                                                      "taking the test: items can share source texts")] = False,
) -> str:
    """Record an answer for one item in an interactive run and score it with the official
    per-item metric (CER for reading, correct/incorrect label, or chrF++). Submitting again
    replaces the earlier answer. Interactive runs are stored under results/ but never ranked."""
    row = _row(item_id, split.value)
    run_dir = _run_dir(run_name, split.value)
    _ensure_interactive_run(run_dir, run_name, split.value, model_name)
    rec = {"id": row["id"], "subset": row["subset"], "text": answer, "submitted_at": _now(), "source": "mcp"}
    with open(run_dir / "predictions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    spec = S.get(row["subset"])
    res = item_result(spec, row, answer)
    done = read_predictions(run_dir / "predictions.jsonl")
    total = len(_manifest(split.value))
    if "hallucinated" in res:
        verdict = ("no text claimed: correct" if not res["hallucinated"]
                   else f"{res['letters']} letters written where nothing is legible")
    elif "cer" in res:
        verdict = (f"CER {res['cer']:.3f} ({res['char_edits']} edits over {res['chars']} reference characters, "
                   f"after the subset's normalisation: {_policy_text(S.presentation(row))})")
    elif "ok" in res:
        verdict = f"label parsed as '{res['label']}': {'correct' if res['ok'] else 'incorrect'}"
    else:
        verdict = f"chrF++ {res['chrf']:.1f}"
    out = [f"Recorded `{row['id']}` for run '{run_name}': {verdict}.",
           f"Progress: {len(done)} of {total} items in '{split.value}' answered."]
    if show_reference:
        out.append(f"Reference: {row.get(S.presentation(row).target)!r}")
    out.append(f"Next: tamilbench_next_item(run_name='{run_name}', split='{split.value}').")
    return "\n".join(out)


# ------------------------------------------------------------------------------- scoring

def _ci_text(ci) -> str:
    return f"{_fmt(ci[0])}–{_fmt(ci[1])}" if ci else ""


def _script_rows(scripts: dict) -> list[str]:
    lines = ["| Script stage | Era | Reading | 95 % CI | Items |", "|---|---|---:|---|---:|"]
    for sc in sorted(scripts, key=lambda x: [s.value for s in LINEAGE_ORDER].index(x)
                     if x in [s.value for s in LINEAGE_ORDER] else 99):
        v = scripts[sc]
        name = SCRIPTS[Script(sc)].name if sc in Script._value2member_map_ else sc
        lines.append(f"| {name} | {v.get('era', '')} | {_fmt(v.get('score'))} | {_ci_text(v.get('ci95'))} | "
                     f"{v.get('n', '')} |")
    return lines


def _score_summary(scores: dict, *, title: str, answered: int | None = None, total: int | None = None) -> str:
    lines = [f"# {title}", ""]
    if answered is not None:
        lines.append(f"Scored on the {answered} answered items of {total}.")
    eras = scores.get("eras") or {}
    m, o = eras.get("modern", {}), eras.get("older", {})
    lines += [f"**Overall:** {_fmt(scores.get('overall'))} · **OCR/HTR:** {_fmt(scores.get('recognition_avg'))} "
              f"= ½ modern script {_fmt(m.get('reading'))} + ½ older scripts {_fmt(o.get('reading'))}"]
    gap = scores.get("era_gap")
    if gap:
        lines.append(f"Modern − older reading: {gap['diff']:+.1f} (95 % CI {gap['ci95'][0]:+.1f} to "
                     f"{gap['ci95'][1]:+.1f}){'' if gap['separable'] else ', not significant'}")
    if scores.get("scripts"):
        lines += [""] + _script_rows(scores["scripts"])
    lines += ["", "| Track | Score |", "|---|---:|"]
    lines += [f"| {TRACKS[Track(k)].name} | {_fmt(v)} |" for k, v in scores["tracks"].items()]
    lines += ["", "| Subset | n | Score | 95 % CI |", "|---|---:|---:|---|"]
    for sid, v in scores["subsets"].items():
        if v is None:
            lines.append(f"| `{sid}` | — | not supported | |")
        else:
            ci = v.get("ci95") or [None, None]
            lines.append(f"| `{sid}` | {v.get('n', '')} | {_fmt(v['score'])} | {_fmt(ci[0])}–{_fmt(ci[1])} |")
    fm = scores.get("failure_modes")
    if fm:
        lines += ["", "**Failure modes** (share of reading items): " + ", ".join(
            f"{k.replace('_rate', '').replace('_', ' ')} {100 * v:.1f} %" for k, v in fm.items() if k != "n")]
    pr = scores.get("prior_reliance") or {}
    if pr.get("corpus_text") is not None:
        lines.append(f"**Reading vs guessing:** real text {_fmt(pr['corpus_text'])}, nonce words "
                     f"{_fmt(pr.get('nonce_words'))}")
    if pr.get("recitation_index"):
        lines.append("**Recitation index** (−1 reads the image, +1 recites the canonical text): " + ", ".join(
            f"{k} {v:+.2f}" for k, v in pr["recitation_index"].items()))
    hal = scores.get("hallucination")
    if hal:
        lines.append(f"**Hallucination on blank and effaced surfaces:** {100 * hal['rate']:.1f} % of "
                     f"{hal['n']} control items answered with letters")
    return "\n".join(lines)


def _score(run_name: str, split: str, answered_only: bool | None, n_boot: int) -> tuple[dict, str]:
    from . import runner
    from .scoring import score_run
    run_dir = _run_dir(run_name, split)
    meta = _run_meta(run_dir)
    if meta is None or not (run_dir / "predictions.jsonl").exists():
        raise ToolError(f"No run '{run_name}' on split '{split}'. Start one with tamilbench_run_model, "
                        "tamilbench_submit_answer or tamilbench_import_predictions.")
    interactive = meta.get("model", {}).get("kind") == "interactive"
    if answered_only is None:
        answered_only = interactive
    if not answered_only:
        s = runner.score(run_dir, data_dir(), split=split, n_boot=n_boot)
        return s, _score_summary(s, title=f"{run_name} · {split}")
    preds_raw = read_predictions(run_dir / "predictions.jsonl")
    rows = [r for r in _manifest(split) if r["id"] in preds_raw]
    if not rows:
        raise ToolError(f"Run '{run_name}' has no answers on split '{split}' yet.")
    supported = set(meta["model"].get("supports") or []) or None
    s = score_run(rows, {k: v.get("text") for k, v in preds_raw.items()}, supported_tasks=supported,
                  n_boot=n_boot, keep_per_sample=False)
    s.update({"answered_only": True, "n_answered": len(rows), "n_total": len(_manifest(split)),
              "model": meta["model"], "split": split, "scored_at": _now()})
    (run_dir / "scores-answered.json").write_text(_dump(s), encoding="utf-8")
    return s, _score_summary(s, title=f"{run_name} · {split} (answered items)", answered=len(rows),
                             total=len(_manifest(split)))


@mcp.tool(name="tamilbench_score_run", annotations=WRITES_LOCAL, structured_output=False)
async def tamilbench_score_run(
    run_name: RunName,
    split: SplitArg = Split.TEST,
    answered_only: Annotated[bool | None, Field(description="Score only the items that have answers. Default: "
                                                            "true for interactive runs, false otherwise (missing "
                                                            "answers then count as empty, as in official scoring)")]
    = None,
    response_format: Fmt = ResponseFormat.MARKDOWN,
) -> str:
    """Score a run with the official code: per-subset scores with cluster-bootstrap 95 %
    intervals, per-script and per-era scores, track scores, Overall and OCR/HTR (modern and
    older scripts weighted equally), failure modes and the reading-vs-guessing and
    hallucination diagnostics. Writes scores.json (or scores-answered.json)."""
    s, md = await anyio.to_thread.run_sync(lambda: _score(run_name, split.value, answered_only, 1000))
    if response_format == ResponseFormat.JSON:
        return _dump({k: v for k, v in s.items()})
    return md


@mcp.tool(name="tamilbench_get_diagnostics", annotations=READ_ONLY, structured_output=False)
def tamilbench_get_diagnostics(
    run_name: RunName,
    split: SplitArg = Split.TEST,
    subset_id: Annotated[str | None, Field(description="Only this subset (adds its letter confusions)")] = None,
    response_format: Fmt = ResponseFormat.MARKDOWN,
) -> str:
    """Show what the error rate hides for a scored run: empty, overlong and looping answers,
    Markdown, wrong-script letters, the order-free error and reading-order gap, sample-level
    NED, the most-confused Tamil letters and the recitation index on the real leaves."""
    run_dir = _run_dir(run_name, split.value)
    path = next((p for p in (run_dir / "scores.json", run_dir / "scores-answered.json") if p.exists()), None)
    if path is None:
        raise ToolError(f"Run '{run_name}' has not been scored on '{split.value}'. Call tamilbench_score_run first.")
    s = json.loads(path.read_text(encoding="utf-8"))
    subs = {k: v for k, v in s["subsets"].items() if v and "diagnostics" in v and (not subset_id or k == subset_id)}
    if subset_id and subset_id not in subs:
        raise ToolError(f"No reading diagnostics for '{subset_id}' in this run (unknown, unsupported or not a "
                        "reading subset).")
    data = {"run": run_name, "split": split.value, "failure_modes": s.get("failure_modes"),
            "recitation_index": (s.get("prior_reliance") or {}).get("recitation_index"),
            "subsets": {k: {"score": v["score"], **v["diagnostics"],
                            **({"confusions": v.get("confusions")} if subset_id else {})} for k, v in subs.items()}}
    if response_format == ResponseFormat.JSON:
        return _dump(data)
    lines = [f"# Diagnostics: {run_name} · {split.value}", ""]
    if data["failure_modes"]:
        lines.append("Failure modes over all reading items: " + ", ".join(
            f"{k.replace('_rate', '').replace('_', ' ')} {100 * v:.1f} %"
            for k, v in data["failure_modes"].items() if k != "n"))
    if data["recitation_index"]:
        lines.append("Recitation index (−1 reads, +1 recites): " + ", ".join(
            f"{k} {v:+.2f}" for k, v in data["recitation_index"].items()))
    lines += ["", "| Subset | Score | CER (no spaces) | Order-free | Order gap | NED | Wrong-script items | "
                  "Empty | Loops |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, v in data["subsets"].items():
        lines.append(f"| `{k}` | {_fmt(v['score'])} | {v['cer_nospace']:.3f} | {v['bag_cer']:.3f} | "
                     f"{v['order_gap']:.3f} | {v['ned']:.3f} | {100 * v['offscript_item_rate']:.1f} % | "
                     f"{100 * v['empty_rate']:.1f} % | {100 * v['repetition_rate']:.1f} % |")
    if subset_id:
        conf = data["subsets"][subset_id].get("confusions") or {}
        subs_list = ", ".join(f"{a}→{b} ×{n}" for a, b, n in conf.get("substitutions", [])) or "none"
        lines += ["", f"Most frequent letter substitutions in `{subset_id}`: {subs_list}"]
    return "\n".join(lines)


@mcp.tool(name="tamilbench_compare_runs", annotations=READ_ONLY, structured_output=False)
async def tamilbench_compare_runs(
    run_a: RunName,
    run_b: RunName,
    split: SplitArg = Split.TEST,
    response_format: Fmt = ResponseFormat.MARKDOWN,
) -> str:
    """Compare two runs on paired differences (A − B) with a cluster bootstrap: per subset,
    per track, per script stage, for the modern and older halves, OCR/HTR and Overall, each
    with a 95 % interval and a p-value. An interval that includes zero is a statistical tie."""
    from .compare import compare

    def load(name):
        d = _run_dir(name, split.value)
        meta = _run_meta(d)
        if meta is None:
            raise ToolError(f"No run '{name}' on split '{split.value}'.")
        preds = {k: v.get("text") for k, v in read_predictions(d / "predictions.jsonl").items()}
        return preds, set(meta["model"].get("supports") or []) or None

    (pa, sa), (pb, sb) = load(run_a), load(run_b)
    res = await anyio.to_thread.run_sync(
        lambda: compare(list(_manifest(split.value)), pa, pb, supported_a=sa, supported_b=sb))
    if response_format == ResponseFormat.JSON:
        return _dump(res)

    def row(name, r):
        if not r:
            return f"| {name} | — | | | |"
        return (f"| {name} | {r['diff']:+.2f} | {r['ci95'][0]:+.2f} to {r['ci95'][1]:+.2f} | {r['p']:.3f} | "
                f"{'yes' if r['separable'] else 'no (tie)'} |")
    lines = [f"# {run_a} − {run_b} ({split.value})", "", "| | Difference | 95 % CI | p | Separable |",
             "|---|---:|---|---:|---|", row("**Overall**", res["overall"]), row("**OCR/HTR**",
                                                                              res["recognition_avg"])]
    lines += [row(f"{ERAS[Era(e)].name} (reading)", parts.get("reading")) for e, parts in res["eras"].items()]
    lines += [row(SCRIPTS[Script(k)].name if k in Script._value2member_map_ else k, v)
              for k, v in res["scripts"].items()]
    lines += [row(TRACKS[Track(k)].name, v) for k, v in res["tracks"].items()]
    lines += [row(f"`{k}`", v) for k, v in res["subsets"].items()]
    return "\n".join(lines)


# ------------------------------------------------------------------------------- other models

_CREDENTIALS: dict[str, list[tuple[str, ...]]] = {
    "anthropic": [("ANTHROPIC_API_KEY",)], "openai": [("OPENAI_API_KEY",)],
    "google": [("GEMINI_API_KEY", "GOOGLE_API_KEY")], "gemini": [("GEMINI_API_KEY", "GOOGLE_API_KEY")],
    "mistral-ocr": [("MISTRAL_API_KEY",)], "google-vision": [("GOOGLE_VISION_API_KEY",)],
    "azure-di": [("AZURE_DI_ENDPOINT",), ("AZURE_DI_KEY",)],
}


def availability(entry: dict) -> tuple[bool, str]:
    """Whether a registry entry can run in this server's environment, and what it needs."""
    provider = entry.get("provider")
    if provider == "import":
        return False, "no built-in adapter: run it elsewhere and bring the outputs in with tamilbench_import_predictions"
    if provider == "tesseract":
        if not shutil.which("tesseract"):
            return False, "needs the tesseract binary with Tamil data (apt install tesseract-ocr tesseract-ocr-tam)"
        tessdata = str((entry.get("params") or {}).get("tessdata", ""))
        if "TESSDATA_BEST" in tessdata:
            from .models import ENV_DEFAULTS
            path = os.environ.get("TESSDATA_BEST") or ENV_DEFAULTS["TESSDATA_BEST"]
            if not Path(path).exists():
                return False, "needs tessdata_best: run `tamilbench fetch-tessdata`"
        return True, "local OCR engine"
    if provider in ("easyocr", "paddleocr"):
        ok = importlib.util.find_spec(provider) is not None
        return ok, "local OCR engine" if ok else f"needs `pip install {provider}`"
    if provider in ("compat", "openai-compatible"):
        return True, ("needs an OpenAI-compatible server (vLLM, SGLang, OpenRouter…): pass params "
                      "{'base_url': ...}; default http://localhost:8000/v1")
    missing = [" or ".join(group) for group in _CREDENTIALS.get(provider, [])
               if not any(os.environ.get(name) for name in group)]
    if missing:
        return False, "set " + " and ".join(missing) + " in the MCP server's environment"
    return True, "credentials found"


@mcp.tool(name="tamilbench_list_models", annotations=READ_ONLY, structured_output=False)
def tamilbench_list_models(
    status: Annotated[ModelStatus, Field(description="'evaluated' (has test-split scores), 'pending' or 'all'")]
    = ModelStatus.ALL,
    response_format: Fmt = ResponseFormat.MARKDOWN,
) -> str:
    """List the systems tracked by the leaderboard (models/registry.yaml): provider, API model
    id, whether test-split results exist, and whether it can run from this server now (API
    key present, engine installed). Key values are never shown."""
    from .models import load_registry
    out = []
    for e in load_registry():
        scored = (results_dir(results_root(), e["id"], "test") / "scores.json").exists()
        if status == ModelStatus.EVALUATED and not scored or status == ModelStatus.PENDING and scored:
            continue
        ok, need = availability(e)
        out.append({"id": e["id"], "name": e.get("name"), "org": e.get("org"), "provider": e.get("provider"),
                    "model": e.get("model"), "evaluated": scored, "runnable": ok, "requirement": need,
                    "verify_id": bool(e.get("verify_id"))})
    if response_format == ResponseFormat.JSON:
        return _dump({"count": len(out), "models": out})
    lines = [f"# Models ({len(out)})", "", "| id | name | provider | evaluated | runnable now | note |",
             "|---|---|---|---|---|---|"]
    lines += [f"| `{m['id']}` | {m['name']} | {m['provider']} | {'yes' if m['evaluated'] else 'no'} | "
              f"{'yes' if m['runnable'] else 'no'} | {m['requirement']}{' · API id unverified' if m['verify_id'] else ''} |"
              for m in out]
    lines += ["", "Any other system: tamilbench_run_model with model='provider:model-id' (e.g. "
                  "'compat:Qwen/Qwen3-VL-30B-A3B-Instruct'), or tamilbench_import_predictions."]
    return "\n".join(lines)


_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()


def _job_view(job: dict) -> dict:
    run_dir = Path(job["run_dir"])
    done = read_predictions(run_dir / "predictions.jsonl") if (run_dir / "predictions.jsonl").exists() else {}
    answered = sum(1 for i in job["item_ids"] if i in done and (done[i].get("text") is not None or done[i].get("refusal")))
    errors = sum(1 for i in job["item_ids"] if i in done and done[i].get("error"))
    view = {k: job[k] for k in ("job_id", "model", "split", "status", "started_at", "finished_at", "error")}
    view.update({"total": len(job["item_ids"]), "answered": answered, "errors": errors,
                 "elapsed_s": round((job.get("finished") or time.time()) - job["t0"], 1)})
    if job.get("summary"):
        view["summary"] = job["summary"]
    return view


def _run_job(job: dict, adapter, subsets: list[str] | None, limit: int | None, concurrency: int) -> None:
    from . import runner
    try:
        with redirect_stdout(sys.stderr):
            runner.run(adapter, data_dir(), split=job["split"], results_root=results_root(), subset_ids=subsets,
                       limit=limit, concurrency=concurrency, progress=False)
            scores = runner.score(Path(job["run_dir"]), data_dir(), split=job["split"])
        job["summary"] = {"overall": scores["overall"], "recognition_avg": scores["recognition_avg"],
                          "modern_script": (scores["eras"].get("modern") or {}).get("reading"),
                          "older_scripts": (scores["eras"].get("older") or {}).get("reading"),
                          "tracks": scores["tracks"], "complete": scores.get("complete")}
        job["status"] = "finished"
    except Exception as e:  # noqa: BLE001 - reported to the client through tamilbench_run_status
        job["status"], job["error"] = "failed", f"{type(e).__name__}: {e}"[:500]
    finally:
        job["finished"] = time.time()
        job["finished_at"] = _now()


@mcp.tool(name="tamilbench_run_model",
          annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True,
                                      openWorldHint=True), structured_output=False)
async def tamilbench_run_model(
    model: Annotated[str, Field(description="A registry id from tamilbench_list_models (e.g. 'claude-opus-5-5', "
                                            "'tesseract-tam-best') or 'provider:model' (e.g. "
                                            "'compat:Qwen/Qwen3-VL-30B-A3B-Instruct')", min_length=2, max_length=200)],
    split: SplitArg = Split.LITE,
    subsets: Annotated[list[str] | None, Field(description="Only these subset ids")] = None,
    limit_per_subset: Annotated[int | None, Field(description="At most this many items per subset (a quick check; "
                                                              "results are then not rankable)", ge=1, le=200)] = None,
    concurrency: Annotated[int, Field(description="Parallel requests", ge=1, le=16)] = 4,
    params: Annotated[dict[str, str | int | float | bool] | None,
                      Field(description="Adapter parameters, e.g. {'effort': 'high'} or {'base_url': "
                                        "'http://gpu:8000/v1'}. Never pass API keys here")] = None,
    wait: Annotated[bool, Field(description="Block until the run finishes (use only for small runs); otherwise "
                                            "start it in the background and poll tamilbench_run_status")] = False,
) -> str:
    """Evaluate a system through its adapter on the benchmark, exactly as the CLI does: same
    images, prompts and post-processing. Results go to results/<id>/v1-<split>/ and are
    resumable; the run is scored when it finishes. API keys come from the server's
    environment (see tamilbench_list_models)."""
    from .models import create, registry_entry
    if subsets:
        for sid in subsets:
            _spec(sid)
    if params and any(re.search(r"(key|token|secret|password)", k, re.I) for k in params):
        raise ToolError("Do not pass credentials as parameters. Set the API key in the MCP server's environment.")
    entry = registry_entry(model)
    if entry:
        ok, need = availability(entry)
        if not ok:
            raise ToolError(f"{model} cannot run here: {need}.")
    try:
        adapter = create(model, **(params or {}))
    except SystemExit as e:
        raise ToolError(str(e)) from None
    except TypeError as e:
        raise ToolError(f"Invalid adapter parameters for {model}: {e}") from None
    run_dir = results_dir(results_root(), adapter.slug, split.value)
    with _JOBS_LOCK:
        busy = [j for j in _JOBS.values() if j["run_dir"] == str(run_dir) and j["status"] == "running"]
        if busy:
            raise ToolError(f"{model} is already running on '{split.value}' (job {busy[0]['job_id']}). "
                            "Poll it with tamilbench_run_status.")
        rows = [r for r in _manifest(split.value) if not subsets or r["subset"] in subsets]
        if limit_per_subset:
            seen: dict[str, int] = {}
            kept = []
            for r in rows:
                if seen.get(r["subset"], 0) < limit_per_subset:
                    seen[r["subset"]] = seen.get(r["subset"], 0) + 1
                    kept.append(r)
            rows = kept
        rows = [r for r in rows if S.get(r["subset"]).task.value in adapter.supports]
        job = {"job_id": uuid.uuid4().hex[:8], "model": adapter.slug, "split": split.value, "status": "running",
               "run_dir": str(run_dir), "item_ids": [r["id"] for r in rows], "t0": time.time(),
               "started_at": _now(), "finished_at": None, "error": None}
        _JOBS[job["job_id"]] = job
    if wait:
        await anyio.to_thread.run_sync(lambda: _run_job(job, adapter, subsets, limit_per_subset, concurrency))
        return _dump(_job_view(job))
    threading.Thread(target=_run_job, args=(job, adapter, subsets, limit_per_subset, concurrency),
                     daemon=True).start()
    return (f"Started job {job['job_id']}: {adapter.slug} on {len(rows)} items of '{split.value}'. "
            f"Poll with tamilbench_run_status(job_id='{job['job_id']}'). Results: {run_dir}")


@mcp.tool(name="tamilbench_run_status", annotations=READ_ONLY, structured_output=False)
def tamilbench_run_status(
    job_id: Annotated[str | None, Field(description="A job id from tamilbench_run_model; omit to list all jobs "
                                                    "started by this server")] = None,
) -> str:
    """Report the progress of model runs started with tamilbench_run_model: items answered,
    errors, elapsed time and, when finished, the headline scores."""
    with _JOBS_LOCK:
        jobs = list(_JOBS.values())
    if job_id:
        jobs = [j for j in jobs if j["job_id"] == job_id]
        if not jobs:
            raise ToolError(f"No job '{job_id}' in this server session. Jobs do not survive a restart; runs "
                            "resume where they stopped when started again.")
    if not jobs:
        return "No runs started in this session. Start one with tamilbench_run_model."
    return _dump([_job_view(j) for j in jobs])


@mcp.tool(name="tamilbench_import_predictions", annotations=WRITES_LOCAL, structured_output=False)
async def tamilbench_import_predictions(
    run_name: RunName,
    file_path: Annotated[str, Field(description="Absolute path to a JSONL file with one {\"id\": ..., \"text\": ...} "
                                                "per line, produced by any system")],
    supports: Annotated[list[Task], Field(description="Tasks the system performs; unsupported tasks are shown as "
                                                      "'—', never zero", min_length=1)],
    split: SplitArg = Split.TEST,
    name: Annotated[str | None, Field(description="Display name for the system", max_length=100)] = None,
) -> str:
    """Bring in the outputs of a system that was run outside this harness and score them.
    Items without an answer count as empty for the tasks the system supports."""
    from . import runner
    path = Path(file_path).expanduser()
    if not path.is_file() or path.suffix.lower() != ".jsonl":
        raise ToolError(f"{file_path} is not a .jsonl file.")
    if path.stat().st_size > _MAX_IMPORT_BYTES:
        raise ToolError("The file is larger than 50 MB.")
    run_dir = _run_dir(run_name, split.value)
    meta = _run_meta(run_dir)
    if meta is not None and meta.get("model", {}).get("kind") not in ("imported",):
        raise ToolError(f"'{run_name}' already holds a {meta['model'].get('kind')} run. Choose another run_name.")
    ids = _by_id(split.value)
    recs = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            raise ToolError(f"Line {n} is not valid JSON.") from None
        if not isinstance(r, dict) or "id" not in r:
            raise ToolError(f"Line {n} has no 'id'.")
        if r["id"] not in ids:
            raise ToolError(f"Line {n}: unknown item id '{r['id']}' for split '{split.value}'.")
        recs.append(r)
    runner.import_predictions(path, model_id=run_name, name=name, supports=[t.value for t in supports],
                              split=split.value, results_root=results_root())
    s, md = await anyio.to_thread.run_sync(lambda: _score(run_name, split.value, False, 1000))
    return f"Imported {len(recs)} answers into '{run_name}'.\n\n" + md


# ------------------------------------------------------------------------------- leaderboard

@mcp.tool(name="tamilbench_get_leaderboard", annotations=READ_ONLY, structured_output=False)
def tamilbench_get_leaderboard(
    include_pending: Annotated[bool, Field(description="Also list systems awaiting evaluation")] = False,
    response_format: Fmt = ResponseFormat.MARKDOWN,
) -> str:
    """Show the current leaderboard: ranked systems with Overall, OCR/HTR, the modern-script
    and older-script halves and track scores, statistical ties (≈) between neighbouring
    ranks, and failure-mode rates."""
    path = _REPO / "leaderboard" / "data" / "leaderboard.json"
    if not path.exists():
        raise ToolError("No leaderboard yet. Build it with tamilbench_update_leaderboard.")
    lb = json.loads(path.read_text(encoding="utf-8"))
    models = [m for m in lb["models"] if include_pending or m["status"] == "evaluated"]
    if response_format == ResponseFormat.JSON:
        keys = ("id", "name", "org", "status", "rank_overall", "rank_recognition", "overall", "recognition_avg",
                "modern_script", "older_scripts", "tracks", "vs_next", "failure_modes")
        return _dump({"generated_at": lb["generated_at"], "models": [{k: m.get(k) for k in keys} for m in models],
                      "diagnostics": lb.get("diagnostics")})
    lines = [f"# Leaderboard (benchmark {lb['benchmark']['version']}, generated {lb['generated_at'][:10]})", "",
             "| # | System | Overall | OCR/HTR | Modern script | Older scripts | "
             + " | ".join(t["name"] for t in lb["tracks"]) + " |",
             "|---:|---|---:|---:|---:|---:|" + "---:|" * len(lb["tracks"])]
    for m in models:
        if m["status"] != "evaluated":
            continue
        rank = m.get("rank_overall") or f"({m.get('rank_recognition', '')})"
        vs = (m.get("vs_next") or {}).get("overall" if m.get("rank_overall") else "recognition_avg")
        if vs and not vs["separable"]:
            rank = f"{rank} ≈"
        lines.append(f"| {rank} | {m['name']} | {_fmt(m.get('overall'))} | {_fmt(m.get('recognition_avg'))} | "
                     f"{_fmt(m.get('modern_script'))} | {_fmt(m.get('older_scripts'))} | "
                     + " | ".join(_fmt((m.get('tracks') or {}).get(t['id'])) for t in lb["tracks"]) + " |")
    pending = [m["id"] for m in models if m["status"] != "evaluated"]
    if pending:
        lines += ["", "Awaiting evaluation: " + ", ".join(f"`{p}`" for p in pending)]
    lines += ["", "Ranks in parentheses: OCR/HTR rank for systems that only read. OCR/HTR = ½ modern script + ½ "
                  "older scripts. ≈: not separable from the next rank (paired bootstrap, 95 %). Per-script "
                  "scores: tamilbench_get_script_scores."]
    return "\n".join(lines)


@mcp.tool(name="tamilbench_get_script_scores", annotations=READ_ONLY, structured_output=False)
def tamilbench_get_script_scores(
    run_name: Annotated[str | None, Field(description="A scored run (results/ folder name); omit for every system "
                                                      "on the leaderboard", max_length=100)] = None,
    split: SplitArg = Split.TEST,
    response_format: Fmt = ResponseFormat.MARKDOWN,
) -> str:
    """Reading scores per script stage — modern Tamil, pre-reform Tamil, Grantha–Tamil, Grantha
    and Tamil-Brahmi — with 95 % intervals, the modern and older halves, the gap between
    them, each script's scores by track and its identification scores."""
    if run_name:
        run_dir = _run_dir(run_name, split.value)
        path = next((p for p in (run_dir / "scores.json", run_dir / "scores-answered.json") if p.exists()), None)
        if path is None:
            raise ToolError(f"Run '{run_name}' has not been scored on '{split.value}'. Call tamilbench_score_run.")
        s = json.loads(path.read_text(encoding="utf-8"))
        data = {"run": run_name, "split": split.value, "eras": s.get("eras"), "era_gap": s.get("era_gap"),
                "scripts": s.get("scripts")}
        if response_format == ResponseFormat.JSON:
            return _dump(data)
        lines = [f"# Scores by script: {run_name} · {split.value}", ""]
        for e, v in (s.get("eras") or {}).items():
            lines.append(f"- **{v['name']}:** reading {_fmt(v.get('reading'))} ({_ci_text(v.get('reading_ci95'))})"
                         + (f", composite {_fmt(v['composite'])}" if v.get("composite") is not None else ""))
        lines += [""] + _script_rows(s.get("scripts") or {})
        for sc, v in (s.get("scripts") or {}).items():
            cells = ", ".join(f"{TRACKS[Track(t)].name} {_fmt(x)}" for t, x in v["tracks"].items())
            ident = v.get("identification") or {}
            extra = "; ".join(f"{k.replace('_', ' ')} {x:.1f}" for k, x in ident.items())
            lines.append(f"- {SCRIPTS[Script(sc)].name}: {cells}" + (f"; {extra}" if extra else ""))
        return "\n".join(lines)
    path = _REPO / "leaderboard" / "data" / "leaderboard.json"
    if not path.exists():
        raise ToolError("No leaderboard yet. Build it with tamilbench_update_leaderboard.")
    lb = json.loads(path.read_text(encoding="utf-8"))
    ev = [m for m in lb["models"] if m["status"] == "evaluated"]
    order = [s.value for s in LINEAGE_ORDER if any(s.value in (m.get("scripts") or {}) for m in ev)]
    if response_format == ResponseFormat.JSON:
        return _dump({"scripts": order, "models": [{"id": m["id"], "name": m["name"], "modern_script":
                     m.get("modern_script"), "older_scripts": m.get("older_scripts"), "era_gap": m.get("era_gap"),
                     "scripts": m.get("scripts")} for m in ev]})
    lines = ["# Reading scores by script stage (test split)", "",
             "| System | Modern script | Older scripts | " + " | ".join(SCRIPTS[Script(x)].name for x in order) + " |",
             "|---|---:|---:|" + "---:|" * len(order)]
    for m in ev:
        sc = m.get("scripts") or {}
        lines.append(f"| {m['name']} | {_fmt(m.get('modern_script'))} | {_fmt(m.get('older_scripts'))} | "
                     + " | ".join(_fmt((sc.get(x) or {}).get("score")) for x in order) + " |")
    lines += ["", "Older scripts = mean of the older script stages, each weighted equally. OCR/HTR = ½ modern + ½ "
                  "older. Vatteluttu and medieval Tamil have no items yet."]
    return "\n".join(lines)


@mcp.tool(name="tamilbench_update_leaderboard", annotations=WRITES_LOCAL, structured_output=False)
async def tamilbench_update_leaderboard() -> str:
    """Rebuild leaderboard/data/leaderboard.json, the static site and the README table from
    every scored run in results/ (interactive runs are not ranked)."""
    from .leaderboard import build

    def go():
        with redirect_stdout(sys.stderr):
            return build(results_root(), data_dir(), _REPO / "leaderboard" / "data" / "leaderboard.json",
                         readme=_REPO / "README.md")
    lb = await anyio.to_thread.run_sync(go)
    n_eval = sum(m["status"] == "evaluated" for m in lb["models"])
    return (f"Leaderboard rebuilt: {n_eval} evaluated, {len(lb['models']) - n_eval} awaiting evaluation. "
            "See it with tamilbench_get_leaderboard.")


# ------------------------------------------------------------------------------- resources & prompts

@mcp.resource("tamilbench://methodology", name="methodology", title="Methodology and scoring rubric",
              mime_type="text/markdown")
def methodology() -> str:
    """docs/METHODOLOGY.md: what is measured, how the data is built, every prompt, the rubric."""
    return (_REPO / "docs" / "METHODOLOGY.md").read_text(encoding="utf-8")


@mcp.resource("tamilbench://dataset-card", name="dataset-card", title="Dataset card (subsets.json)",
              mime_type="application/json")
def dataset_card() -> str:
    """subsets.json: subsets, counts, seed and the contamination canary."""
    return (data_dir() / "subsets.json").read_text(encoding="utf-8")


@mcp.prompt(name="take_tamilbench", title="Take the Tamil OCR / HTR benchmark")
def take_tamilbench(run_name: str = "claude-mcp", split: str = "lite", subset_id: str = "") -> str:
    """Instructions for answering the benchmark item by item through this server."""
    scope = f" of subset {subset_id}" if subset_id else ""
    return (f"You are taking the Tamil OCR / HTR Benchmark as run '{run_name}' on split '{split}'{scope}.\n"
            f"Repeat until every item is answered: call tamilbench_next_item(run_name='{run_name}', "
            f"split='{split}'{f', subset_id={subset_id!r}' if subset_id else ''}); read the image; answer exactly "
            "as its prompt asks, from the image alone, with no commentary; then call tamilbench_submit_answer "
            "with that answer. Do not request reference answers and do not use other tools or sources. When "
            f"tamilbench_next_item says everything is answered, call tamilbench_score_run(run_name='{run_name}', "
            f"split='{split}') and report the scores, the failure modes and the weakest subsets.\n"
            f"Note: {CANARY}")


@mcp.prompt(name="evaluate_model", title="Evaluate a model on the benchmark")
def evaluate_model(model: str, split: str = "lite") -> str:
    """Instructions for running, scoring and comparing a model through this server."""
    return (f"Evaluate '{model}' on the Tamil OCR / HTR Benchmark (split '{split}').\n"
            "1. tamilbench_list_models: check that it is runnable here; if not, say exactly what is missing.\n"
            f"2. tamilbench_run_model(model='{model}', split='{split}'); poll tamilbench_run_status until it "
            "finishes.\n"
            f"3. tamilbench_score_run on the run; then tamilbench_compare_runs against the best evaluated system "
            "from tamilbench_get_leaderboard.\n"
            "4. Summarise the track scores, statistical ties, failure modes and the subsets where it is weakest.")


def main(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8000) -> None:
    if transport == "stdio":
        mcp.run("stdio")
    else:
        mcp.run("streamable-http", host=host, port=port)


if __name__ == "__main__":
    main()
