"""The MCP server, driven end to end through an in-process MCP client."""

import asyncio
import json

import pytest

pytest.importorskip("mcp")
from mcp import Client  # noqa: E402

from tamilbench import CANARY  # noqa: E402
from tamilbench import mcp_server as srv  # noqa: E402
from tamilbench import subsets as S  # noqa: E402
from tamilbench.prompts import PROMPTS  # noqa: E402

from .conftest import DATA  # noqa: E402

pytestmark = pytest.mark.skipif(not (DATA / "manifest-lite.jsonl").exists(), reason="data/v1 is not present")


def call(name, args=None):
    async def go():
        async with Client(srv.mcp) as client:
            return await client.call_tool(name, args or {})
    return asyncio.run(go())


def text(res) -> str:
    return "\n".join(c.text for c in res.content if c.type == "text")


@pytest.fixture(autouse=True)
def tmp_results(tmp_path, monkeypatch):
    monkeypatch.setenv("TAMILBENCH_RESULTS", str(tmp_path / "results"))
    srv._JOBS.clear()
    return tmp_path / "results"


def lite_ids(subset):
    return [r["id"] for r in srv._manifest("lite") if r["subset"] == subset]


def test_tools_prompts_and_resources_are_registered():
    async def go():
        async with Client(srv.mcp) as client:
            return await client.list_tools(), await client.list_prompts(), await client.list_resources()
    tools, prompts, resources = asyncio.run(go())
    names = {t.name for t in tools.tools}
    assert {"tamilbench_list_subsets", "tamilbench_get_item", "tamilbench_next_item", "tamilbench_submit_answer",
            "tamilbench_score_run", "tamilbench_run_model", "tamilbench_compare_runs",
            "tamilbench_get_leaderboard"} <= names
    assert all(n.startswith("tamilbench_") for n in names)
    assert {p.name for p in prompts.prompts} == {"take_tamilbench", "evaluate_model"}
    assert {str(r.uri) for r in resources.resources} == {"tamilbench://methodology", "tamilbench://dataset-card"}


def test_browsing_subsets_and_items():
    listing = json.loads(text(call("tamilbench_list_subsets", {"response_format": "json"})))
    assert listing["count"] == len(S.SUBSETS)
    cict = text(call("tamilbench_get_subset", {"subset_id": "palm-leaf-cict"}))
    assert "Central Institute of Classical Tamil" in cict and PROMPTS["recognition-palm-leaf-cict"] in cict
    page = json.loads(text(call("tamilbench_list_items", {"subset_id": "grantha", "limit": 5,
                                                         "response_format": "json"})))
    assert page["count"] == 5 and page["has_more"] and page["next_offset"] == 5
    assert all("reference" not in it for it in page["items"])
    with_ref = json.loads(text(call("tamilbench_list_items", {"subset_id": "grantha", "limit": 1,
                                                             "include_reference": True, "response_format": "json"})))
    assert with_ref["items"][0]["reference"]


def test_get_item_returns_image_and_exact_prompt():
    full = call("tamilbench_get_item", {"item_id": "grantha-tamil-0001", "split": "lite"})
    assert [c.type for c in full.content] == ["text", "image"]
    assert PROMPTS["recognition-grantha-tamil"] in full.content[0].text
    assert "Reference" not in full.content[0].text
    small = call("tamilbench_get_item", {"item_id": "grantha-tamil-0001", "split": "lite", "max_side": 256})
    assert len(small.content[1].data) < len(full.content[1].data)
    err = call("tamilbench_get_item", {"item_id": "nope-0001"})
    assert err.is_error and "tamilbench_list_items" in text(err)


def test_taking_the_benchmark_interactively(tmp_results):
    ids = lite_ids("print-digital")
    refs = {r["id"]: r["text"] for r in srv._manifest("lite") if r["subset"] == "print-digital"}
    for _ in range(3):
        nxt = call("tamilbench_next_item", {"run_name": "pytest-run", "split": "lite", "subset_id": "print-digital"})
        item_id = next(i for i in ids if f"`{i}`" in nxt.content[0].text)
        assert refs[item_id] not in nxt.content[0].text            # the answer is never shown
        res = text(call("tamilbench_submit_answer", {"run_name": "pytest-run", "item_id": item_id,
                                                     "answer": refs[item_id], "split": "lite"}))
        assert "CER 0.000" in res
    nxt = call("tamilbench_next_item", {"run_name": "pytest-run", "split": "lite", "subset_id": "print-digital"})
    assert "Progress: 3 of 20" in nxt.content[0].text
    scores = json.loads(text(call("tamilbench_score_run", {"run_name": "pytest-run", "split": "lite",
                                                           "response_format": "json"})))
    assert scores["answered_only"] and scores["n_answered"] == 3
    assert scores["subsets"]["print-digital"]["score"] == 100.0
    meta = json.loads((tmp_results / "pytest-run" / "v1-lite" / "run.json").read_text())
    assert meta["model"]["kind"] == "interactive"
    diag = text(call("tamilbench_get_diagnostics", {"run_name": "pytest-run", "split": "lite"}))
    assert "print-digital" in diag


def test_interactive_answers_cannot_touch_model_runs(tmp_results):
    run = tmp_results / "tesseract-tam-best" / "v1-lite"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"model": {"kind": "ocr-engine"}}))
    res = call("tamilbench_submit_answer", {"run_name": "tesseract-tam-best", "item_id": lite_ids("scene")[0],
                                            "answer": "x", "split": "lite"})
    assert res.is_error and "another run_name" in text(res)
    bad = call("tamilbench_submit_answer", {"run_name": "../escape", "item_id": lite_ids("scene")[0],
                                            "answer": "x", "split": "lite"})
    assert bad.is_error


def test_running_importing_and_comparing_models(tmp_path):
    done = json.loads(text(call("tamilbench_run_model", {"model": "oracle:x", "split": "lite",
                                                         "subsets": ["grantha"], "wait": True})))
    assert done["status"] == "finished" and done["answered"] == done["total"] == 20
    blank = tmp_path / "blank.jsonl"
    blank.write_text("\n".join(json.dumps({"id": i, "text": ""}) for i in lite_ids("grantha")) + "\n")
    imported = text(call("tamilbench_import_predictions", {"run_name": "blank-ocr", "file_path": str(blank),
                                                           "supports": ["recognition"], "split": "lite"}))
    assert "Imported 20 answers" in imported
    cmp = json.loads(text(call("tamilbench_compare_runs", {"run_a": "oracle__x", "run_b": "blank-ocr",
                                                           "split": "lite", "response_format": "json"})))
    assert cmp["subsets"]["grantha"]["diff"] == 100.0 and cmp["subsets"]["grantha"]["separable"]
    status = json.loads(text(call("tamilbench_run_status")))
    assert status[0]["status"] == "finished"


def test_credentials_are_never_taken_as_arguments(monkeypatch):
    res = call("tamilbench_run_model", {"model": "oracle:x", "params": {"api_key": "sk-test"}})
    assert res.is_error and "environment" in text(res)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    models = json.loads(text(call("tamilbench_list_models", {"response_format": "json"})))["models"]
    claude = next(m for m in models if m["provider"] == "anthropic")
    assert not claude["runnable"] and "ANTHROPIC_API_KEY" in claude["requirement"]
    blocked = call("tamilbench_run_model", {"model": claude["id"], "split": "lite", "limit_per_subset": 1})
    assert blocked.is_error and "ANTHROPIC_API_KEY" in text(blocked)


def test_prompt_and_leaderboard():
    async def go():
        async with Client(srv.mcp) as client:
            return await client.get_prompt("take_tamilbench", {"run_name": "my-run"})
    prompt = asyncio.run(go())
    body = prompt.messages[0].content.text
    assert "tamilbench_next_item" in body and "my-run" in body and CANARY in body
    lb = call("tamilbench_get_leaderboard")
    assert not lb.is_error and "Tesseract" in text(lb)


def test_docs_cover_every_tool():
    async def go():
        async with Client(srv.mcp) as client:
            return await client.list_tools()
    doc = (DATA.parents[1] / "docs" / "MCP.md").read_text(encoding="utf-8")
    missing = [t.name for t in asyncio.run(go()).tools if f"`{t.name}`" not in doc]
    assert not missing, missing


def test_script_scores_and_control_items():
    md = text(call("tamilbench_get_script_scores"))
    assert "Older scripts" in md and "Tamil-Brahmi" in md
    data = json.loads(text(call("tamilbench_get_script_scores", {"response_format": "json"})))
    assert data["scripts"][0] == "tamil-brahmi" and data["models"][0]["era_gap"]["separable"]
    blank = next(r for r in srv._manifest("lite") if r["subset"] == "blank-controls")
    item = call("tamilbench_get_item", {"item_id": blank["id"], "split": "lite"})
    assert PROMPTS[S.get(blank["as_subset"]).prompt] in item.content[0].text
    assert "Blank & effaced" not in item.content[0].text           # presented like the item it imitates
    res = text(call("tamilbench_submit_answer", {"run_name": "pytest-blank", "item_id": blank["id"],
                                                 "answer": "[no text]", "split": "lite"}))
    assert "no text claimed: correct" in res
