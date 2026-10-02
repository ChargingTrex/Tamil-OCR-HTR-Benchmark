# Using the benchmark from Claude: the MCP server

`tamilbench mcp` serves the benchmark over the [Model Context Protocol](https://modelcontextprotocol.io),
so Claude (in Claude Code, Claude Desktop or any other MCP client) can:

* **explore** the subsets and items, and see each test image with the exact prompt a
  model receives;
* **take the benchmark itself**, item by item, with every answer scored by the official
  code;
* **test other models**: run any registered system through its adapter (Claude, GPT,
  Gemini, open-weights models behind an OpenAI-compatible server, OCR services and
  engines), import outputs produced elsewhere, score them, compare two runs with paired
  statistics, read failure-mode diagnostics, and rebuild the leaderboard.

## Install

```bash
pip install -e ".[mcp]"      # from the repository root; needs Python 3.10+
tamilbench validate          # the server reads data/v1 and writes results/
```

## Connect a client

**Claude Code.** The repository ships a project-scoped [`.mcp.json`](../.mcp.json): open
the repository in Claude Code and approve the `tamilbench` server when asked. To add it by
hand instead:

```bash
claude mcp add tamilbench -- tamilbench mcp
```

**Claude Desktop.** Add the server to `claude_desktop_config.json` (Settings → Developer →
Edit Config), with absolute paths:

```json
{
  "mcpServers": {
    "tamilbench": {
      "command": "/path/to/Tamil-OCR-HTR-Benchmark/.venv/bin/tamilbench",
      "args": ["mcp"],
      "env": {
        "TAMILBENCH_DATA": "/path/to/Tamil-OCR-HTR-Benchmark/data/v1",
        "TAMILBENCH_RESULTS": "/path/to/Tamil-OCR-HTR-Benchmark/results",
        "ANTHROPIC_API_KEY": "sk-ant-…"
      }
    }
  }
}
```

**Other clients and remote use.** Any MCP client can launch `tamilbench mcp` over stdio. For
an HTTP endpoint (several clients, or another machine on a private network):

```bash
tamilbench mcp --transport streamable-http --host 127.0.0.1 --port 8000   # → http://127.0.0.1:8000/mcp
```

To inspect the server interactively: `npx @modelcontextprotocol/inspector tamilbench mcp`.

### Environment variables

| Variable | Purpose |
|---|---|
| `TAMILBENCH_DATA` | The `data/v1` directory (default: the repository's) |
| `TAMILBENCH_RESULTS` | Where runs are written (default: the repository's `results/`) |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `MISTRAL_API_KEY`, `GOOGLE_VISION_API_KEY`, `AZURE_DI_ENDPOINT` + `AZURE_DI_KEY`, `OPENAI_COMPAT_API_KEY` | Credentials for the systems you want to run. They are read from the server's environment only |
| `TESSDATA_BEST` | Location of Tesseract's `tessdata_best` models (`tamilbench fetch-tessdata`) |

## Tools

| Tool | What it does | Writes? |
|---|---|---|
| `tamilbench_list_subsets` | The 18 subsets: track, task, size, provenance, scoring | — |
| `tamilbench_get_subset` | One subset in full: description, prompt verbatim, policy, labels, credit | — |
| `tamilbench_list_items` | Items of a subset, paged (references only on request) | — |
| `tamilbench_get_item` | An item's image and its exact system and task prompts | — |
| `tamilbench_next_item` | The next unanswered item of a run | — |
| `tamilbench_submit_answer` | Records an answer and returns its official per-item score | `results/<run>/` |
| `tamilbench_score_run` | Scores a run: subsets with 95 % intervals, tracks, averages, failure modes | `scores.json` |
| `tamilbench_get_diagnostics` | Failure modes, order-free error, letter confusions, recitation index | — |
| `tamilbench_compare_runs` | Paired comparison of two runs, with intervals, p-values and ties | — |
| `tamilbench_list_models` | Registered systems, whether they have results, and whether they can run here | — |
| `tamilbench_run_model` | Runs a system through its adapter, in the background or blocking | `results/<id>/` |
| `tamilbench_run_status` | Progress of background runs | — |
| `tamilbench_import_predictions` | Imports and scores a JSONL of outputs from any system | `results/<run>/` |
| `tamilbench_get_leaderboard` | Ranked systems, ties and track scores | — |
| `tamilbench_update_leaderboard` | Rebuilds `leaderboard.json`, the site and the README table | `leaderboard/`, `README.md` |

Two prompts start the common workflows: **`take_tamilbench`** (answer the benchmark item by
item) and **`evaluate_model`** (run, score and compare a model). Two resources carry the
reference material: `tamilbench://methodology` and `tamilbench://dataset-card`.

## Workflows

**Let Claude take the benchmark.** In Claude Code or Claude Desktop, use the
`take_tamilbench` prompt, or ask:

> Take the Tamil OCR benchmark on the lite split as run `claude-desktop-oct`, then score it
> and tell me where you were weakest.

Claude then loops: `tamilbench_next_item` → reads the image → `tamilbench_submit_answer` → …
→ `tamilbench_score_run`. Each answer is scored immediately with the official metric, and
the reference stays hidden unless asked for.

**Test another model.**

> Which models can run here? Evaluate qwen3-vl-30b-a3b on the lite split against my vLLM
> server at http://gpu:8000/v1, then compare it with tesseract-tam-best.

This becomes `tamilbench_list_models`, then `tamilbench_run_model(model="qwen3-vl-30b-a3b",
params={"base_url": "http://gpu:8000/v1"})`, `tamilbench_run_status` and
`tamilbench_compare_runs`. A system without an adapter is run anywhere and its JSONL
(`{"id": …, "text": …}` per line) imported with `tamilbench_import_predictions`.

**Read results.** `tamilbench_get_leaderboard`, `tamilbench_get_diagnostics(run_name=…,
subset_id="palm-leaf-cict")` and `tamilbench_compare_runs(run_a=…, run_b=…)`.

## Rules that keep results honest

* **Interactive runs are not ranked.** When Claude answers through a chat client, its
  system prompt and tools differ from the benchmark's fixed API protocol (methodology §4),
  so those runs are tagged `interactive` and left off the leaderboard. Ranked Claude
  results come from `tamilbench_run_model` (or `tamilbench run`) with the API adapter.
* **References stay hidden while a test is being taken.** `tamilbench_next_item` never
  shows them, and `tamilbench_submit_answer` reveals one only when asked
  (`show_reference=true`). Items can share source texts, so revealing references
  mid-run would inflate later answers.
* **Interactive answers cannot overwrite model runs.** A run folder that holds an adapter
  or imported run refuses interactive answers.
* **Credentials never travel through tool calls.** API keys come only from the server's
  environment; `tamilbench_run_model` rejects key-like parameters and
  `tamilbench_list_models` reports only whether a key is present.
* **Paths are constrained.** Run names are single folder names (letters, digits, `.`, `_`,
  `-`); imports accept only `.jsonl` files up to 50 MB with known item ids; the HTTP
  transport binds to `127.0.0.1` unless told otherwise.

## Notes

* Long runs: `tamilbench_run_model` starts in the background by default; poll
  `tamilbench_run_status`. Runs are resumable, so a restarted server continues where a run
  stopped when it is started again.
* The real palm leaves in `palm-leaf-cict` are the work of the **Central Institute of
  Classical Tamil (CICT)**, *CICT Tirukkural Ground Truth Corpus*, CC BY 4.0; the server
  shows this credit with every item from that subset.
