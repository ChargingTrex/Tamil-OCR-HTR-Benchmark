# Contributing

## Adding a model result

1. Run the full public test split: `tamilbench run --model <id>` (add an entry to
   `models/registry.yaml` first if the model is new; mark unverified API identifiers
   with `verify_id: true`).
2. Check the reproducibility list in `docs/METHODOLOGY.md` §6.4: `tamilbench validate`
   passes, `run.json` records every adapter parameter, failed items were retried, and
   refusals are reported.
3. Rebuild the leaderboard with `tamilbench leaderboard` and open a pull request with
   `results/<id>/v1-test/` (predictions, `run.json`, `scores.json`), `leaderboard/` and
   `README.md`.

Results that change prompts, add few-shot examples, use tools, or were tuned on the
test images are not ranked (§6.3). Systems without an adapter can be run anywhere and
imported with `tamilbench import-predictions`.

The **evaluate** workflow (`.github/workflows/evaluate.yml`) runs a model from GitHub
Actions with the repository's secrets and commits the results to the branch it runs on.

## Adding an adapter

Subclass `models.base.Adapter`, implement `_predict`, register the provider in
`models/__init__.py`, and add a test with a mocked client. Output post-processing is
fixed (`clean_output`) and must not be specialised per model.

## Adding data

Real data is welcome. See `docs/real-data.md` for admission rules and the item format,
and `docs/annotation-guidelines.md` for transcription conventions. New subsets and
changed references create a new benchmark version.

## Development

```bash
pip install -e ".[dev]"
pytest -q
tamilbench validate
```

Rebuilding the data needs the build extra and Chromium for the screenshot subsets:
`pip install -e ".[build]"`, then `tamilbench build` (see the README).
