"""End-to-end checks on the published data: integrity, oracle/blank scoring, determinism."""

import hashlib
import re

import pytest

from tamilbench import build as B
from tamilbench import subsets as S
from tamilbench.scoring import score_run
from tamilbench.text import tamil
from tamilbench.text.grantha import iast_to_grantha

from .conftest import DATA


def test_checksums_of_a_sample(lite_manifest):
    for row in lite_manifest[::12]:
        data = (DATA / row["image"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["sha256"], row["id"]


def test_every_item_has_a_reference(lite_manifest):
    for row in lite_manifest:
        spec = S.get(row["subset"])
        assert row.get(spec.target), row["id"]
        if spec.labels:
            assert row[spec.target] in spec.labels


def _oracle(manifest):
    return {r["id"]: r[S.get(r["subset"]).target] for r in manifest}


def test_oracle_scores_100_and_blank_scores_0(lite_manifest):
    best = score_run(lite_manifest, _oracle(lite_manifest), n_boot=20, keep_per_sample=False)
    worst = score_run(lite_manifest, {r["id"]: "" for r in lite_manifest}, n_boot=20, keep_per_sample=False)
    for sid, sub in best["subsets"].items():
        assert sub["score"] == pytest.approx(100.0), sid
        assert worst["subsets"][sid]["score"] == pytest.approx(0.0), sid
    assert best["overall"] == pytest.approx(100.0)
    assert worst["overall"] == pytest.approx(0.0)


def test_missing_predictions_count_as_empty(lite_manifest):
    preds = _oracle(lite_manifest)
    first = next(r["id"] for r in lite_manifest if r["subset"] == "print-digital")
    del preds[first]
    s = score_run(lite_manifest, preds, n_boot=20, keep_per_sample=False)
    assert s["subsets"]["print-digital"]["score"] < 100.0


def test_unsupported_tasks_are_not_zero(lite_manifest):
    s = score_run(lite_manifest, _oracle(lite_manifest), supported_tasks={"recognition"}, n_boot=20,
                  keep_per_sample=False)
    assert s["overall"] is None
    assert s["recognition_avg"] == pytest.approx(100.0)


@pytest.mark.parametrize("subset", ["print-digital", "handwriting", "palm-leaf-synth", "tamil-brahmi",
                                    "grantha", "grantha-tamil"])
def test_generation_is_deterministic(subset):
    a = B.GENERATORS[subset](B.sample_rng(B.DEFAULT_SEED, "test", subset, 3))
    b = B.GENERATORS[subset](B.sample_rng(B.DEFAULT_SEED, "test", subset, 3))
    assert a.text == b.text
    assert a.image.tobytes() == b.image.tobytes()


def test_grantha_tamil_reference_matches_rendered_text(lite_manifest):
    """Re-deriving the Grantha from each reference's IAST must give the text that was drawn."""
    rows = [r for r in lite_manifest if r["subset"] == "grantha-tamil"]
    assert rows
    for r in rows:
        derived = []
        for word in r["text"].split():
            derived.append("".join(iast_to_grantha(run) if run[0].isascii() or not ("஀" <= run[0] <= "௿")
                                    else run for run in re.findall(r"[஀-௿]+|[^஀-௿]+", word)))
        drawn = re.sub(r"\s", "", r["text_native"]).replace(tamil.VIRAMA, "")
        assert re.sub(r"\s", "", "".join(derived)).replace(tamil.VIRAMA, "") == drawn, r["id"]
