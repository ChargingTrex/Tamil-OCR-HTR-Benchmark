import json

import numpy as np
import pytest

from tamilbench import CANARY
from tamilbench import subsets as S
from tamilbench.compare import compare
from tamilbench.metrics import diagnostics as diag
from tamilbench.scoring import cluster_ids, score_recognition, score_run

from .conftest import DATA


@pytest.mark.parametrize("ch, script", [
    ("க", "TAMIL"), ("௧", "TAMIL"), ("a", "LATIN"), ("ā", "LATIN"), ("\U00011315", "GRANTHA"),
    ("क", "DEVANAGARI"), ("ക", "MALAYALAM"), ("\U00011013", "BRAHMI"), ("1", "Common"), (".", "Common"),
    ("̥", "Inherited"),
])
def test_script_of(ch, script):
    assert diag.script_of(ch) == script


def test_failure_mode_detectors():
    assert diag.offscript_letters("तमिल தமிழ்", {"TAMIL"})[0] == 4
    assert diag.has_markup("**தமிழ்**") and diag.has_markup("# தலைப்பு") and diag.has_markup("<b>அ</b>")
    assert diag.has_markup("| அ | ஆ |\n|---|---|")
    assert not diag.has_markup("தமிழ் நாடு - 2024")
    assert diag.has_repetition("அம்மா " * 6)
    assert not diag.has_repetition("தமிழ்நாடு அரசு புதிய கல்வித் திட்டத்தை அறிவித்தது")


@pytest.mark.parametrize("ref, hyp, edits", [("abc", "cab", 0), ("abc", "abd", 1), ("abc", "ab", 1), ("ab", "abcd", 2)])
def test_bag_edits_ignore_order(ref, hyp, edits):
    assert diag.bag_edits(ref, hyp) == edits


def test_order_gap_isolates_serialisation_errors():
    d = diag.item_diagnostics("முதல் வரி இரண்டாம் வரி", "இரண்டாம் வரி முதல் வரி", None, {"TAMIL"})
    assert d["bag_edits"] == 0 and d["nospace_edits"] > 0
    assert diag.aggregate_diagnostics([d])["order_gap"] > 0


def _cict_row(text, canonical, rid="x-1"):
    return {"id": rid, "subset": "palm-leaf-cict", "text": text, "text_canonical": canonical,
            "text_source": "cict:" + rid, "lexical": "corpus"}


def test_recitation_index_runs_from_reading_to_reciting():
    spec = S.get("palm-leaf-cict")
    rows = [_cict_row("அகர முதல என் வெழுத்தெல்லா ஆதி பகவன் முத ற்றே வுல கு",
                      "அகர முதல எழுத்தெல்லாம் ஆதி பகவன் முதற்றே உலகு")]
    reads = score_recognition(spec, rows, [rows[0]["text"]], 50, 0)
    recites = score_recognition(spec, rows, [rows[0]["text_canonical"]], 50, 0)
    assert reads["recitation_index"] == -1.0
    assert recites["recitation_index"] == 1.0
    assert reads["score"] == 100.0 and recites["score"] < 100.0


def test_clusters_follow_source_texts():
    rows = [{"id": "a", "text_source": "modern:m001"}, {"id": "b", "text_source": "modern:m001"},
            {"id": "c", "text_source": "nonce"}, {"id": "d", "text_source": "nonce"},
            {"id": "e", "text_source": "classical:k001,classical:k002"}]
    c = cluster_ids(rows)
    assert c[0] == c[1] and c[2] != c[3] and len(set(c)) == 4


def test_paired_comparison(lite_manifest):
    oracle = {r["id"]: r[S.get(r["subset"]).target] for r in lite_manifest}
    same = compare(lite_manifest, oracle, oracle, n_boot=100)
    assert same["overall"]["diff"] == 0 and same["overall"]["p"] == 1.0 and not same["overall"]["separable"]
    vs_blank = compare(lite_manifest, oracle, {}, n_boot=100)
    assert vs_blank["recognition_avg"]["diff"] == pytest.approx(100.0)
    assert vs_blank["recognition_avg"]["separable"]
    assert vs_blank["subsets"]["grantha-tamil"]["p"] == 0.0


def test_run_level_failure_modes(lite_manifest):
    rows = [r for r in lite_manifest if r["subset"] == "grantha"]
    preds = {r["id"]: "संस्कृतम् " * 3 for r in rows}           # Devanagari instead of IAST
    s = score_run(rows, preds, n_boot=20, keep_per_sample=False)
    assert s["subsets"]["grantha"]["diagnostics"]["offscript_letter_rate"] == 1.0
    assert s["failure_modes"]["offscript_item_rate"] == 1.0
    assert "cluster bootstrap" in s["ci_method"]


def test_canary_and_canonical_text_in_manifest(lite_manifest):
    assert all(r.get("canary") == CANARY for r in lite_manifest)
    assert json.loads((DATA / "subsets.json").read_text())["canary"] == CANARY
    cict = [r for r in lite_manifest if r["subset"] == "palm-leaf-cict"]
    assert cict and all(r.get("text_canonical") for r in cict)
    assert all(len(r["text_canonical"].split("\n")) == len(r["text"].split("\n")) for r in cict)


def test_bootstrap_weights_are_paired():
    from tamilbench.scoring import boot_weights
    rows = [{"id": str(i), "text_source": "nonce"} for i in range(10)]
    assert np.array_equal(boot_weights(rows, 30, 0), boot_weights(rows, 30, 0))
    assert boot_weights(rows, 30, 0).sum(axis=1).tolist() == [10.0] * 30
