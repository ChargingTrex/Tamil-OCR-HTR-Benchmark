"""Script-balanced scoring, the control subsets and the robustness and human-baseline tooling."""

import csv
import json
import random

import numpy as np
import pytest

from tamilbench import prompts
from tamilbench import robustness
from tamilbench import runner
from tamilbench import subsets as S
from tamilbench.compare import compare
from tamilbench.metrics import recognition as rec_m
from tamilbench.metrics.unit_tests import best_match, run_tests
from tamilbench.scoring import aggregate, score_run
from tamilbench.taxonomy import Era, Script, era_of
from tamilbench.text.perturb import IAST_CLASSES, TAMIL_CLASSES, perturb_iast, perturb_tamil

from .conftest import DATA


def _answers(manifest, good):
    """Perfect answers for items whose script satisfies ``good``, empty ones otherwise."""
    return {r["id"]: (r[S.presentation(r).target] if good(r) else "") for r in manifest}


def test_every_lineage_script_has_an_era_and_distractors_have_none():
    for s in Script:
        assert (era_of(s) is None) == (s in (Script.MALAYALAM, Script.KANNADA, Script.TELUGU, Script.SINHALA,
                                             Script.LATIN)), s
    assert era_of("screen") is None and era_of("tamil-modern") == Era.MODERN


def test_modern_and_older_halves_weigh_the_same(lite_manifest):
    modern_only = score_run(lite_manifest, _answers(lite_manifest, lambda r: era_of(r["script"]) == Era.MODERN),
                            supported_tasks={"recognition"}, n_boot=20, keep_per_sample=False)
    assert modern_only["eras"]["modern"]["reading"] == pytest.approx(100.0)
    assert modern_only["eras"]["older"]["reading"] == pytest.approx(0.0)
    assert modern_only["recognition_avg"] == pytest.approx(50.0)
    older_only = score_run(lite_manifest, _answers(lite_manifest, lambda r: era_of(r["script"]) == Era.OLDER),
                           supported_tasks={"recognition"}, n_boot=20, keep_per_sample=False)
    assert older_only["recognition_avg"] == pytest.approx(50.0)
    assert older_only["era_gap"]["diff"] == pytest.approx(-100.0) and older_only["era_gap"]["separable"]


def test_each_older_script_stage_counts_equally(lite_manifest):
    """Reading only Tamil-Brahmi perfectly is worth a quarter of the older half, however
    few Tamil-Brahmi items there are."""
    s = score_run(lite_manifest, _answers(lite_manifest, lambda r: r["script"] == "tamil-brahmi"),
                  supported_tasks={"recognition"}, n_boot=20, keep_per_sample=False)
    older = s["eras"]["older"]
    assert set(older["scripts"]) == {"tamil-brahmi", "grantha", "grantha-tamil", "tamil-pre-reform"}
    assert s["scripts"]["tamil-brahmi"]["score"] == pytest.approx(100.0)
    assert older["reading"] == pytest.approx(25.0)
    assert s["recognition_avg"] == pytest.approx(12.5)


def test_mixed_subsets_are_split_by_script(lite_manifest):
    rows = [r for r in lite_manifest if r["subset"] == "stone"]
    assert {r["script"] for r in rows} == {"tamil-modern", "tamil-pre-reform"}
    s = score_run(lite_manifest, _answers(lite_manifest, lambda r: r["script"] == "tamil-modern"),
                  supported_tasks={"recognition"}, n_boot=20, keep_per_sample=False)
    by = s["subsets"]["stone"]["by_script"]
    assert by["tamil-modern"]["score"] == pytest.approx(100.0) and by["tamil-pre-reform"]["score"] == pytest.approx(0.0)
    assert sum(v["n"] for v in by.values()) == len(rows)
    # the modern epigraphy cell averages the modern stone and copper slices
    assert s["scripts"]["tamil-modern"]["tracks"]["epigraphy"] == pytest.approx(100.0)
    assert s["scripts"]["tamil-pre-reform"]["tracks"]["epigraphy"] == pytest.approx(0.0)


def test_overall_is_the_mean_of_the_two_composites(lite_manifest):
    oracle = {r["id"]: r[S.presentation(r).target] for r in lite_manifest}
    no_translation = {**oracle, **{r["id"]: "" for r in lite_manifest if r["subset"] == "translate-en"}}
    s = score_run(lite_manifest, no_translation, n_boot=20, keep_per_sample=False)
    assert s["eras"]["modern"]["composite"] == pytest.approx(87.5)     # translation is ⅛ of the modern half
    assert s["eras"]["older"]["composite"] == pytest.approx(100.0)
    assert s["overall"] == pytest.approx(93.75)
    assert s["overall_ci95"][0] <= s["overall"] <= s["overall_ci95"][1]


def test_identification_is_split_by_era(lite_manifest):
    oracle = {r["id"]: r[S.presentation(r).target] for r in lite_manifest}
    # call every older-script image "tamil-modern": the modern class loses precision, the older classes recall
    preds = {**oracle, **{r["id"]: "tamil-modern" for r in lite_manifest if r["subset"] == "script-id"}}
    sid = score_run(lite_manifest, preds, n_boot=20, keep_per_sample=False)["subsets"]["script-id"]
    assert sid["by_era"]["older"]["score"] == pytest.approx(0.0)
    assert 0 < sid["by_era"]["modern"]["score"] < 100
    assert set(sid["by_era"]["older"]["classes"]) == {"tamil-pre-reform", "tamil-brahmi", "grantha", "grantha-tamil"}


def test_aggregate_works_on_bootstrap_replicates():
    layout = {"print-digital": ["tamil-modern"], "grantha": ["grantha"]}
    subs = {"print-digital": {"score": np.array([80.0, 90.0]), "by_script": {"tamil-modern": np.array([80.0, 90.0])},
                              "by_era": {}},
            "grantha": {"score": np.array([10.0, 30.0]), "by_script": {"grantha": np.array([10.0, 30.0])}, "by_era": {}}}
    agg = aggregate(subs, layout)
    assert np.allclose(agg["recognition_avg"], [45.0, 60.0])
    assert agg["overall"] is None           # identification and translation are missing


def test_compare_reports_era_and_script_differences(lite_manifest):
    a = _answers(lite_manifest, lambda r: True)
    b = _answers(lite_manifest, lambda r: era_of(r["script"]) == Era.MODERN)
    res = compare(lite_manifest, a, b, supported_a={"recognition"}, supported_b={"recognition"}, n_boot=50)
    assert res["eras"]["modern"]["reading"]["diff"] == pytest.approx(0.0)
    assert res["eras"]["older"]["reading"]["diff"] == pytest.approx(100.0)
    assert res["recognition_avg"]["diff"] == pytest.approx(50.0) and res["recognition_avg"]["separable"]
    assert res["scripts"]["grantha"]["separable"] and not res["scripts"]["tamil-modern"]["separable"]


def test_abstaining_counts_as_an_empty_answer():
    assert rec_m.is_abstention("[no text]") and rec_m.is_abstention(" No legible text. ")
    assert not rec_m.is_abstention("no text here: தமிழ்")
    assert rec_m.answer_text("[no text]") == "" and rec_m.answer_text(None) == ""
    for key, text in prompts.PROMPTS.items():
        assert (prompts.NO_TEXT in text) == key.startswith("recognition"), key


def test_controls_are_presented_like_the_items_they_imitate(lite_manifest):
    controls = [r for r in lite_manifest if r["subset"] in S.CONTROL_SUBSETS]
    assert controls and all(r.get("as_subset") in S.RANKED_SUBSETS for r in controls)
    for r in controls:
        assert S.presentation(r).id == r["as_subset"]
        assert S.presentation(r).prompt in prompts.PROMPTS
    eras = {era_of(r["script"]) for r in controls if r["subset"] == "blank-controls"}
    assert eras == {Era.MODERN, Era.OLDER}


def test_perturbed_items_measure_prior_pull(lite_manifest):
    rows = [r for r in lite_manifest if r["subset"] == "perturbed"]
    assert all(r["text_canonical"] and r["lexical"] == "perturbed" for r in rows)
    reads = {r["id"]: r[S.presentation(r).target] for r in rows}
    recites = {r["id"]: r["text_canonical"] for r in rows}
    sr = score_run(rows, reads, n_boot=20, keep_per_sample=False)["prior_reliance"]["prior_pull"]
    sc = score_run(rows, recites, n_boot=20, keep_per_sample=False)["prior_reliance"]["prior_pull"]
    assert sr["index"] == pytest.approx(-1.0) and sc["index"] == pytest.approx(1.0)
    assert set(sr["by_era"]) == {"modern", "older"}


def test_perturbation_stays_inside_letter_classes():
    rng = random.Random(7)
    text = "அகர முதல எழுத்தெல்லாம் ஆதி பகவன் முதற்றே உலகு"
    out = perturb_tamil(text, rng)
    assert out != text and len(out.split()) == len(text.split())
    for a, b in zip(text, out):
        if a != b:
            assert any(a in g and b in g for g in TAMIL_CLASSES)
    iast = "yadā yadā hi dharmasya glānirbhavati bhārata"
    out = perturb_iast(iast, random.Random(3))
    assert out != iast
    for a, b in zip(iast, out):
        if a != b:
            assert any(a in g and b in g for g in IAST_CLASSES)


def test_line_tests_check_presence_and_order():
    assert best_match("abcd", "xxabzdyy") == (1, 5)
    lines = ["தமிழ் நாடு", "சென்னை நகரம்", "மதுரை மாநகர்"]
    good = run_tests(lines, " ".join(lines))
    assert good["present"] == 3 and good["order"] == 2
    swapped = run_tests(lines, " ".join([lines[1], lines[0], lines[2]]))
    assert swapped["present"] == 3 and swapped["order"] == 1
    assert run_tests(lines, "")["present"] == 0


def test_prompt_variants_keep_the_conventions():
    assert set(prompts.PROMPT_VARIANTS) == set(prompts.PROMPTS)
    for key, variants in prompts.PROMPT_VARIANTS.items():
        assert len(variants) == prompts.N_VARIANTS and variants[0] == prompts.PROMPTS[key]
        assert len(set(variants)) == prompts.N_VARIANTS, key
        if key.startswith("recognition"):
            assert all(prompts.NO_TEXT in v for v in variants)
        if key in ("script-id", "medium-id"):
            labels = S.get(key).labels
            assert all(all(f"- {lab}:" in v for lab in labels) for v in variants)


def _fake_run(root, slug, split, rows, tag, variant, answers):
    d = runner.results_dir(root, slug, split, tag)
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"model": {"id": slug, "kind": "vlm-api", "supports": ["recognition"]},
                                            "tag": tag, "prompt_variant": variant, "split": split}))
    with open(d / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "subset": r["subset"], "text": answers(r)}, ensure_ascii=False) + "\n")


def test_robustness_summarises_prompt_and_repeat_spread(tmp_path, lite_manifest):
    rows = [r for r in lite_manifest if r["task"] == "recognition"]
    perfect = lambda r: r[S.presentation(r).target]  # noqa: E731
    half = lambda r: perfect(r) if era_of(r["script"]) == Era.MODERN else ""  # noqa: E731
    _fake_run(tmp_path, "m", "lite", rows, None, 0, perfect)
    _fake_run(tmp_path, "m", "lite", rows, "p1", 1, half)
    _fake_run(tmp_path, "m", "lite", rows, "r1", 0, perfect)
    out = robustness.summarize(tmp_path / "m", DATA, split="lite", n_boot=20)
    assert out["variants"] == [0, 1] and out["repeats"] == [0, 1]
    assert out["prompt_spread"]["recognition_avg"]["range"] == pytest.approx(50.0)
    assert out["repeat_spread"]["recognition_avg"]["range"] == pytest.approx(0.0)
    assert out["answers_changed_between_repeats"] == 0.0
    assert (tmp_path / "m" / "v1-lite-robustness.json").exists()
    assert robustness.tag_for(0, 0) is None and robustness.tag_for(2, 0) == "p2" and robustness.tag_for(0, 1) == "r1"


def test_human_readers_come_in_through_a_reading_sheet(tmp_path):
    sheet = tmp_path / "sheet.csv"
    n = runner.write_reading_sheet(DATA, sheet, split="lite", subset_ids=["grantha"])
    with open(sheet, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert n == len(rows) == 20 and not any(r["answer"] for r in rows)
    assert all(prompts.PROMPTS["recognition-grantha"] == r["instruction"] for r in rows)
    with open(sheet, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "subset", "image", "instruction", "answer"])
        for r in rows:
            w.writerow([r["id"], r["subset"], r["image"], r["instruction"], "svasti śrī"])
    out = runner.import_predictions(sheet, model_id="reader-1", name="Reader 1", supports=["recognition"],
                                    split="lite", results_root=tmp_path / "results", kind="human",
                                    reader={"expertise": "Grantha"})
    meta = json.loads((out / "run.json").read_text())
    assert meta["model"]["kind"] == "human" and meta["model"]["params"]["reader"]["expertise"] == "Grantha"
    assert len(runner.read_predictions(out / "predictions.jsonl")) == 20


def test_text_layer_removes_ink_without_moving_the_layout():
    from tamilbench.render import media as M
    lines = ["தமிழ் நாடு"]
    with_text, _ = M.born_digital(lines, np.random.default_rng(1))
    with M.text_layer(lambda m: np.zeros_like(m)):
        blank, _ = M.born_digital(lines, np.random.default_rng(1))
    assert blank.size == with_text.size
    assert np.unique(np.asarray(blank.convert("L"))).size == 1   # one flat colour: nothing to read
    assert np.unique(np.asarray(with_text.convert("L"))).size > 1
    again, _ = M.born_digital(lines, np.random.default_rng(1))   # the hook is gone afterwards
    assert again.tobytes() == with_text.tobytes()


def test_leaderboard_shows_both_halves_and_every_script():
    from pathlib import Path

    from tamilbench.leaderboard import markdown_table
    path = Path(__file__).resolve().parents[1] / "leaderboard" / "data" / "leaderboard.json"
    if not path.exists():
        pytest.skip("no leaderboard built")
    lb = json.loads(path.read_text(encoding="utf-8"))
    table = markdown_table(lb)
    assert "| Modern script | Older scripts |" in table and "Reading by script stage" in table
    assert set(lb["eras"]) == {"modern", "older"}
    assert lb["eras"]["older"]["scripts"][0] == "tamil-brahmi"
    for m in (m for m in lb["models"] if m["status"] == "evaluated"):
        assert m["recognition_avg"] == pytest.approx((m["modern_script"] + m["older_scripts"]) / 2, abs=0.011)
        older = [m["scripts"][s]["score"] for s in lb["eras"]["older"]["scripts"] if s in m["scripts"]]
        assert m["older_scripts"] == pytest.approx(sum(older) / len(older), abs=0.011)
    site = path.parents[1]
    assert (site / "scripts.html").exists() and (site / "data" / "scores-by-script.csv").exists()
    assert 'href="scripts.html"' in (site / "index.html").read_text(encoding="utf-8")
