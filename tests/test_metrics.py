"""The worked examples of docs/METHODOLOGY.md §5.9, computed with the scoring code."""

import pytest

from tamilbench.metrics import classification as cls
from tamilbench.metrics.recognition import TextPolicy, aggregate, sample_stats
from tamilbench.metrics.translation import corpus_chrf
from tamilbench.subsets import get

DEFAULT = TextPolicy()
EPIGRAPHIC = TextPolicy.from_dict(get("stone").policy)


def score(stats):
    return round(100 * max(0.0, 1 - stats["char_edits"] / stats["chars"]), 1)


def test_example_a_modern_print():
    st = sample_stats("தமிழ்நாடு அரசு புதிய கல்வித் திட்டத்தை இன்று அறிவித்தது.",
                      "தமிழ்நாடு அரக புதிய கல்வித் திட்டத்தை இன்று அறிவித்தது", DEFAULT)
    assert (st["char_edits"], st["chars"]) == (3, 56)
    assert (st["letter_edits"], st["letters"]) == (2, 36)
    assert (st["word_edits"], st["words"]) == (2, 7)
    assert score(st) == 94.6


def test_example_b_epigraphic_conventions():
    ref = "இத்தர்மம் இரக்ஷிப்பார் ஸ்ரீபாதம் என் தலைமேலன"
    hyp = "இததரமம இரக்ஷிப்பார் ஸ்ரீபாதம்என்தலைமெலன"
    assert sample_stats(ref, hyp, EPIGRAPHIC)["char_edits"] == 0
    st = sample_stats(ref, hyp, DEFAULT)
    assert (st["char_edits"], st["chars"]) == (6, 44)


def test_example_c_reading_beats_reciting():
    policy = TextPolicy.from_dict(get("palm-leaf-cict").policy)
    st = sample_stats("அகர முதல என் வெழுத்தெல்லா ஆதி பகவன் முத ற்றே வுல கு",
                      "அகர முதல எழுத்தெல்லாம் ஆதி பகவன் முதற்றே உலகு ௧", policy)
    assert (st["char_edits"], st["chars"]) == (6, 37)
    assert score(st) == 83.8


def test_example_d_invisible_differences():
    assert sample_stats("“வணக்கம்” — தமிழ்‌", '"வணக்கம்" - தமிழ்', DEFAULT)["char_edits"] == 0


def test_example_e_grantha():
    policy = TextPolicy.from_dict(get("grantha").policy)
    st = sample_stats("svasti śrī", "Svasti śrīḥ", policy)
    assert (st["char_edits"], st["chars"], score(st)) == (1, 9, 88.9)
    assert sample_stats("saṃskṛtam", "saṁskr̥tam", policy)["char_edits"] == 0


@pytest.mark.parametrize("hyp, edits", [
    ("Kalyāṇaguṇa ங்களை", 0),
    ("kalyanagunaங்களை", 3),
    ("kalyāṇaguṇaṅkaḷai", 6),
    ("கல்யாணகுணங்களை", 11),
])
def test_example_e_grantha_tamil(hyp, edits):
    policy = TextPolicy.from_dict(get("grantha-tamil").policy)
    st = sample_stats("kalyāṇaguṇaங்களை", hyp, policy)
    assert (st["char_edits"], st["chars"]) == (edits, 15)


def test_micro_average_and_floor():
    a = sample_stats("அஆ", "அஆ")
    b = sample_stats("இஈஉஊ", "")
    agg = aggregate([a, b])
    assert agg["cer"] == pytest.approx(4 / 6)
    assert aggregate([sample_stats("அ", "அஆஇஈ")])["score"] == 0.0   # never negative


@pytest.mark.parametrize("answer, label", [
    ("tamil-brahmi", "tamil-brahmi"),
    ("The script is Tamil Brahmi.", "tamil-brahmi"),
    ("Grantha (used for Sanskrit)", "grantha"),
    ("**grantha**", "grantha"),
    ("tamil", cls.INVALID),
    ("malayalam or kannada", cls.INVALID),
    ("", cls.INVALID),
])
def test_example_f_label_parsing(answer, label):
    from tamilbench.taxonomy import SCRIPT_ID_LABELS
    assert cls.parse_label(answer, SCRIPT_ID_LABELS.labels) == label


def test_macro_f1_penalises_collapsing_classes():
    labels = ["a", "b", "c"]
    gold = ["a", "b", "c"] * 4
    assert cls.scores(gold, gold, labels)["macro_f1"] == 1.0
    collapsed = cls.scores(gold, ["a"] * 12, labels)
    assert collapsed["accuracy"] == pytest.approx(1 / 3)
    assert collapsed["macro_f1"] < collapsed["accuracy"]


def test_example_g_chrf():
    ref = "The Tamil Nadu government announced a new education scheme today."
    hyp = "The government announced a new education plan today."
    assert corpus_chrf([hyp], [ref]) == pytest.approx(65.95, abs=0.01)


def test_chrf_matches_sacrebleu():
    sacrebleu = pytest.importorskip("sacrebleu")
    refs = ["The bus to Madurai leaves at six.", "Shop closed on Sunday", "Water is precious; save it!",
            "Welcome to Chennai Central", "Tamil is a classical language."]
    hyps = ["Bus to Madurai departs at 6.", "Shop is closed Sunday", "Save water, it is precious",
            "Welcome to Chennai", ""]
    ours = corpus_chrf(hyps, refs)
    theirs = sacrebleu.metrics.CHRF(word_order=2).corpus_score(hyps, [refs]).score
    assert ours == pytest.approx(theirs, abs=1e-6)
