import pytest

from tamilbench import corpus
from tamilbench.text import brahmi, indic, manipravalam as mp, tamil
from tamilbench.text.grantha import iast_to_grantha, normalize_iast
from tamilbench.taxonomy import Script


def test_normalize_ignores_invisible_and_typographic_differences():
    assert tamil.normalize("“வணக்கம்” — தமிழ்‌") == tamil.normalize('"வணக்கம்" - தமிழ்')
    assert tamil.normalize("ஶ்ரீ") == "ஸ்ரீ"
    assert tamil.normalize("கொ") == "கொ"           # two-part vowel sign composes
    assert tamil.normalize(" a\n  b ") == "a b"
    assert tamil.normalize("a b\nc", spaces="remove") == "abc"
    assert tamil.normalize("குறள் ௧௨", strip_tamil_numerals=True).strip() == "குறள்"


def test_letters_are_tamil_aksharas():
    assert tamil.letters("கொக்ஸ்") == ["கொ", "க்", "ஸ்"]


def test_palm_leaf_spelling_helpers():
    assert tamil.drop_pulli("தர்மம்") == "தரமம"
    assert tamil.merge_long_e_o("தேவோ") == "தெவொ"
    assert tamil.contains_reform_syllable("பணம் கொண்டாடு") is False
    assert tamil.contains_reform_syllable("தலை") is True


def test_tamil_brahmi_tb3():
    assert brahmi.to_brahmi("தமிழ்", brahmi.Orthography.TB3) == "\U00011022\U0001102B\U0001103A\U00011035\U00011070"


def test_grantha_from_iast():
    assert iast_to_grantha("svasti śrī") == "𑌸𑍍𑌵𑌸𑍍𑌤𑌿 𑌶𑍍𑌰𑍀"
    assert iast_to_grantha("hariḥ oṃ").endswith("\U00011350")      # GRANTHA OM
    assert iast_to_grantha("jagat").endswith("\U0001134D")          # final consonant takes the virama
    assert iast_to_grantha("oṃkāra")[0] != "\U00011350"             # oṃ inside a word is spelled out


def test_iast_folding():
    # whitespace is left to the subset's TextPolicy; only case, ISO 15919 and daṇḍas fold here
    assert normalize_iast("Saṁskr̥tam ||").strip() == "saṃskṛtam"
    assert normalize_iast("namaḥ ।").strip() == "namaḥ"


def test_distractor_scripts_convert_without_tamil_left_over():
    for script in indic.SUPPORTED_SCRIPTS:
        out = indic.convert("தமிழ் மொழி இனிமையானது", script)
        assert not any("஀" <= ch <= "௿" for ch in out), script
    assert Script.MALAYALAM in indic.SUPPORTED_SCRIPTS


def test_manipravalam_markup():
    words = mp.parse("{kalyāṇaguṇa}ங்களை அருளிச்செய்தார் {hariḥ oṃ}")
    assert words[0] == [("gr", "kalyāṇaguṇa"), ("ta", "ங்களை")]
    assert [mp.reference(w) for w in words] == ["kalyāṇaguṇaங்களை", "அருளிச்செய்தார்", "hariḥ", "oṃ"]
    assert mp.native(words[0]) == iast_to_grantha("kalyāṇaguṇa") + "ங்களை"
    assert mp.native(words[3]) == "\U00011350"


@pytest.mark.parametrize("bad", ["{abc", "abc}", "{அ}", "abc", "{a}{"])
def test_manipravalam_markup_errors(bad):
    with pytest.raises(ValueError):
        mp.parse(bad)


def test_corpora_load():
    pools = corpus.all_pools()
    assert all(len(items) > 0 for items in pools.values())
    assert len(corpus.manipravalam()) == 48
    assert sum(mp.has_grantha(mp.parse(it.meta["markup"])) for it in corpus.manipravalam()) >= 45
    lex = set(corpus.lexicon())
    assert len(lex) > 1000 and not any(any("a" <= c <= "z" for c in w) for w in lex)


def test_nonce_lines_avoid_real_words():
    import random
    rng = random.Random(0)
    lex = set(corpus.lexicon())
    for _ in range(50):
        assert not any(w in lex for w in corpus.nonce_line(rng).split())
    for _ in range(50):
        assert mp.parse(mp.nonce_markup(rng))
