import re

from tamilbench import prompts
from tamilbench import subsets as S
from tamilbench.models import create, load_registry
from tamilbench.models.base import clean_output

from .conftest import DOCS, ROOT


def test_every_prompt_is_quoted_verbatim_in_the_methodology():
    doc = (DOCS / "METHODOLOGY.md").read_text(encoding="utf-8")
    for name, text in [*prompts.PROMPTS.items(), ("SYSTEM", prompts.SYSTEM)]:
        assert text.strip() in doc, f"prompt {name} differs from docs/METHODOLOGY.md"


def test_every_subset_is_documented_with_its_count():
    doc = (DOCS / "METHODOLOGY.md").read_text(encoding="utf-8")
    for spec in S.SUBSETS:
        assert re.search(rf"\| `{re.escape(spec.id)}` \| [a-z-]+ \| {spec.count} \|", doc), spec.id
        assert spec.prompt in prompts.PROMPTS


def test_linked_docs_exist():
    for name in ("real-data.md", "annotation-guidelines.md", "RELATED-WORK.md", "DATASHEET.md"):
        assert (DOCS / name).exists()
    assert "<!-- LEADERBOARD:START -->" in (ROOT / "README.md").read_text(encoding="utf-8")


def test_cict_is_credited_wherever_its_data_appears():
    credit = "Central Institute of Classical Tamil"
    for path in (ROOT / "README.md", ROOT / "DATA_LICENSE.md", ROOT / "CITATION.cff", DOCS / "METHODOLOGY.md",
                 DOCS / "DATASHEET.md", ROOT / "leaderboard" / "_page.src.html"):
        assert credit in path.read_text(encoding="utf-8"), path.name
    assert credit in S.get("palm-leaf-cict").attribution


def test_clean_output_only_strips_wrappers():
    assert clean_output("  ```text\nதமிழ்\n```  ") == "தமிழ்"
    assert clean_output("“தமிழ்”") == "தமிழ்"
    assert clean_output("'grantha'") == "grantha"
    assert clean_output("Here is the text: தமிழ்") == "Here is the text: தமிழ்"


def test_registry_is_well_formed():
    reg = load_registry()
    ids = [e["id"] for e in reg]
    assert len(ids) == len(set(ids))
    for e in reg:
        assert {"id", "name", "provider", "model"} <= set(e), e.get("id")


def test_debug_adapters_round_trip():
    oracle = create("oracle:x")
    pred = oracle.predict(b"", "image/png", "p", "s", {"text": "தமிழ்", "_target_field": "text"})
    assert pred.text == "தமிழ்"
    assert create("blank:x").predict(b"", "image/png", "p", "s", {}).text == ""


def test_engine_crashes_are_named():
    from tamilbench.models.local import _exit_reason
    assert _exit_reason(-8) == "tesseract killed by SIGFPE"
    assert _exit_reason(1) == "tesseract exited with code 1"


def test_every_prompt_paraphrase_is_in_the_appendix():
    doc = (DOCS / "METHODOLOGY.md").read_text(encoding="utf-8")
    appendix = doc[doc.index("## Appendix A: prompt paraphrases"):]
    for key, variants in prompts.PROMPT_VARIANTS.items():
        for text in variants[1:]:
            assert text.strip() in appendix, f"a paraphrase of {key} differs from Appendix A"
