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
    for name in ("real-data.md", "annotation-guidelines.md"):
        assert (DOCS / name).exists()
    assert "<!-- LEADERBOARD:START -->" in (ROOT / "README.md").read_text(encoding="utf-8")


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
