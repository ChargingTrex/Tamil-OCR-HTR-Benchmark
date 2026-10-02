from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "v1"
DOCS = ROOT / "docs"


@pytest.fixture(scope="session")
def lite_manifest():
    from tamilbench.runner import load_manifest
    path = DATA / "manifest-lite.jsonl"
    if not path.exists():
        pytest.skip("data/v1 is not present")
    return load_manifest(path)
