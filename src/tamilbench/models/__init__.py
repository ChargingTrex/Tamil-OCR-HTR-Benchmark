"""Model adapters and the registry of tracked models.

``create("claude-opus-5-5")`` builds an adapter from ``models/registry.yaml``;
``create("compat:Qwen/Qwen3-VL-30B-A3B-Instruct?base_url=http://gpu:8000/v1")`` builds one
ad hoc (``provider:model`` plus optional ``?key=value&…`` adapter parameters).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import parse_qsl

import yaml

from .base import Adapter, Prediction  # noqa: F401

REGISTRY_PATH = Path(os.environ.get("TAMILBENCH_REGISTRY", Path(__file__).resolve().parents[3] / "models" / "registry.yaml"))


def _providers() -> dict:
    from .claude import ClaudeAdapter
    from .http_vlm import CompatAdapter, GeminiAdapter, OpenAIChatAdapter
    from .local import BlankAdapter, EasyOCRAdapter, OracleAdapter, PaddleOCRAdapter, TesseractAdapter
    from .ocr_services import AzureReadAdapter, GoogleVisionAdapter, MistralOCRAdapter
    return {
        "anthropic": ClaudeAdapter, "openai": OpenAIChatAdapter, "google": GeminiAdapter,
        "gemini": GeminiAdapter, "compat": CompatAdapter, "openai-compatible": CompatAdapter,
        "mistral-ocr": MistralOCRAdapter, "google-vision": GoogleVisionAdapter, "azure-di": AzureReadAdapter,
        "tesseract": TesseractAdapter, "easyocr": EasyOCRAdapter, "paddleocr": PaddleOCRAdapter,
        "oracle": OracleAdapter, "blank": BlankAdapter,
    }


_ENV = re.compile(r"\$\{([A-Z0-9_]+)\}")
ENV_DEFAULTS = {"TESSDATA_BEST": str(Path.home() / ".cache" / "tamilbench" / "tessdata_best")}


def _expand(v):
    if isinstance(v, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1)) or ENV_DEFAULTS.get(m.group(1), ""), v) or None
    return v


def load_registry(path: Path = REGISTRY_PATH) -> list[dict]:
    if not Path(path).exists():
        return []
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []


def registry_entry(model_id: str) -> dict | None:
    return next((e for e in load_registry() if e["id"] == model_id), None)


def _coerce(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"true": True, "false": False, "none": None, "null": None}.get(v.lower(), v)


def create(spec: str, **overrides) -> Adapter:
    providers = _providers()
    entry = registry_entry(spec)
    if entry:
        if entry["provider"] == "import":
            raise SystemExit(f"{spec} has no built-in adapter: run it yourself and use "
                             f"`tamilbench import-predictions --model {spec} …`")
        params = {k: _expand(v) for k, v in (entry.get("params") or {}).items()}
        params.update(overrides)
        params.setdefault("slug", entry["id"])
        return providers[entry["provider"]](entry["model"], **params)
    if ":" not in spec:
        raise SystemExit(f"unknown model {spec!r}: not in {REGISTRY_PATH.name} and not provider:model")
    provider, rest = spec.split(":", 1)
    model, _, query = rest.partition("?")
    params = {k: _coerce(v) for k, v in parse_qsl(query)}
    params.update(overrides)
    if provider not in providers:
        raise SystemExit(f"unknown provider {provider!r}; available: {', '.join(sorted(providers))}")
    return providers[provider](model, **params)
