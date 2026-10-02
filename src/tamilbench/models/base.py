"""Common interface for everything that can be evaluated: VLM APIs, OCR APIs, local OCR
engines and open-weights models served behind an OpenAI-compatible endpoint."""

from __future__ import annotations

import base64
import random
import re
import time
from dataclasses import asdict, dataclass, field

ALL_TASKS = frozenset({"recognition", "script-id", "medium-id", "translation"})
RECOGNITION_ONLY = frozenset({"recognition"})


@dataclass
class Prediction:
    text: str | None                      # None = no answer (error / refusal)
    latency_s: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None
    refusal: bool = False
    raw: dict = field(default_factory=dict)   # small provider-specific extras (stop reason, model echo…)

    def to_json(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v not in (None, {}, False) or k == "text"}


class TransientError(Exception):
    """Retryable failure (rate limit, 5xx, network)."""


class Adapter:
    """Subclass and implement :meth:`_predict`."""

    provider: str = "base"
    kind: str = "vlm-api"            # vlm-api | ocr-api | ocr-engine | vlm-open | debug
    supports: frozenset[str] = ALL_TASKS

    def __init__(self, model: str, **params):
        self.model = model
        self.params = params

    # -- identity -------------------------------------------------------------------------
    @property
    def id(self) -> str:
        return f"{self.provider}:{self.model}"

    @property
    def slug(self) -> str:
        s = self.params.get("slug") or f"{self.provider}__{self.model}"
        return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

    def metadata(self) -> dict:
        public = {k: v for k, v in self.params.items() if "key" not in k.lower() and k != "slug"}
        return {"id": self.id, "provider": self.provider, "model": self.model, "kind": self.kind,
                "supports": sorted(self.supports), "params": public}

    # -- prediction -----------------------------------------------------------------------
    def predict(self, image: bytes, mime: str, prompt: str, system: str, sample: dict,
                max_retries: int = 5) -> Prediction:
        t0 = time.time()
        delay = 2.0
        for attempt in range(max_retries + 1):
            try:
                pred = self._predict(image, mime, prompt, system, sample)
                pred.latency_s = round(time.time() - t0, 3)
                if pred.text is not None:
                    pred.text = clean_output(pred.text)
                return pred
            except TransientError as e:
                if attempt == max_retries:
                    return Prediction(None, round(time.time() - t0, 3), error=f"transient: {e}")
                time.sleep(delay + random.uniform(0, 1))
                delay = min(delay * 2, 60)
            except Exception as e:  # noqa: BLE001 - recorded, never crashes a run
                return Prediction(None, round(time.time() - t0, 3), error=f"{type(e).__name__}: {e}"[:500])
        raise AssertionError("unreachable")

    def _predict(self, image: bytes, mime: str, prompt: str, system: str, sample: dict) -> Prediction:
        raise NotImplementedError

    def close(self) -> None:
        pass


_QUOTES = {'"': '"', "'": "'", "“": "”", "‘": "’", "«": "»"}
_FENCE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?```\s*$", re.S)


def clean_output(text: str) -> str:
    """The only post-processing applied to model output, identical for every model:
    strip surrounding whitespace, a single enclosing Markdown code fence, and one pair of
    enclosing quotes."""
    t = text.strip()
    m = _FENCE.match(t)
    if m:
        t = m.group(1).strip()
    if len(t) >= 2 and _QUOTES.get(t[0]) == t[-1]:
        t = t[1:-1].strip()
    return t


def b64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode("ascii")


def http_status_error(status: int, body: str) -> Exception:
    if status == 429 or status >= 500:
        return TransientError(f"HTTP {status}: {body[:300]}")
    return RuntimeError(f"HTTP {status}: {body[:500]}")
