"""Anthropic Claude models via the official ``anthropic`` Python SDK.

Notes that matter for a benchmark:
  * Sampling parameters (temperature/top_p) are rejected by the newest Claude models, so
    none are sent unless explicitly configured.
  * Thinking is adaptive and always on for the newest models; depth is controlled with
    ``output_config.effort``, which is set explicitly and recorded with every result.
  * Server-side refusal fallbacks are deliberately NOT enabled: a fallback would mean a
    different model produced the answer. Refusals are recorded and counted instead.
"""

from __future__ import annotations

from .base import Adapter, Prediction, TransientError, b64

# Claude models that accept ``output_config.effort`` (Haiku 4.5 and Sonnet 4.5 reject it).
_NO_EFFORT = ("claude-haiku", "claude-sonnet-4-5", "claude-3")


class ClaudeAdapter(Adapter):
    provider = "anthropic"
    kind = "vlm-api"

    def __init__(self, model: str, effort: str | None = "high", max_tokens: int = 16000,
                 timeout: float = 600.0, client=None, **params):
        super().__init__(model, effort=effort, max_tokens=max_tokens, **params)
        import anthropic  # optional dependency: pip install anthropic
        self._sdk = anthropic
        self.client = client or anthropic.Anthropic(timeout=timeout, max_retries=2)

    def _request(self, image: bytes, mime: str, prompt: str, system: str) -> dict:
        req = {
            "model": self.model,
            "max_tokens": int(self.params["max_tokens"]),
            "system": system,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64(image)}},
                {"type": "text", "text": prompt},
            ]}],
        }
        effort = self.params.get("effort")
        if effort and not self.model.startswith(_NO_EFFORT):
            req["output_config"] = {"effort": effort}
        if "temperature" in self.params:   # only for models that still accept it
            req["temperature"] = float(self.params["temperature"])
        return req

    def _predict(self, image, mime, prompt, system, sample) -> Prediction:
        sdk = self._sdk
        try:
            resp = self.client.messages.create(**self._request(image, mime, prompt, system))
        except (sdk.RateLimitError, sdk.APIConnectionError, sdk.APITimeoutError) as e:
            raise TransientError(str(e)) from e
        except sdk.APIStatusError as e:
            if e.status_code >= 500 or e.status_code == 529:
                raise TransientError(f"{e.status_code}: {e.message}") from e
            raise
        usage = getattr(resp, "usage", None)
        raw = {"stop_reason": resp.stop_reason, "served_by": getattr(resp, "model", None)}
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            raw["refusal_category"] = getattr(details, "category", None)
            return Prediction(None, refusal=True, raw=raw,
                              input_tokens=getattr(usage, "input_tokens", None),
                              output_tokens=getattr(usage, "output_tokens", None))
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return Prediction(text, input_tokens=getattr(usage, "input_tokens", None),
                          output_tokens=getattr(usage, "output_tokens", None), raw=raw)
