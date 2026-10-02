"""Vision-language models reached over plain HTTPS: OpenAI, Google Gemini, and any
OpenAI-compatible endpoint (vLLM, SGLang, LMDeploy, Ollama, OpenRouter, Together, …),
which is how open-weights models such as Qwen3-VL, PaddleOCR-VL, DeepSeek-OCR, dots.ocr,
olmOCR, GLM-OCR or Gemma are evaluated."""

from __future__ import annotations

import os

import httpx

from .base import Adapter, Prediction, TransientError, b64, http_status_error


def _env(*names: str) -> str | None:
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    return None


class _HTTPAdapter(Adapter):
    def __init__(self, model: str, timeout: float = 600.0, transport: httpx.BaseTransport | None = None, **params):
        super().__init__(model, **params)
        self.http = httpx.Client(timeout=timeout, transport=transport)

    def _post(self, url: str, *, headers: dict, json: dict) -> dict:
        try:
            r = self.http.post(url, headers=headers, json=json)
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            raise TransientError(str(e)) from e
        if r.status_code >= 300:
            raise http_status_error(r.status_code, r.text)
        return r.json()

    def close(self) -> None:
        self.http.close()


class OpenAIChatAdapter(_HTTPAdapter):
    """OpenAI Chat Completions with an image_url data URI."""

    provider = "openai"
    kind = "vlm-api"
    default_base = "https://api.openai.com/v1"
    key_env = ("OPENAI_API_KEY",)

    def __init__(self, model: str, base_url: str | None = None, api_key_env: str | None = None,
                 max_tokens: int = 16000, reasoning_effort: str | None = None, detail: str = "high", **params):
        super().__init__(model, max_tokens=max_tokens, reasoning_effort=reasoning_effort, detail=detail,
                         base_url=base_url or self.default_base, **params)
        self.base_url = (base_url or self.default_base).rstrip("/")
        self.api_key = _env(api_key_env) if api_key_env else _env(*self.key_env)

    def _body(self, image, mime, prompt, system) -> dict:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64(image)}",
                                                        "detail": self.params.get("detail", "high")}},
                ]},
            ],
        }
        body[self._max_tokens_field()] = int(self.params["max_tokens"])
        if self.params.get("reasoning_effort"):
            body["reasoning_effort"] = self.params["reasoning_effort"]
        if "temperature" in self.params:
            body["temperature"] = float(self.params["temperature"])
        return body

    def _max_tokens_field(self) -> str:
        return "max_completion_tokens"

    def _predict(self, image, mime, prompt, system, sample) -> Prediction:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = self._post(f"{self.base_url}/chat/completions", headers=headers,
                          json=self._body(image, mime, prompt, system))
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage = data.get("usage") or {}
        raw = {"finish_reason": choice.get("finish_reason"), "served_by": data.get("model")}
        if msg.get("refusal"):
            return Prediction(None, refusal=True, raw={**raw, "refusal": str(msg["refusal"])[:300]})
        content = msg.get("content")
        if isinstance(content, list):   # some servers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return Prediction(content or "", input_tokens=usage.get("prompt_tokens"),
                          output_tokens=usage.get("completion_tokens"), raw=raw)


class CompatAdapter(OpenAIChatAdapter):
    """Any OpenAI-compatible server. Example: ``compat:Qwen/Qwen3-VL-30B-A3B-Instruct?base_url=http://localhost:8000/v1``."""

    provider = "compat"
    kind = "vlm-open"
    default_base = "http://localhost:8000/v1"
    key_env = ("OPENAI_COMPAT_API_KEY", "OPENROUTER_API_KEY", "TOGETHER_API_KEY")

    def __init__(self, model: str, temperature: float = 0.0, **params):
        super().__init__(model, temperature=temperature, **params)

    def _max_tokens_field(self) -> str:
        return "max_tokens"


class GeminiAdapter(_HTTPAdapter):
    """Google Gemini API (generativelanguage.googleapis.com, v1beta generateContent)."""

    provider = "google"
    kind = "vlm-api"
    base_url = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, model: str, max_tokens: int = 16000, generation_config: dict | None = None, **params):
        super().__init__(model, max_tokens=max_tokens, generation_config=generation_config or {}, **params)
        self.api_key = _env("GEMINI_API_KEY", "GOOGLE_API_KEY")

    def _predict(self, image, mime, prompt, system, sample) -> Prediction:
        gen = {"maxOutputTokens": int(self.params["max_tokens"]), **self.params.get("generation_config", {})}
        if "temperature" in self.params:
            gen["temperature"] = float(self.params["temperature"])
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [
                {"inlineData": {"mimeType": mime, "data": b64(image)}},
                {"text": prompt},
            ]}],
            "generationConfig": gen,
        }
        data = self._post(f"{self.base_url}/models/{self.model}:generateContent",
                          headers={"x-goog-api-key": self.api_key or ""}, json=body)
        usage = data.get("usageMetadata") or {}
        fb = data.get("promptFeedback") or {}
        cands = data.get("candidates") or []
        if fb.get("blockReason") or not cands:
            return Prediction(None, refusal=True, raw={"block_reason": fb.get("blockReason")})
        cand = cands[0]
        finish = cand.get("finishReason")
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        raw = {"finish_reason": finish, "served_by": data.get("modelVersion")}
        if finish in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "RECITATION") and not text:
            return Prediction(None, refusal=True, raw=raw)
        out = (usage.get("candidatesTokenCount") or 0) + (usage.get("thoughtsTokenCount") or 0)
        return Prediction(text, input_tokens=usage.get("promptTokenCount"), output_tokens=out or None, raw=raw)
