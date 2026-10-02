"""Dedicated OCR services: Mistral OCR, Google Cloud Vision, Azure AI Document Intelligence.
These only transcribe, so they take part in the recognition subsets only."""

from __future__ import annotations

import os
import re
import time

from .base import RECOGNITION_ONLY, Prediction, TransientError, b64, http_status_error
from .http_vlm import _env, _HTTPAdapter

_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{2,}.*$", re.M)


def markdown_to_text(md: str) -> str:
    t = _MD_IMAGE.sub("", md)
    t = _MD_TABLE_RULE.sub("", t)
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)
    t = re.sub(r"(\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\1", r"\2", t)
    t = t.replace("|", " ")
    t = re.sub(r"\$\s*([^$]*?)\s*\$", r"\1", t)
    return re.sub(r"[ \t]+", " ", t).strip()


class MistralOCRAdapter(_HTTPAdapter):
    provider = "mistral-ocr"
    kind = "ocr-api"
    supports = RECOGNITION_ONLY
    url = "https://api.mistral.ai/v1/ocr"

    def __init__(self, model: str = "mistral-ocr-latest", **params):
        super().__init__(model, **params)
        self.api_key = _env("MISTRAL_API_KEY")

    def _predict(self, image, mime, prompt, system, sample) -> Prediction:
        body = {"model": self.model,
                "document": {"type": "image_url", "image_url": f"data:{mime};base64,{b64(image)}"}}
        data = self._post(self.url, headers={"Authorization": f"Bearer {self.api_key}"}, json=body)
        pages = data.get("pages") or []
        md = "\n".join(p.get("markdown", "") for p in pages)
        return Prediction(markdown_to_text(md), raw={"pages": len(pages)})


class GoogleVisionAdapter(_HTTPAdapter):
    """Cloud Vision DOCUMENT_TEXT_DETECTION with a Tamil language hint."""

    provider = "google-vision"
    kind = "ocr-api"
    supports = RECOGNITION_ONLY
    url = "https://vision.googleapis.com/v1/images:annotate"

    def __init__(self, model: str = "document-text-detection", language_hints=("ta",), **params):
        super().__init__(model, language_hints=list(language_hints), **params)
        self.api_key = _env("GOOGLE_VISION_API_KEY")
        self.token = _env("GOOGLE_VISION_ACCESS_TOKEN")

    def _predict(self, image, mime, prompt, system, sample) -> Prediction:
        feature = "DOCUMENT_TEXT_DETECTION" if self.model.startswith("document") else "TEXT_DETECTION"
        body = {"requests": [{"image": {"content": b64(image)}, "features": [{"type": feature}],
                              "imageContext": {"languageHints": self.params["language_hints"]}}]}
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        url = self.url + (f"?key={self.api_key}" if self.api_key and not self.token else "")
        data = self._post(url, headers=headers, json=body)
        res = (data.get("responses") or [{}])[0]
        if res.get("error"):
            raise RuntimeError(str(res["error"])[:300])
        return Prediction((res.get("fullTextAnnotation") or {}).get("text", ""))


class AzureReadAdapter(_HTTPAdapter):
    """Azure AI Document Intelligence ``prebuilt-read`` (async analyze + poll)."""

    provider = "azure-di"
    kind = "ocr-api"
    supports = RECOGNITION_ONLY

    def __init__(self, model: str = "prebuilt-read", api_version: str = "2024-11-30", **params):
        super().__init__(model, api_version=api_version, **params)
        self.endpoint = (os.environ.get("AZURE_DI_ENDPOINT") or "").rstrip("/")
        self.api_key = _env("AZURE_DI_KEY")

    def _predict(self, image, mime, prompt, system, sample) -> Prediction:
        url = (f"{self.endpoint}/documentintelligence/documentModels/{self.model}:analyze"
               f"?api-version={self.params['api_version']}")
        headers = {"Ocp-Apim-Subscription-Key": self.api_key or ""}
        r = self.http.post(url, headers=headers, json={"base64Source": b64(image)})
        if r.status_code != 202:
            raise http_status_error(r.status_code, r.text)
        op = r.headers["Operation-Location"]
        for _ in range(120):
            time.sleep(1.0)
            g = self.http.get(op, headers=headers)
            if g.status_code >= 300:
                raise http_status_error(g.status_code, g.text)
            data = g.json()
            status = data.get("status")
            if status == "succeeded":
                return Prediction((data.get("analyzeResult") or {}).get("content", ""))
            if status == "failed":
                raise RuntimeError(str(data.get("error"))[:300])
        raise TransientError("analyze timed out")
