"""Local OCR engines: Tesseract (always available in CI), EasyOCR and PaddleOCR (optional
extras, each downloading its own Tamil weights on first use), plus debug adapters."""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile

from .base import RECOGNITION_ONLY, Adapter, Prediction

_PSM = {"character": 10, "word": 8, "line": 7}


class TesseractAdapter(Adapter):
    """``tesseract:<lang>`` — e.g. ``tam``, ``Tamil`` (script model), ``tam+eng``.
    Options: ``tessdata`` (directory, e.g. one holding tessdata_best models), ``psm``
    (``auto`` picks by granularity), ``oem``."""

    provider = "tesseract"
    kind = "ocr-engine"
    supports = RECOGNITION_ONLY

    def __init__(self, model: str = "tam", tessdata: str | None = None, psm: str = "auto", oem: int = 1, **params):
        super().__init__(model, tessdata=tessdata, psm=psm, oem=oem, **params)
        self.exe = shutil.which("tesseract")
        if not self.exe:
            raise RuntimeError("tesseract not found on PATH (apt install tesseract-ocr tesseract-ocr-tam)")
        if tessdata:
            for lang in model.split("+"):
                if not os.path.exists(os.path.join(str(tessdata), f"{lang}.traineddata")):
                    raise RuntimeError(f"{lang}.traineddata not found in {tessdata} — run `tamilbench fetch-tessdata`")

    def metadata(self) -> dict:
        md = super().metadata()
        try:
            md["engine_version"] = subprocess.run([self.exe, "--version"], capture_output=True, text=True,
                                                  timeout=30).stdout.splitlines()[0]
        except Exception:  # noqa: BLE001
            pass
        if self.params.get("tessdata"):
            md["params"]["tessdata"] = os.path.basename(str(self.params["tessdata"]).rstrip("/"))
        return md

    def _predict(self, image, mime, prompt, system, sample) -> Prediction:
        psm = self.params["psm"]
        if psm == "auto":
            psm = _PSM.get(sample.get("granularity", "block"), 6)
        suffix = ".png" if mime == "image/png" else ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix) as f:
            f.write(image)
            f.flush()
            cmd = [self.exe, f.name, "stdout", "-l", self.model, "--psm", str(psm), "--oem", str(self.params["oem"])]
            if self.params.get("tessdata"):
                cmd += ["--tessdata-dir", str(self.params["tessdata"])]
            env = {**os.environ, "OMP_THREAD_LIMIT": "1"}   # parallelism comes from the runner, not OpenMP
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip()[:300] or _exit_reason(r.returncode))
        return Prediction(r.stdout, raw={"psm": psm})


def _exit_reason(code: int) -> str:
    """Describe a non-zero exit; a negative code means the process was killed by a signal
    (Tesseract 5.3 can die with SIGFPE on some inputs, leaving stderr empty)."""
    if code < 0:
        import signal
        try:
            return f"tesseract killed by {signal.Signals(-code).name}"
        except ValueError:
            return f"tesseract killed by signal {-code}"
    return f"tesseract exited with code {code}"


class EasyOCRAdapter(Adapter):
    """``easyocr:ta`` (pip install easyocr; downloads weights on first run)."""

    provider = "easyocr"
    kind = "ocr-engine"
    supports = RECOGNITION_ONLY

    def __init__(self, model: str = "ta", gpu: bool = False, **params):
        super().__init__(model, gpu=gpu, **params)
        import easyocr
        langs = [x for x in model.split("+") if x]
        self.reader = easyocr.Reader(langs, gpu=gpu)

    def _predict(self, image, mime, prompt, system, sample) -> Prediction:
        lines = self.reader.readtext(image, detail=0, paragraph=sample.get("granularity") in ("block", "page"))
        return Prediction("\n".join(lines))


class PaddleOCRAdapter(Adapter):
    """``paddleocr:ta`` (pip install paddleocr paddlepaddle). Handles both the 2.x ``ocr()``
    and the 3.x ``predict()`` result shapes."""

    provider = "paddleocr"
    kind = "ocr-engine"
    supports = RECOGNITION_ONLY

    def __init__(self, model: str = "ta", **params):
        super().__init__(model, **params)
        from paddleocr import PaddleOCR
        self.engine = PaddleOCR(lang=model)

    def _predict(self, image, mime, prompt, system, sample) -> Prediction:
        import numpy as np
        from PIL import Image
        arr = np.asarray(Image.open(io.BytesIO(image)).convert("RGB"))[:, :, ::-1]
        if hasattr(self.engine, "predict"):
            res = self.engine.predict(arr)
            texts = []
            for page in res or []:
                page = page.json["res"] if hasattr(page, "json") else page
                texts += list(page.get("rec_texts", []))
            return Prediction("\n".join(texts))
        res = self.engine.ocr(arr, cls=True) or []
        texts = [ln[1][0] for page in res if page for ln in page]
        return Prediction("\n".join(texts))


class OracleAdapter(Adapter):
    """Returns the reference — only for testing the pipeline (never on the leaderboard)."""

    provider = "oracle"
    kind = "debug"

    def _predict(self, image, mime, prompt, system, sample) -> Prediction:
        target = sample.get("_target_field", "text")
        return Prediction(str(sample.get(target) or ""))


class BlankAdapter(Adapter):
    """Always answers with an empty string — the floor of every metric."""

    provider = "blank"
    kind = "debug"

    def _predict(self, image, mime, prompt, system, sample) -> Prediction:
        return Prediction("")
