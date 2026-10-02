# Tamil OCR · HTR Benchmark

**தமிழ் எழுத்துணரி அளவுகோல்** — a benchmark for reading Tamil from images, across the
whole history of the script and every surface it is written on.

It tests systems on **Tamil-Brahmi** cave and potsherd inscriptions, **Grantha**, mixed
**Grantha–Tamil** (maṇipravāḷam) manuscripts, **pre-1978** palm-leaf and letterpress
orthography, and the **modern reformed script**. The surfaces are screens,
born-digital text, scanned and photographed print, handwriting, signboards, **palm
leaves**, **stone**, estampages, **copper plates** and pottery. It scores four tasks:
transcription (OCR/HTR), script identification, medium identification, and reading
plus translation into English. Frontier multimodal models, open-weights VLMs, OCR
services and classical OCR engines are measured on one scale.

* **1,654 test images in 18 subsets**, grouped into 8 leaderboard tracks
* **Real data** — 26 palm leaves of a Tirukkural manuscript with expert diplomatic
  transcriptions from the **Central Institute of Classical Tamil (CICT)** (*CICT
  Tirukkural Ground Truth Corpus*, CC BY 4.0) — alongside 17 procedurally rendered subsets
* **Reading, not reciting** — nonce-word controls in every reading subset, a
  diplomatic (scribe's-spelling) reference for real leaves, and a prior-reliance
  diagnostic that compares real text with invented words
* **Reproducible** — deterministic seeded builds, SHA-256 checksums, a regenerable
  private split, and the full prompts and rubric in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)

## Leaderboard

Scores are 0–100, higher is better. Reading tracks score 100 × (1 − CER); identification
is macro-F1; translation is chrF++. *Overall* averages all 8 tracks and needs every task;
*OCR/HTR* averages the 6 reading tracks and ranks engines that only transcribe. The
interactive leaderboard, with per-subset heatmaps, examples and confidence intervals, is
in [`leaderboard/`](leaderboard/) (open `leaderboard/index.html`, or publish the folder
with GitHub Pages).

<!-- LEADERBOARD:START -->
| # | Model | Overall | OCR/HTR avg | Print & Digital | Screen Text | Handwriting (HTR) | Scene Text | Manuscripts & Old Print | Epigraphy | Script & Medium ID | Image → English |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| (1) | **Tesseract 5 · tam (tessdata_best)** <br><sub>Tesseract OCR</sub> | — | 53.4 | 83.8 | 88.0 | 81.4 | 16.9 | 36.8 | 13.5 | — | — |
| (2) | **Tesseract 5 · tam (Ubuntu tessdata)** <br><sub>Tesseract OCR</sub> | — | 52.6 | 83.3 | 88.4 | 81.2 | 18.1 | 32.7 | 11.8 | — | — |
| (3) | **Tesseract 5 · script/Tamil (tessdata_best)** <br><sub>Tesseract OCR</sub> | — | 50.5 | 82.6 | 88.3 | 80.8 | 12.8 | 35.6 | 2.8 | — | — |

**Awaiting evaluation** (adapters ready — add an API key or endpoint and run `tamilbench run --model <id>`): `claude-fable-5-1`, `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-4-5`, `gpt-6-astra`, `gpt-6-1-sol`, `gpt-6-sol`, `gpt-6-luna`, `gemini-3-8-flash`, `gemini-3-7-flash`, `gemini-3-1-pro`, `mistral-ocr-4`, `google-cloud-vision`, `azure-read`, `sarvam-vision-2-1`, `bodhan-indicocr`, `qwen3-vl-235b-a22b`, `qwen3-vl-30b-a3b`, `qwen3-5-397b-a17b`, `paddleocr-vl`, `deepseek-ocr`, `deepseek-ocr-2`, `glm-ocr`, `dots-ocr`, `olmocr-2`, `gemma-3-27b`, `easyocr-ta`, `paddleocr-ta`

<sub>Scores are 0–100 (higher is better): 100·(1−CER) for reading tasks, macro-F1 for identification, chrF++ for translation. Overall = mean of the 8 tracks; OCR/HTR avg = mean of the 6 reading tracks, so OCR engines that cannot classify or translate are ranked there (in parentheses). ≈ marks a rank not statistically separable from the next (paired cluster bootstrap, 95 %). Benchmark v1, generated 2026-10-02.</sub>
<!-- LEADERBOARD:END -->

## What is tested

| Track | Subsets |
|---|---|
| Print & Digital | born-digital text in a dozen typefaces · scanned print · phone-photographed print · Tamil numerals, fractions and signs |
| Screen Text | real Chromium screenshots: news, chat, settings, forms, tables, subtitles, notifications |
| Handwriting | handwriting-style faces with elastic distortion, slant and ruled paper (synthetic proxy) |
| Scene Text | shop boards, road and bus signs, LED destination boards, wall paintings, banners |
| Manuscripts & Old Print | pre-1978 letterpress · synthetic palm leaves · **real CICT palm leaves** · Grantha–Tamil maṇipravāḷam |
| Epigraphy | stone inscriptions and plaques · copper plates · Tamil-Brahmi (TB-I/II/III on rock, estampage and pottery) · Grantha |
| Script & Medium ID | 8 scripts (modern, pre-reform, Tamil-Brahmi, Grantha; Malayalam, Kannada, Telugu, Sinhala) · 9 writing media |
| Image → English | read Tamil from print, screens, signs and handwriting and translate it |

Gaps are listed rather than hidden: Vatteluttu and medieval (Chola/Pandya) letterforms
need real, licensed images with expert readings, and the synthetic media are proxies for
real ones. See the roadmap in the methodology.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"            # Python 3.10+
tamilbench validate                # check every image against its checksum
tamilbench list-subsets
tamilbench list-models
```

Run a model and score it (results go to `results/<model>/v1-test/`):

```bash
tamilbench fetch-tessdata          # once, for the tessdata_best Tesseract models
tamilbench run --model tesseract-tam-best
tamilbench run --model claude-opus-5-5 --split lite    # 371-item smoke test
tamilbench run --model claude-opus-5-5                 # full public test split
tamilbench leaderboard             # rebuild leaderboard/ and the table above
```

API models read their keys from the environment: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`GEMINI_API_KEY` (or `GOOGLE_API_KEY`), `MISTRAL_API_KEY`, `GOOGLE_VISION_API_KEY`,
`AZURE_DI_ENDPOINT` + `AZURE_DI_KEY`. Open-weights models are served through any
OpenAI-compatible endpoint (vLLM, SGLang, Together, OpenRouter):

```bash
tamilbench run --model "compat:Qwen/Qwen3-VL-30B-A3B-Instruct?base_url=http://gpu:8000/v1"
```

Systems without an adapter can be run anywhere and their answers imported:

```bash
tamilbench import-predictions --model my-ocr --file preds.jsonl --supports recognition
```

Runs are resumable, record every parameter that affects behaviour (model identifier,
reasoning effort or temperature, prompt version, manifest hash) and keep the raw
predictions, so anyone can re-score them.

Compare two systems on paired differences (a cluster bootstrap over source texts) and see
whether their gap is real:

```bash
tamilbench compare tesseract-tam-best tesseract-tam
```

Besides the ranked scores, every run reports diagnostics that a character error rate
hides: empty, overlong and looping answers, Markdown, letters in the wrong script, an
order-free error that separates reading-order mistakes from misread letters, the most
confused Tamil letters, and a **recitation index** that shows whether a system reads the
real Tirukkural leaves or writes the textbook version instead. The research behind these
choices is reviewed in [`docs/RELATED-WORK.md`](docs/RELATED-WORK.md).

## Repository layout

```
src/tamilbench/        package: taxonomy, text tools, renderers, builder, metrics, adapters, CLI
  corpus/data/         authored and public-domain text pools (Tamil, Sanskrit, maṇipravāḷam)
  render/              shaping-correct text rendering and medium simulators
  models/              model adapters; models/registry.yaml lists tracked models
data/v1/               the v1 test set: images, manifests, checksums
results/               predictions, run metadata and scores per model
leaderboard/           static leaderboard site (index.html, data/leaderboard.json)
docs/METHODOLOGY.md    methodology, prompts and scoring rubric in full
docs/real-data.md      real data in v1, wanted sources, how to contribute a dataset
docs/RELATED-WORK.md   review of benchmark, OCR/HTR and Tamil OCR research, and what it changed here
docs/DATASHEET.md      datasheet and BetterBench self-assessment
docs/annotation-guidelines.md  transcription conventions per script
tests/                 pytest suite: worked examples, determinism, data integrity, docs
.github/workflows/     ci (tests), evaluate (run a model with repository secrets), pages
assets/fonts/          vendored OFL fonts (licences in assets/fonts/licenses/)
```

## Rebuilding the data

```bash
git clone https://github.com/ChargingTrex/tamil-ocr   # for the CICT leaves
tamilbench build --out data/v1 --cict-root tamil-ocr/cict-htr
```

The build is deterministic: the same seed and software stack reproduce byte-identical
images. Maintainers can generate an unseen private split with the same distribution
(`tamilbench build --split private --seed <secret>`) to check suspicious results.

## Acknowledgements

The real palm-leaf subset is built from the **CICT Tirukkural Ground Truth Corpus** of the
**Central Institute of Classical Tamil (CICT)**, Chennai (data curator: Kannan Krishnan),
published under CC BY 4.0 with a DOI for every leaf
([CICT digital archive](https://www.digitalarchives.cict.in/)). The leaf images and the
expert diplomatic transcriptions are CICT's work. This benchmark crops the leaves, masks
the digitiser's caption and uses the transcriptions as references. Please cite the corpus
whenever you use or report results on `palm-leaf-cict`.

The standard Tirukkural text used by the recitation diagnostic comes from the
[`thirukkural`](https://pypi.org/project/thirukkural/) package by Vaasudevan Srinivasan (MIT).

## Licences

* Code: Apache-2.0.
* Synthetic images, manifests and authored text: CC BY 4.0.
* CICT Tirukkural leaves and transcriptions: © Central Institute of Classical Tamil
  (CICT), *CICT Tirukkural Ground Truth Corpus*, CC BY 4.0 (data curator: Kannan
  Krishnan); per-leaf DOIs are recorded in the manifest.
* Tirukkural edition text: public domain, via the `thirukkural` package (MIT).
* Fonts: SIL Open Font License 1.1 (see `assets/fonts/licenses/`).

## Citation

```bibtex
@misc{tamilbench2026,
  title  = {Tamil OCR / HTR Benchmark: reading Tamil from Tamil-Brahmi to the screen},
  year   = {2026},
  howpublished = {\url{https://github.com/ChargingTrex/Tamil-OCR-HTR-Benchmark}},
  note   = {Benchmark version v1}
}
```
