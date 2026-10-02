# Related work, and what it changes in this benchmark

A review of research on (1) how benchmarks should be built and reported, (2) OCR and
handwriting benchmarks for multimodal models and OCR engines, (3) how vision-language
readers fail, (4) how OCR output should be measured, and (5) Tamil and other Indian-language
OCR/HTR. For each line of work it records what the paper found, what it implies for this
benchmark, and what we did about it.

*Searched October 2026 with a web search engine. Publisher sites, arXiv and Wikipedia were
not reachable from the build environment, so details come from abstracts, project pages
and proceedings listings; numbers are quoted only where a source states them.*

**Status key:** ✅ done (in this repository) · 🟡 partly done · 🔜 planned for the next
benchmark version · 📦 data candidate (licence to confirm before use).

---

## Summary: what changed

| Finding | Sources | Change in tamilbench | Status |
|---|---|---|---|
| Benchmark items drawn from the same source are correlated; naive intervals are too narrow. Models should be compared on paired, item-level differences. | Miller 2024 | Cluster bootstrap over source texts; `tamilbench compare` (paired differences, intervals, p-values); ties marked ≈ on the leaderboard | ✅ |
| Low CER can hide hallucination: normalisation, invented content, repetition loops, Markdown, text in the wrong script. | "When Low CER is Not Enough" 2026; Karamolegkou et al. 2026; CC-OCR; Devanagari stress test 2026 | Failure-mode diagnostics per subset and per run: empty, overlong, repetition, markup, wrong-script letters | ✅ |
| Most model errors on hard handwriting are prior-driven: fluent text that the image does not support. | WildHandBench 2026; Karamolegkou et al. 2026 | **Recitation index** on the real Tirukkural leaves: does the answer move from the scribe's text towards the standard edition? | ✅ |
| CER depends on reading order; page-level evaluation needs an order-free measure to separate recognition from serialisation errors. | Clausner et al. 2020; Bourne et al. 2026; OmniDocBench | Bag-of-characters error, order gap, sample-level NED | ✅ |
| Synthetic renders overstate quality; English OCR results do not predict Indic results. | Devanagari stress test 2026; GlotOCR Bench 2026; Thai synthetic-data study 2026 | Proxy-validity check (synthetic vs real palm leaves); real data is the top priority in the roadmap | 🟡 |
| Benchmarks should state a contamination policy; canary strings let data be filtered from training sets. | BIG-bench; contamination survey 2025 | Canary GUID in every manifest row and in the dataset card; private split already exists | ✅ |
| Leaderboards are distorted by private testing of many variants and selective disclosure. | Singh et al. 2025 | Admission rule: report every configuration evaluated for submission; never withdraw low results | ✅ |
| Document the dataset and assess the benchmark against a quality checklist. | Gebru et al. 2021; Reuel et al. 2024 | `docs/DATASHEET.md` with a BetterBench self-assessment | ✅ |
| Calibrated human baselines show how far models are from expert readers. | WildHandBench 2026 | Expert readers on the `lite` split | 🔜 |
| One prompt per task is fragile. | Mizrahi et al. 2024 | Prompt-paraphrase robustness on `lite` | 🔜 |
| Some subsets may not separate models; their noise should be measured. | Heineman et al. 2025 | Signal-to-noise per subset once enough systems are scored | 🔜 |
| Binary "unit tests" are a robust alternative to edit distance for long documents. | olmOCR-Bench | Presence/order tests for screens and long blocks | 🔜 |

---

## 1. How benchmarks should be built and reported

* **BetterBench** — Reuel-Lamparth, Hardy, Smith, Lamparth, Hardy, Kochenderfer. *Assessing
  AI Benchmarks, Uncovering Issues, and Establishing Best Practices.* NeurIPS 2024 Datasets
  & Benchmarks ([paper](https://arxiv.org/pdf/2411.12990), [site](https://betterbench.stanford.edu/)).
  46 criteria across design, implementation, documentation and maintenance, applied to 24
  benchmarks, with large quality differences found. *For us:* we now keep a self-assessment
  against its four stages in [`DATASHEET.md`](DATASHEET.md); the weakest stage is design,
  because few domain experts (epigraphists, palaeographers) have reviewed the items yet.
* **Datasheets for Datasets** — Gebru et al., *Communications of the ACM* 64(12), 2021
  ([ACM](https://dl.acm.org/doi/10.1145/3458723)). Motivation, composition, collection,
  uses and limits for every dataset. *For us:* [`DATASHEET.md`](DATASHEET.md).
* **Adding Error Bars to Evals** — Evan Miller, 2024 ([arXiv:2411.00640](https://arxiv.org/pdf/2411.00640)).
  Report standard errors; use *clustered* standard errors when questions come in related
  groups; compare models on question-level *paired* differences; use power analysis.
  *For us:* the biggest single methodological change. Images rendered from the same
  sentence or couplet are clustered; intervals and model comparisons use a paired cluster
  bootstrap (`scoring.py`, `compare.py`).
* **Lessons from the Trenches on Reproducible Evaluation of Language Models** — Biderman,
  Schoelkopf, Sutawika et al., 2024 ([arXiv:2405.14782](https://arxiv.org/pdf/2405.14782)).
  Evaluation results are sensitive to setup; publish prompts, settings and outputs.
  *For us:* already in `v1` (prompts quoted verbatim and tested, `run.json`, predictions
  committed).
* **State of What Art? A Call for Multi-Prompt LLM Evaluation** — Mizrahi, Kaplan, Malkin,
  Dror, Shahaf, Stanovsky, *TACL* 12, 2024 ([ACL Anthology](https://aclanthology.org/2024.tacl-1.52/)).
  Single-template benchmarks are brittle; evaluate across instruction paraphrases.
  *For us:* 🔜 prompt-robustness diagnostic on `lite`.
* **Signal and Noise** — Heineman, Hofmann, Magnusson, Gu, Smith, Hajishirzi, Lo, Dodge,
  NeurIPS 2025 ([arXiv:2508.13144](https://arxiv.org/abs/2508.13144)). A benchmark's
  ability to separate models (signal) against its random variability (noise) predicts how
  reliable decisions based on it are. *For us:* 🔜 per-subset signal-to-noise once enough
  systems are scored; small subsets (26 real leaves) are the likely weak spots.
* **The Leaderboard Illusion** — Singh et al., NeurIPS 2025 ([arXiv:2504.20879](https://arxiv.org/pdf/2504.20879)).
  Undisclosed private testing of many variants and selective score disclosure biased
  Chatbot Arena rankings. *For us:* ✅ admission rule in the methodology (§6.3).
* **Contamination.** The BIG-bench canary GUID convention
  ([BIG-bench](https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/training_on_test_set/README.md));
  *Recent Advances in LLM Benchmarks against Data Contamination: From Static to Dynamic
  Evaluation* ([arXiv:2502.17521](https://arxiv.org/pdf/2502.17521)); CapBencher
  ([arXiv:2505.18102](https://arxiv.org/pdf/2505.18102)). Canaries can be stripped, so
  they complement rather than replace held-out data. *For us:* ✅ canary in every manifest
  row; the seed-driven private split remains the real defence.

## 2. OCR and document benchmarks for multimodal models

* **OCRBench** — Liu et al., *Science China Information Sciences* 67, 2024
  ([paper](https://link.springer.com/article/10.1007/s11432-024-4235-6), [code](https://github.com/yuliang-liu/multimodalocr)).
  29 datasets; large multimodal models were weak on multilingual, handwritten and
  *non-semantic* text. *For us:* the nonce-word controls in every reading subset test
  non-semantic text directly.
* **OCRBench v2** — 2025 ([arXiv:2501.00321](https://arxiv.org/abs/2501.00321)). 31
  scenarios, 10,000 human-verified QA pairs, most models below 50; a **private test set**
  of 1,500 images showed the same trends as the public one. *For us:* supports our
  private-split design.
* **CC-OCR** — Yang et al., ICCV 2025 ([arXiv:2412.02210](https://arxiv.org/abs/2412.02210)).
  Multi-scene, multilingual, document parsing and key-information tracks (7,058 images,
  41 % from real applications); finds weaknesses in grounding, multi-orientation and
  **repetition hallucination**. *For us:* ✅ repetition detector.
* **OmniDocBench** — Ouyang et al., CVPR 2025 ([code](https://github.com/opendatalab/OmniDocBench)).
  Page-level parsing with reading-order annotations; text scored by normalised edit
  distance averaged per sample. *For us:* ✅ sample-level NED and an order-free error
  alongside CER.
* **olmOCR-Bench / olmOCR 2** — Poznanski, Soldaini et al., 2025
  ([arXiv:2510.19817](https://arxiv.org/abs/2510.19817)). 7,010 binary unit tests (text
  present, absent, in order; tables) over 1,402 PDFs instead of fuzzy matching. *For us:*
  🔜 presence and order tests for screenshots and long blocks, where one layout slip
  dominates CER.
* **Language-specific OCR benchmarks for VLMs.** KITAB-Bench for Arabic (ACL Findings
  2025; [code](https://github.com/mbzuai-oryx/KITAB-Bench)), ThaiOCRBench (IJCNLP-AACL 2025;
  [ACL Anthology](https://aclanthology.org/2025.ijcnlp-long.89/)), PsOCR for Pashto (2025;
  [arXiv:2505.10055](https://arxiv.org/abs/2505.10055)). Common findings: VLMs beat
  traditional OCR on CER (KITAB-Bench reports about 60 %), proprietary models lead,
  handwriting and fine-grained recognition drop most, and errors include language bias
  and hallucinated content. PsOCR builds its benchmark synthetically from 1,000 font
  families, as our print tiers do. *For us:* confirms the design of a language-specific,
  multi-domain benchmark; Tamil has had none.
* **GlotOCR Bench** — ICML 2026 ([arXiv:2604.12978](https://arxiv.org/abs/2604.12978)).
  158 scripts, clean and degraded renderings; on 148 of them every model scores below
  10 % Acc@5. *For us:* motivates testing scripts beyond modern Tamil (Tamil-Brahmi,
  Grantha) and degraded renderings.
* **Multilingual document parsing.** MORE (ICML 2026; 149 languages, real documents;
  [arXiv:2607.02956](https://arxiv.org/abs/2607.02956)) and MDPBench
  ([code](https://github.com/Yuliang-Liu/MultimodalOCR/blob/main/MDPBench/README.md)).

## 3. How vision-language readers fail

* **When Low CER is Not Enough** — 2026 ([arXiv:2607.24077](https://arxiv.org/abs/2607.24077)).
  On historical Uruguayan documents VLMs beat traditional OCR on CER and WER but produce
  orthographic normalisation, spurious content and meaning-changing substitutions that
  the metrics barely register. *For us:* ✅ failure-mode diagnostics; the CICT subset's
  diplomatic reference already penalises normalisation.
* **Reading or Guessing?** — Karamolegkou, Angleraud, Sagot, Clérice (Inria), 2026
  ([arXiv:2605.27750](https://arxiv.org/abs/2605.27750)). On Ancient Greek and Arabic
  editions, VLMs show repetition collapse, markup emission and **off-script generation**,
  and under controlled character perturbations drift further from the page than
  traditional recognisers. *For us:* ✅ markup and wrong-script detectors; 🔜 perturbed-text
  items that generalise the recitation index.
* **WildHandBench** — 2026 ([arXiv:2608.22959](https://arxiv.org/abs/2608.22959)). 500
  handwritten documents with calibrated **human baselines** (humans 77.09 vs best model
  71.85) and a **Prior-Driven Error** metric: 63–91 % of model errors come from language
  priors, against 49 % for humans. *For us:* ✅ the recitation index is our prior-driven
  measure on a famous text; 🔜 human baselines.
* **Seeing is Believing? (KIE-HVQA)** — NeurIPS 2025 ([arXiv:2506.20168](https://arxiv.org/abs/2506.20168)).
  OCR hallucination under document degradation. *For us:* 🔜 blank and illegible
  controls whose correct answer is empty.
* **Can OCR-VLMs Read Devanagari?** — 2026 ([arXiv:2606.29213](https://arxiv.org/abs/2606.29213),
  [code](https://github.com/Aditya-PS-05/devanagari-ocr-benchmark)). Ten systems, from
  EasyOCR to frontier models: on clean synthetic text all score chrF++ 91–98, but on real
  scans nine of ten collapse; one OCR-VLM loops to 71× the reference length; English OCR
  rank does not predict Hindi rank. *For us:* the strongest warning about synthetic
  proxies. ✅ proxy-validity check and repetition detector; real data first in the roadmap.
* **Evaluating LLMs for Historical Document OCR** — Levchenko, LM4DH 2025
  ([ACL Anthology](https://aclanthology.org/2025.lm4dh-1.7/)). Period-specific metrics
  (historical character preservation, archaic insertion) and protocols for contamination
  control and stability testing on 18th-century Russian print; LLM post-correction made
  results worse. *For us:* 🔜 stability (repeat-run) testing; our pre-reform and palm-leaf
  policies already decide which historical conventions count.
* **How Far Can Synthetic Data Take Thai OCR?** — 2026 ([arXiv:2609.03595](https://arxiv.org/abs/2609.03595)).
  Typeface diversity, two-dimensional structure and real handwriting glyphs drive transfer
  from synthetic to real. *For us:* supports wide font coverage; handwriting remains the
  weakest proxy.

## 4. Measuring OCR output

* **A survey of OCR evaluation tools and metrics** — Neudecker et al., HIP 2021
  ([ACM](https://dl.acm.org/doi/10.1145/3476887.3476888)). Tools disagree because of
  unspecified details; error rates are often not comparable. *For us:* normalisation is
  specified step by step and the worked examples are unit tests.
* **Flexible Character Accuracy** — Clausner, Pletschacher, Antonacopoulos, *Pattern
  Recognition Letters* 131, 2020 ([PRImA](https://www.primaresearch.org/publications/PRL_Clausner_FlexibleCharacterAccuracy)).
  A reading-order-independent accuracy. **The Character Error Vector** — Bourne, Simbeye,
  Nockels, 2026 ([arXiv:2604.06160](https://arxiv.org/abs/2604.06160)): a bag-of-characters
  page metric that splits parsing from recognition errors. *For us:* ✅ bag-of-characters
  error and order gap. On the real leaves, about a third of Tesseract's error is ordering.
* **CATMuS Medieval** — ICDAR 2024 ([data](https://zenodo.org/records/12743230)).
  Consistent graphemic transcription guidelines across 200+ manuscripts. *For us:* our
  annotation guidelines take the same diplomatic stance.

## 5. Handwriting and historical documents with LLMs

* **Benchmarking Large Language Models for Handwritten Text Recognition** — 2025
  ([arXiv:2503.15195](https://arxiv.org/abs/2503.15195)). Multimodal LLMs are competitive
  on modern English handwriting; specialised systems (Transkribus) keep the lead on
  non-English and historical documents. *For us:* expect Tamil palm leaves and
  inscriptions to remain hard; the real CICT leaves are where this shows.
* **Unlocking the Archives** — Humphries et al., *Historical Methods* 58(3), 2025
  ([arXiv:2411.03340](https://arxiv.org/pdf/2411.03340)). LLM transcription and correction
  of English historical handwriting reached CER 1.8 %. This is a high-resource ceiling to
  compare Tamil against.
* **Indic handwriting competitions.** ICDAR 2023 Competition on Indic Handwriting Text
  Recognition; **ICDAR 2025 IHDR** ([site](https://ilocr.iiit.ac.in/icdar_2025_Indic_HDR/)):
  10,000 mobile-camera page images in ten Indic languages including Tamil, about 1,000
  writers. PLATTER, page-level Indic HTR ([arXiv:2502.06172](https://arxiv.org/abs/2502.06172)).
  📦 the IHDR Tamil pages are the best candidate to replace our font-based handwriting
  proxy.

## 6. Tamil and Indian-language OCR / HTR

### Benchmarks and evaluations

* **Zero-shot OCR Accuracy of Low-Resourced Languages: Sinhala and Tamil** — RANLP 2025
  ([ACL Anthology](https://aclanthology.org/2025.ranlp-1.56/)). Six engines (Cloud Vision,
  Surya, Document AI, Tesseract, Subasa, EasyOCR); Document AI was best for Tamil (CER
  0.78 %) on their data; includes a synthetic Tamil benchmark set. *For us:* the closest
  prior Tamil OCR comparison; it covers modern print only.
* **IndicVisionBench** — Krutrim, ICLR 2026 ([arXiv:2511.04727](https://arxiv.org/pdf/2511.04727),
  [data](https://huggingface.co/datasets/krutrim-ai-labs/IndicVisionBench)). An OCR track of
  876 document images in 10 Indic languages including Tamil, alongside VQA and translation.
* **Sarvam Indic OCR Bench** — vendor benchmark ([blog](https://www.sarvam.ai/blogs/sarvam-vision)).
  Block-level samples in 22 Indian languages from newspapers, textbooks and historical
  writing (1800–present). Vendor-run and vendor-reported.
* **Designing Production-Scale OCR for India** — Faraz, Kolla, Kulkarni, Agarwal, 2026
  ([arXiv:2602.16430](https://arxiv.org/abs/2602.16430)). Fine-tuning an OCR-specialised
  model beat end-to-end multilingual VLM training on accuracy–latency trade-offs.
* **OCR Synthetic Benchmark Dataset for Indic Languages** — Saini et al., 2022
  ([arXiv:2205.02543](https://arxiv.org/abs/2205.02543)). 90k synthetic images in 23
  languages.

**What none of them cover:** historical Tamil scripts, palm leaves, inscriptions,
pre-reform print, script and medium identification, or tests of reading against
recitation. That is the gap this benchmark fills.

### Datasets (📦 candidates for real subsets)

| Dataset | What | Tamil content | Notes |
|---|---|---|---|
| [Mozhi](https://arxiv.org/abs/2205.06740) (IIIT Hyderabad) | Printed line and word images, 13 languages | Yes | 1.2M words / about 120k lines in all |
| [IIIT-INDIC-HW-WORDS](https://cvit.iiit.ac.in/research/projects/cvit-projects/iiit-indic-hw-words) (ICDAR 2021) | Handwritten words, 8 scripts | Yes | 872k instances, 135 writers |
| [ICDAR 2025 IHDR](https://ilocr.iiit.ac.in/icdar_2025_Indic_HDR/) | Handwritten pages, mobile camera | Yes | Page-level, about 1,000 writers |
| [IndicSTR12](https://cvit.iiit.ac.in/research/projects/cvit-projects/indicstr) (ICDAR 2023) | Real scene word images, 12 languages | Yes, ≥1,000 words | Web-crawled images |
| [Bharat Scene Text](https://github.com/Bhashini-IITJ/BharatSceneTextDataset) (IJDAR 2026) | 6,582 Wikimedia Commons scene images, 11 languages | 513 test words | Includes **script identification** |
| [uTHCD](https://arxiv.org/pdf/2103.07676) (IEEE Access 2021) | Isolated handwritten characters | 156 classes, about 91k samples | Online and offline |
| [HP Labs Tamil characters](https://lipitk.sourceforge.net/datasets/tamilchardata.htm) | Isolated handwritten characters | 156 classes, 82,928 samples | Research use only |
| [THPLMD](https://www.sciencedirect.com/science/article/pii/S2352340924000738) (*Data in Brief*, 2024) | Palm-leaf manuscript images with binarised ground truth | Nālaṭiyār, Tolkāppiyam, Tirikaṭukam | Character-level |
| [CICT Tirukkural GT](https://www.digitalarchives.cict.in/) | Palm leaves with diplomatic transcriptions | 133 leaves | ✅ used: 26 test leaves |
| GHTNet stone inscriptions ([Heritage Science 2025](https://www.nature.com/articles/s40494-025-02097-9)) | 1,500 stone inscription images | Tamil-Brahmi (810), Vatteluttu (720) | The only Vatteluttu images found |
| Vatteluttu characters ([JISIS 2025](https://jisis.org/article/2025.I1.030/71775/)) | 1,800 segmented images, 28 characters | Vatteluttu | Character-level |

### Historical-script recognition

Tamil-Brahmi ([Neural Computing and Applications 2024](https://link.springer.com/article/10.1007/s00521-024-10137-x)),
ancient Tamil inscriptions ([Heritage Science 2024](https://www.nature.com/articles/s40494-024-01522-9);
[GHTNet 2025](https://www.nature.com/articles/s40494-025-02097-9)), Vatteluttu
([JISIS 2025](https://jisis.org/article/2025.I1.030/71775/)), Grantha
([CNN, 2023](https://link.springer.com/article/10.1007/s41870-023-01247-1);
[palm-leaf shape context](https://ieeexplore.ieee.org/document/8282574/)), Tamil palm leaves
([Heritage Science 2024](https://www.nature.com/articles/s40494-024-01438-4)), and the
reviews of [computational epigraphy](https://arxiv.org/abs/2406.06570) and
[ancient-script recognition](https://arxiv.org/pdf/2506.19208).

Nearly all of this work classifies **isolated, pre-segmented characters** on small datasets
that are not shared, and reports accuracy on its own split. Results are not comparable
across papers, and none evaluates line- or page-level reading into Unicode. This is the
main argument for a shared, line- and leaf-level benchmark with public references, and the
reason the real data listed above needs line-level transcriptions before it can be used.

---

## Next steps, in priority order

1. **Real data for every proxy** (largest validity gain): IHDR Tamil pages for
   handwriting, Bharat Scene Text and IndicSTR12 for scene text, Mozhi for print, and more
   CICT leaves, each after a licence check (`real-data.md`).
2. **Perturbed real text**, generalising the recitation index to all reading subsets.
3. **Blank and illegible controls** for hallucination.
4. **Human baselines** on `lite` from expert readers of Tamil, Tamil-Brahmi and Grantha.
5. **Prompt-robustness and stability** diagnostics.
6. **Signal-to-noise per subset** once enough systems are scored.
