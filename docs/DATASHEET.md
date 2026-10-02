# Datasheet: Tamil OCR / HTR Benchmark, v1

Following *Datasheets for Datasets* (Gebru et al., 2021), with a self-assessment against
the BetterBench criteria (Reuel et al., 2024) at the end.

## Motivation

* **Purpose.** To measure how well systems read Tamil writing from images, from
  Tamil-Brahmi to the modern script and from screens to palm leaves, stone and metal, and
  to rank frontier multimodal models, open-weights models, OCR services and OCR engines on
  one scale.
* **Gap addressed.** Existing Tamil and Indian-language OCR evaluations cover modern print,
  scene text or isolated handwritten characters. None covers historical scripts,
  manuscripts or inscriptions at line or leaf level, script and medium identification, or
  reading against recitation (see [`RELATED-WORK.md`](RELATED-WORK.md)).
* **Creators and funding.** Built in the open in this repository. The real palm-leaf data
  is the work of the **Central Institute of Classical Tamil (CICT)**.

## Composition

* **Instances.** 1,654 test images in 18 subsets: 15 reading subsets, 2 identification
  subsets and 1 image-to-English subset (methodology §2.4). A `lite` split of 371 items is
  a subset of `test`.
* **Real vs synthetic.** 26 items are real: leaves of a Tirukkural palm-leaf manuscript
  from the *CICT Tirukkural Ground Truth Corpus* (Central Institute of Classical Tamil,
  CC BY 4.0, data curator Kannan Krishnan), with CICT's expert diplomatic
  transcriptions. All other items are rendered by this repository from known text, so
  their references match the image by construction.
* **Text sources.** Sentences, signage, plaques, names, interface strings, inscription-style
  and maṇipravāḷam-style lines written for the benchmark; public-domain classical Tamil and
  Sanskrit (methodology §3.1). No personal data: personal names in the `names` pool are
  common names, not records of individuals.
* **Labels.** Each item has a reference (Tamil Unicode, IAST for Grantha, Tamil + IAST for
  Grantha–Tamil, an English translation, or a class label) and metadata: script, medium,
  granularity, provenance, lexical kind (corpus / random words / nonce), source text,
  rendering parameters, licence, and the contamination canary. CICT leaves also carry the
  source DOI and the standard edition text of their couplets.
* **Known errors.** Synthetic references are exact. CICT transcriptions reflect their
  editors' conventions; the scoring policy neutralises spacing, puḷḷi and vowel length.
* **Splits.** `test` (ranked), `lite` (quick checks), and a regenerable `private` split
  with a secret seed for verifying suspicious results.

## Collection and processing

* **Rendering.** HarfBuzz-shaped text in 33 OFL fonts, with medium simulators for print,
  screens (real Chromium renders), handwriting, scene text, palm leaves, stone,
  estampages, copper plates and pottery. Degradation never erases letters (§3.6).
* **Real data processing.** CICT leaves are cropped to the leaf outline and the
  digitiser's red caption is painted out, so that verse numbers cannot be read off the
  caption. Only leaves in the corpus's held-out `val` and `test` splits are used.
* **Determinism.** Every synthetic item is a pure function of (version, seed, split,
  subset, index); rebuilding reproduces byte-identical images, verified by SHA-256.

## Uses

* **Intended.** Comparing systems that read Tamil from images; diagnosing failure modes
  (wrong-script output, recitation, ordering errors); tracking progress on historical
  scripts.
* **Not intended.** Training (the test data carries a canary; training on it disqualifies
  a result), or claims about real handwriting, stone or copper from the synthetic proxies
  alone.

## Distribution and licences

Code Apache-2.0; synthetic data CC BY 4.0; CICT leaves and transcriptions CC BY 4.0 with
attribution to CICT; fonts SIL OFL 1.1 ([`DATA_LICENSE.md`](../DATA_LICENSE.md)).

## Maintenance

* **Versioning.** New or changed items, subsets or metrics create a new benchmark version;
  results across versions are not compared (§8).
* **Feedback.** GitHub issues on the repository. Errors in references are fixed in the
  next version and listed in its notes.
* **Contamination.** The canary string in every manifest row lets the data be filtered
  from training corpora; the private split detects memorisation of the public one.

## BetterBench self-assessment

| Stage | What is in place | What is missing |
|---|---|---|
| Design | Stated purpose and scope; taxonomy of scripts, media and tasks; nonce controls; real-data anchor; documented gaps | Review by Tamil epigraphists and palaeographers; human expert baselines |
| Implementation | Open evaluation code; deterministic data build; checksums; adapters for API and local systems; unit tests including the documented worked examples | Repeated-run stability checks for stochastic models |
| Documentation | Methodology with prompts verbatim; rubric; datasheet; licences; related work; per-item provenance | A paper with an external peer review |
| Maintenance | Versioning policy; feedback channel; private split; canary; CI | A named maintainer group and a release schedule |
