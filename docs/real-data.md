# Real data: what is in, what is wanted, and how to add it

The benchmark labels every item `synthetic` or `real`. Synthetic items are rendered by this
repository from known text, so their references are exact by construction; real items are
photographs or scans of artefacts with human transcriptions. Synthetic media are proxies
for real ones. The aim is to replace or complement every proxy with licensed real data
(methodology §7 and §9).

## Admission rules for a real subset

1. **Licence that allows redistribution** of the images and transcriptions (CC BY, CC BY-SA,
   CC0, public domain, or written permission recorded in the repository). Licences are
   recorded per item in the manifest.
2. **Expert transcriptions** aligned to the image unit being scored (word, line, leaf or
   page), following [`annotation-guidelines.md`](annotation-guidelines.md).
3. **Held-out items only** where the source has train/test splits, so that systems trained
   on the source's training data are not tested on it.
4. **Provenance per item**: source URL or DOI, holding institution, transcriber, and any
   masking applied (e.g. a digitiser's caption painted out).
5. **No silent cleaning.** References keep the scribe's spelling (diplomatic); conventions
   that are not reading skills (spacing, puḷḷi, vowel length) are neutralised by the
   scoring policy, never by editing the reference.

## In `v1`

| Subset | Source | Licence | Items |
|---|---|---|---:|
| `palm-leaf-cict` | Central Institute of Classical Tamil, *CICT Tirukkural Ground Truth Corpus* (Zenodo, one DOI per leaf); only leaves in the source's `val` and `test` splits | CC BY 4.0 | 26 leaves |

Each leaf is used whole (ten text lines), because the published line crops are not
aligned with the line transcriptions on some leaves. The digitiser's red caption at the
bottom of each scan is painted out (`external/cict.py: mask_caption`), so the image shows
only the scribe's writing. Rebuild with:

```bash
git clone https://github.com/ChargingTrex/tamil-ocr
tamilbench build --out data/v1 --cict-root tamil-ocr/cict-htr
```

## Wanted, by gap

Candidates are listed with their licence status; [`RELATED-WORK.md`](RELATED-WORK.md)
describes each dataset. "To confirm" means the licence has not
been verified for redistribution; such data is not added until it is.

| Gap | Candidate sources | Licence status |
|---|---|---|
| Grantha and Grantha–Tamil manuscripts | Grantha–Tamil photography pages of the CIIL library ([library.ciil.org](https://library.ciil.org/Sites/Photography/GranthaTamil.html)); the Śaiva manuscripts of the French Institute of Pondicherry (UNESCO Memory of the World) | To confirm. These need expert transcriptions (Tamil-script parts in Tamil, Grantha parts in IAST) as well as permission |
| More palm leaves | Further CICT leaves as they are published; THPLMD (*Data in Brief*, 2024); Tamil Palm Leaf Character Dataset (Mendeley Data, doi:10.17632/b7vhz7z83k.1) | CICT: CC BY 4.0. Others: to confirm |
| Old print | Tamil Wikisource proofread pages (page scan plus community-validated text); Mozhi printed Tamil lines (IIIT Hyderabad) | Wikisource scans mostly public domain, text CC BY-SA (share-alike); Mozhi: to confirm |
| Stone and copper (medieval Tamil letterforms) | Photographs of published inscriptions aligned with the readings in *South Indian Inscriptions* and state archaeology reports | To confirm, per photograph |
| Vatteluttu | Hero stones and grants in Tamil Nadu and Kerala with published readings; the stone-inscription images of the GHTNet study (*Heritage Science*, 2025: 720 Vatteluttu, 810 Tamil-Brahmi) | To confirm; these need line-level readings, not character labels |
| Tamil-Brahmi | Cave inscriptions and Keezhadi / Kodumanal potsherds with published readings | To confirm, per photograph |
| Handwriting | ICDAR 2025 IHDR Tamil pages (mobile-camera handwritten pages); IIIT-INDIC-HW-WORDS Tamil words | To confirm (several academic sets are research-only) |
| Scene text | Bharat Scene Text (Wikimedia Commons images; Tamil test words, with script labels); IndicSTR12 Tamil words | Bharat Scene Text: per-image Commons licences, dataset terms to confirm; IndicSTR12: web-crawled, to confirm |

## Contributing a dataset

Open an issue with the source, licence and a sample of ten items. A real subset is a
directory with the images and an `items.jsonl` whose rows look like this:

```json
{"id": "src-0001", "image": "images/src-0001.jpg", "text": "…diplomatic transcription…",
 "script": "grantha-tamil", "medium": "palm-leaf", "granularity": "page",
 "source": {"url": "…", "doi": "…", "holder": "…"}, "license": "CC-BY-4.0",
 "attribution": "…", "transcriber": "…", "reviewer": "…", "split": "test"}
```

`script` and `medium` take the ids of `src/tamilbench/taxonomy.py`. A maintainer adds a
`SubsetSpec` with `provenance=REAL` and a scoring policy, and an importer under
`src/tamilbench/external/` (see `cict.py`). Adding a subset changes the benchmark version
(methodology §8), so real subsets are batched into the next release.
