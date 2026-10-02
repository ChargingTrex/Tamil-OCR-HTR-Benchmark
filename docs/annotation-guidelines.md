# Transcription guidelines

These conventions define the references of real subsets, and they describe what the
synthetic references already are. Scoring normalises some conventions automatically
(methodology §5.1). The rules below say what a transcriber writes; the scoring policy
decides what is compared.

## General

* **Write what is there.** Transcribe the letters on the artefact, not the text you know
  it should be. Do not correct spelling, restore missing letters, or substitute a
  canonical edition (for example, a Tirukkural couplet keeps the scribe's spelling).
* **Unicode.** UTF-8, NFC. No zero-width characters. ஸ்ரீ is written with ஸ (U+0BB8).
* **Line breaks** follow the artefact. They are not scored, because whitespace is
  normalised.
* **Word spaces** are written where the artefact has them. Where it has none (most
  inscriptions and leaves), insert spaces at word boundaries if you can; the scoring policy
  for those subsets ignores spacing.
* **Punctuation** as written. Daṇḍas in Grantha may be written `|` and `||` or `।` and `॥`.
* **Damaged or illegible text.** A reference must contain only letters that can be read
  in the image. If a letter is lost, crop or mask it out of the scored region, or drop the
  item. Never guess, and never use markup for lacunae, since it would be scored as text.

## By script

### Modern Tamil
Standard Tamil Unicode, spelled as written, including Grantha letters (ஜ ஷ ஸ ஹ க்ஷ ஶ)
where the text uses them. Latin words and Arabic digits stay as they are.

### Pre-1978 Tamil (print and manuscripts)
Old ligatures are written with the ordinary modern code points: the old ணா is `ணா`, the old
னை is `னை`. The reform changed glyphs, not spelling.

### Palm-leaf manuscripts (diplomatic)
* Keep the scribe's spelling, including a missing puḷḷi and e/ē or o/ō written alike. The
  epigraphic policy folds these when scoring, so a transcriber may also write them as they
  are, without restoring anything.
* Verse numbers in Tamil numerals are transcribed (`௧௨`); the CICT policy strips them when
  scoring, because editions differ on them.
* Marginal titles and folio numbers outside the text block are not transcribed.

### Tamil-Brahmi
Transliterate into modern Tamil letters (the reading, not the Brahmi code points). In
TB-I and TB-II, where a bare consonant may stand for a consonant alone or with *a*, write
the intended word if the reading is established; otherwise read as TB-III.

### Grantha (Sanskrit)
IAST, lower case: ā ī ū ṛ ṝ ḷ, ṅ ñ ṭ ḍ ṇ, ś ṣ, ṃ (anusvāra), ḥ (visarga), `'`
(avagraha). ISO 15919 spellings (ṁ, r̥, ē, ō) are accepted and folded to IAST. Word spaces
may follow sandhi as printed in a standard edition, but spacing is not scored.

### Grantha–Tamil (maṇipravāḷam)
Each letter is written in the transcription of its own script. Tamil-script letters go
in Tamil Unicode, and Grantha-script letters go in IAST, including inside a word: a
Grantha stem with a Tamil ending is `kalyāṇaguṇaங்களை`. Tamil-block Grantha letters
used inside Tamil script (ஜ ஷ ஸ ஹ) are Tamil script and stay in Tamil Unicode. The
Tamil puḷḷi and Tamil vowel length are folded when scoring, as on other leaves.

### Numerals and signs
Tamil numerals stay Tamil numerals (௧ ௨ ௰ ௱ ௲). Day, month, year, debit, credit and rupee
signs use their Tamil block code points (௳ ௴ ௵ ௶ ௷ ௹). Fractions and measures use the
Tamil Supplement block (U+11FC0–U+11FFF).

## Process for real data

1. Two transcribers work independently from the image.
2. Their transcriptions are compared with the benchmark's own CER under the subset's
   policy. Items above 2 % CER (print) or 5 % (manuscripts and inscriptions) go to an
   adjudicator, who rules on each difference against the image.
3. The manifest records the transcriber, the reviewer, and the source's own transcription
   where one exists, kept as provenance and not used as the reference.
