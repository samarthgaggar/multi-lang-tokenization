# Exploratory context-cluster audit

## Data status (2026-10-26)

Sentence collection is complete for the four languages available in the cached Global Voices streams. The saved database contains:

| Language | Sentences | Word occurrences | Distinct lowercase word forms |
|---|---:|---:|---:|
| English | 358,369 | 6,979,320 | 148,199 |
| Spanish | 247,760 | 5,012,157 | 143,864 |
| Russian | 126,490 | 2,110,189 | 163,459 |
| Hindi | 2,283 | 45,085 | 7,997 |

Fresh repeat checks reproduced 24 of 24 sampled saved sentences exactly for each language. Ukrainian and Telugu are not present in the available dataset streams.

## Method in brief

For a selected word, each sentence containing it is represented by counts of the other words in that sentence; all occurrences of the selected word are left out. K-means groups the sentence contexts without using FlexiTokens' split labels. The split patterns are compared with those groups afterwards. By default, the requested number of groups equals the number of distinct splits observed for the selected word. A pure group has one observed split; a mixed group has more than one.

## Full-corpus word summary

Among word forms seen at least twice, the share with more than one observed FlexiTokens split was:

| Language | Repeated forms | Repeated forms with multiple splits | Share |
|---|---:|---:|---:|
| English | 73,083 | 29,171 | 39.9% |
| Spanish | 73,300 | 25,086 | 34.2% |
| Russian | 81,562 | 24,597 | 30.2% |
| Hindi | 3,551 | 740 | 20.8% |

These percentages count word forms, not token occurrences. They say that the same lowercase spelling was observed with more than one split at least twice in the corpus. They do not say whether the changes were contextually appropriate.

## Example context-cluster analyses

These five examples were selected from the completed data because each appears repeatedly and has multiple observed splits. They are illustrative examples, not a random sample of all words.

| Language | Word | Occurrences (sentence contexts) | Observed splits / clusters | Pure / mixed clusters | Most common split |
|---|---|---:|---:|---:|---|
| English | `organizing` | 271 (270) | 11 / 11 | 0 / 11 | `organizing` (43.5%) |
| English | `political` | 6,872 (6,567) | 6 / 6 | 0 / 6 | `pol|itical` (84.6%) |
| Spanish | `censura` | 736 (702) | 8 / 8 | 0 / 8 | `censura` (93.3%) |
| Russian | `губернатор` | 50 (49) | 7 / 7 | 1 / 6 | `губерн|атор` (56.0%) |
| Hindi | `फोटो` | 32 (27) | 4 / 4 | 0 / 4 | `फोटो` (50.0%) |

For English, the same spelling appeared as `organizing` in “The last thing he was organizing was a protest march…” and as `organizin|g` in “We shouldn’t spend our energy organizing people on the Internet…”. The context groups for this example still contained multiple split patterns. For Spanish, the whole-word split dominated, although every cluster included more than one observed pattern. The one pure Russian cluster contains only one occurrence, so it is weak evidence of a consistent context-to-split relationship.

For `political`, all six groups contained four to six splits, and `pol|itical` was still the dominant split in every group (74.4–87.6% of group occurrences). In these examples, word-count context clusters did not cleanly separate the observed splits. This is an exploratory finding for five selected examples; it does not establish whether context matters across the corpus or why FlexiTokens produced a particular boundary.

An additional English check of `read` found 1,979 occurrences across 1,935 sentences. The two observed patterns were `read` (87.2%) and `R|ead` (12.8%). Both context groups contained both patterns in almost identical proportions, so this example did not show a split difference associated with pronunciation; the variation tracks capitalization. The saved tables are in `results/global_voices_context_analysis/word_explorations/en/read_*`.


## Additional word lookups

I also explored eight more words. The table counts pure and mixed K-means groups only; a sentence with no other pre-tokenized word is an empty context and is not assigned to K-means.

| Language | Word | Occurrences (sentences) | Splits / K-means groups | Pure / mixed groups | Most common split | Largest split share within groups |
|---|---|---:|---:|---:|---|---:|
| English | `organizations` | 1,161 (1,142) | 9 / 9 | 0 / 9 | `organizat|ion|s` (55.6%) | 37.3–67.5% |
| English | `silence` | 552 (542) | 7 / 7 | 0 / 7 | `sil|ence` (94.6%) | 81.3–97.8% |
| Spanish | `política` | 3,319 (3,212) | 8 / 8 | 0 / 8 | `pol|ític|a` (57.2%) | 46.3–71.0% |
| Spanish | `esperanza` | 730 (721) | 8 / 8 | 0 / 8 | `esperanza` (61.8%) | 37.3–75.7% |
| Russian | `фотографии` | 797 (779) | 9 / 9 | 0 / 9 | internal byte cuts (63.6%) | 50.0–76.1% |
| Russian | `группа` | 698 (687) | 6 / 6 | 0 / 6 | internal byte cuts (42.0%) | 37.6–53.3% |
| Hindi | `साथ` | 110 (103) | 3 / 3 | 0 / 3 | whole word (73.6%) | 60.0–81.8% |
| Hindi | `लोगों` | 83 (81) | 3 / 3 | 1 / 2 | whole word (96.4%) | 85.7–100.0% |

For `política`, one sentence had no other word after removing the target; `silence` had two such occurrences. These were saved as empty contexts and not assigned to K-means. For `लोगों`, one pure group contained 29 sentence contexts (30 appearances), all with the whole-word split; its other two groups were mixed. The Russian and Hindi outputs mostly contain byte cuts inside written characters, so their displayed patterns should not be read as ordinary letter splits. Across these eight selected words, 1 of 53 K-means groups was pure and 52 were mixed. The pure group was for Hindi `लोगों` and contained 29 sentence contexts (30 appearances). This is a result for selected examples, not a corpus-wide rate. These examples do not establish that context causes the split.

For Russian and Hindi, some displayed split labels include `[byte cuts: ...]`. These are boundaries in the UTF-8 byte input, and may fall inside a written character; they are not readable letter or space boundaries.

## Saved example outputs

The context assignments, per-split counts, per-cluster split counts, and sampled similarity matrices are saved under `results/global_voices_context_analysis/word_explorations/` in language-specific folders. The notebook contains examples including English `organizing` and `political`, and lets the reader enter another word and language.
