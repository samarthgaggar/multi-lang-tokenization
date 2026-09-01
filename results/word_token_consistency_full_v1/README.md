# FLORES word-vocabulary FlexiTokens consistency analysis

## Protocol

For each checkpoint-supported language (`en`, `es`, `ru`, `uk`, `hi`, `te`), this analysis uses every sentence in the public FLORES `dev` and `devtest` splits: 2,009 source sentences per language.

Each sentence is NFKC-normalized, whitespace is collapsed, and the sentence is split on whitespace. The resulting unique surface strings form that language's empirical vocabulary. Case and punctuation remain part of a vocabulary entry. Each entry records:

- its frequency in the 2,009 source sentences;
- one deterministic segmentation from the fixed 50k-BPE reference;
- five FlexiTokens segmentations, using seeds 42 through 46; and
- the number of distinct FlexiTokens signatures observed across those five seeds.

The analysis therefore measures seed-to-seed boundary consistency for individual FLORES vocabulary entries. It does not measure downstream accuracy, compression, or language-model quality. No numerical threshold defines an acceptable number of signatures; more distinct signatures mean less stable boundaries under this specific seeded inference protocol.

## Validated results

| Language | Whitespace tokens | Unique vocabulary entries | Mean distinct FlexiTokens signatures | Entries with variation |
| --- | ---: | ---: | ---: | ---: |
| English | 42,855 | 12,034 | 1.988 | 64.53% |
| Spanish | 50,900 | 13,575 | 2.053 | 66.94% |
| Russian | 38,816 | 17,545 | 2.281 | 72.68% |
| Ukrainian | 37,566 | 17,267 | 2.318 | 74.03% |
| Hindi | 50,250 | 9,674 | 2.035 | 67.58% |
| Telugu | 33,326 | 14,741 | 2.010 | 66.28% |

The six vocabularies contain 84,836 unique language-specific entries in total. BPE produced exactly one signature for every entry. FlexiTokens varied for 64.53% to 74.03% of vocabulary entries across the five seeds.

## Files

Each language directory contains `word_vocabulary_by_string.csv`, `word_tokenization_map.json`, `top_variable_words.csv`, `per_language_summary.csv`, `run_manifest.json`, and resumable `.seed_checkpoints/` files. The root `per_language_summary.csv` is written only after `combine_word_tokenization_consistency_results.py` validates all six directories.
