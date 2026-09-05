# Deterministic FLORES FlexiTokens segment vocabularies

This is the canonical result set for the six-language FLORES analysis.

For a concise walkthrough of the method, tables, charts, and inspection examples, open [`../../deterministic_flores_vocabulary.ipynb`](../../deterministic_flores_vocabulary.ipynb).

## Method

- Dataset: FLORES `dev` and `devtest`, 2,009 complete sentences per language.
- Languages: English, Spanish, Russian, Ukrainian, Hindi, and Telugu.
- Input unit: each original FLORES sentence is tokenized as one complete input; sentences are not split on whitespace.
- Boundary rule: a boundary is placed exactly when `sigmoid(boundary_logit) > 0.5`.
- Randomness: none. The checkpoint's Gumbel sampling path is disabled for this evaluation.
- Vocabulary: the unique decoded output segments produced across all sentence tokenizations, with each segment's corpus count.

## Results

| Language | Unique output segments | Total segment occurrences | Mean segments per sentence |
| --- | ---: | ---: | ---: |
| English | 9,895 | 76,373 | 38.02 |
| Spanish | 11,741 | 75,943 | 37.80 |
| Russian | 12,137 | 76,948 | 38.30 |
| Ukrainian | 10,980 | 77,605 | 38.63 |
| Hindi | 8,146 | 75,145 | 37.40 |
| Telugu | 10,695 | 75,913 | 37.79 |

`summary.csv` contains these aggregate values. For each language, `*_segment_vocabulary.csv` contains every vocabulary segment, its count, and example source sentences; `*_sentence_tokenizations.csv` records the full-sentence segmentations used to build it.

The charts in `figures/` visualize vocabulary size, segmentation length, total output-segment occurrences, average reuse per vocabulary entry, and the share of entries appearing once. They are descriptive corpus statistics, not a claim about downstream model quality.
