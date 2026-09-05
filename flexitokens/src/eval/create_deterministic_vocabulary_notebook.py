#!/usr/bin/env python3
"""Create a concise, executed-data notebook for the deterministic FLORES report."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK = ROOT / "deterministic_flores_vocabulary.ipynb"


def main():
    def markdown(source):
        return {"cell_type": "markdown", "metadata": {}, "source": source}

    def code(source):
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

    cells = [
        markdown(
            "# Deterministic FlexiTokens vocabularies on FLORES\n\n"
            "This is the canonical report for the six-language FLORES analysis. It replaces the earlier seed-based and whitespace-split analysis."
        ),
        markdown(
            "## Method\n\n"
            "- Each original FLORES sentence is tokenized as one complete input.\n"
            "- No seeds or random boundary samples are used.\n"
            "- A boundary is placed exactly when `sigmoid(boundary_logit) > 0.5`.\n"
            "- The vocabulary is the counted set of unique decoded output segments across all complete-sentence tokenizations.\n"
            "- Every vocabulary entry includes up to three source-sentence examples for manual inspection."
        ),
        code(
            "from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import SVG, display\n\n"
            "RESULT_CANDIDATES = [\n"
            "    Path('results/deterministic_sentence_segment_vocabulary_full_v2'),\n"
            "    Path('flexitokens/results/deterministic_sentence_segment_vocabulary_full_v2'),\n"
            "]\n"
            "RESULTS = next((path for path in RESULT_CANDIDATES if (path / 'summary.csv').is_file()), None)\n"
            "if RESULTS is None:\n"
            "    raise FileNotFoundError('Run this notebook from the repository root or the flexitokens folder.')\n"
            "metadata = json.loads((RESULTS / 'run_metadata.json').read_text())\n"
            "assert metadata['boundary_rule'] == 'sigmoid(boundary_logit) > 0.5'\n"
            "assert metadata['randomness'] == 'none; Gumbel sampling is explicitly disabled for this evaluation'\n"
            "summary = pd.read_csv(RESULTS / 'summary.csv')\n"
            "assert (summary['sentences'] == 2009).all()\n"
            "summary[['language', 'sentences', 'unique_output_segments_vocabulary_size', 'total_output_segment_occurrences', 'mean_segments_per_sentence']]"
        ),
        markdown("## Summary charts\n\nThese descriptive charts are generated from the completed result CSVs. They describe the observed corpus output and do not measure downstream model quality."),
        code(
            "for chart in [\n"
            "    'vocabulary_size_by_language.svg',\n"
            "    'mean_segments_per_sentence.svg',\n"
            "    'total_segment_occurrences_by_language.svg',\n"
            "    'mean_occurrences_per_vocabulary_entry.svg',\n"
            "    'singleton_vocabulary_entries.svg',\n"
            "]:\n"
            "    display(SVG(filename=RESULTS / 'figures' / chart))"
        ),
        markdown("## Manual inspection examples\n\nThe following are actual stored full-sentence segmentations. Spaces and punctuation shown inside a segment are part of that decoded segment."),
        code(
            "english = pd.read_csv(RESULTS / 'en_sentence_tokenizations.csv')\n"
            "english.loc[english['text'].str.contains('invention', regex=False), ['text', 'segments', 'segment_count']].head(1)"
        ),
        code(
            "for language in ['en', 'es', 'ru', 'uk', 'hi', 'te']:\n"
            "    vocabulary = pd.read_csv(RESULTS / f'{language}_segment_vocabulary.csv')\n"
            "    print(f'\\n{language}: {len(vocabulary):,} unique output segments')\n"
            "    display(vocabulary[['segment_display', 'count', 'example_occurrences']].head(5))"
        ),
        markdown(
            "## Files\n\n"
            "- `summary.csv`: per-language totals.\n"
            "- `*_segment_vocabulary.csv`: every unique segment, its count, and source examples.\n"
            "- `*_sentence_tokenizations.csv`: the complete sentence tokenizations used to build the vocabularies.\n"
            "- `README.md`: method, result table, and interpretation boundaries."
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.write_text(json.dumps(notebook, indent=1) + "\n")


if __name__ == "__main__":
    main()
