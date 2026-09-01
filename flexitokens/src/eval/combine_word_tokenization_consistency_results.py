#!/usr/bin/env python3
"""Validate six-language FLORES word-vocabulary consistency results.

The expected protocol uses all 2,009 public FLORES dev + devtest sentences per
language, whitespace-delimited vocabulary entries, and five FlexiTokens seeds.
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd


LANGUAGES = ("en", "es", "ru", "uk", "hi", "te")
SEEDS = [42, 43, 44, 45, 46]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", required=True)
    return parser.parse_args()


def validate_language(directory: Path, language: str) -> pd.DataFrame:
    manifest_path = directory / "run_manifest.json"
    summary_path = directory / "per_language_summary.csv"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise RuntimeError(f"{language}: missing manifest or summary")

    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("status") != "complete"
        or manifest.get("language") != language
        or manifest.get("analysis_unit") != "whitespace_word"
        or manifest.get("splits") != ["dev", "devtest"]
        or manifest.get("seeds") != SEEDS
        or manifest.get("error_count") != 0
    ):
        raise RuntimeError(f"{language}: invalid run provenance")
    artifacts = manifest.get("artifacts", {})
    vocabulary_path = directory / artifacts.get("vocabulary", "")
    variable_path = directory / artifacts.get("top_variable_entries", "")
    map_path = directory / artifacts.get("tokenization_map", "")
    if not all(path.is_file() for path in (vocabulary_path, variable_path, map_path)):
        raise RuntimeError(f"{language}: missing required word-vocabulary artifact")

    summary = pd.read_csv(summary_path)
    if len(summary) != 1:
        raise RuntimeError(f"{language}: summary must have exactly one row")
    row = summary.iloc[0]
    if (
        row["language"] != language
        or row["analysis_unit"] != "whitespace_word"
        or int(row["total_sentences"]) != 2009
        or int(row["seed_repeats"]) != len(SEEDS)
        or int(row["error_count"]) != 0
        or int(row["unique_vocabulary_size"]) < 1
        or int(row["total_whitespace_tokens"]) < int(row["unique_vocabulary_size"])
    ):
        raise RuntimeError(f"{language}: invalid word-vocabulary summary")

    vocabulary = pd.read_csv(vocabulary_path)
    if (
        len(vocabulary) != int(row["unique_vocabulary_size"])
        or vocabulary["text"].duplicated().any()
        or not (vocabulary["analysis_unit"] == "whitespace_word").all()
        or not (vocabulary["bpe_unique_signatures"] == 1).all()
        or int(vocabulary["count"].sum()) != int(row["total_whitespace_tokens"])
    ):
        raise RuntimeError(f"{language}: word-vocabulary table does not reconcile")
    parsed_signatures = vocabulary["seed_signatures"].map(json.loads)
    if not parsed_signatures.map(lambda values: set(values) == {str(seed) for seed in SEEDS}).all():
        raise RuntimeError(f"{language}: incomplete seed-signature map")
    if vocabulary["seed_signatures"].str.contains("<ERROR>", regex=False).any():
        raise RuntimeError(f"{language}: inference error was written as a signature")
    expected_varied = int(vocabulary["has_fxt_seed_variation"].sum())
    if expected_varied != int(row["strings_with_fxt_seed_variation"]):
        raise RuntimeError(f"{language}: variation count does not reconcile")
    expected_top = vocabulary[vocabulary["has_fxt_seed_variation"]].sort_values(
        ["fxt_unique_signatures", "count", "text"], ascending=[False, False, True]
    ).head(200)
    observed_top = pd.read_csv(variable_path)
    if list(observed_top.get("text", [])) != list(expected_top["text"]):
        raise RuntimeError(f"{language}: top-variable-word ranking does not reconcile")

    mapping = json.loads(map_path.read_text())
    metadata = mapping.get("metadata", {})
    if (
        metadata.get("analysis_unit") != "whitespace_word"
        or metadata.get("seeds") != SEEDS
        or int(metadata.get("unique_vocabulary_size", -1)) != len(vocabulary)
        or int(metadata.get("total_whitespace_tokens", -1)) != int(vocabulary["count"].sum())
        or set(mapping.get("vocabulary", {})) != set(vocabulary["text"])
    ):
        raise RuntimeError(f"{language}: JSON vocabulary map does not reconcile")
    return summary


def main():
    root = Path(parse_args().results_root)
    summaries = [validate_language(root / language, language) for language in LANGUAGES]
    combined = pd.concat(summaries, ignore_index=True)
    numeric_columns = combined.select_dtypes(include="number").columns
    if not all(math.isfinite(float(value)) for column in numeric_columns for value in combined[column]):
        raise RuntimeError("combined summary contains a non-finite value")
    combined.to_csv(root / "per_language_summary.csv", index=False)
    print(f"Validated {len(combined)} languages and wrote {root / 'per_language_summary.csv'}")


if __name__ == "__main__":
    main()
