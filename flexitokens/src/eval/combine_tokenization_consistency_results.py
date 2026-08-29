#!/usr/bin/env python3
"""Validate canonical FLORES runs and write one combined summary table."""
import argparse
import json
import math
from pathlib import Path

import pandas as pd


LANGUAGES = ("en", "es", "ru", "uk", "hi", "te")
REQUIRED_ARTIFACTS = (
    "per_language_summary.csv",
    "tokenization_vocabulary_by_string.csv",
    "top_ambiguous_strings.csv",
    "vocab_signatures.json",
    "run_manifest.json",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", required=True)
    return parser.parse_args()


def validate_language(directory: Path, language: str) -> pd.DataFrame:
    missing = [name for name in REQUIRED_ARTIFACTS if not (directory / name).is_file()]
    if missing:
        raise RuntimeError(f"{language}: missing required artifacts: {', '.join(missing)}")

    manifest = json.loads((directory / "run_manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("language") != language:
        raise RuntimeError(f"{language}: incomplete or mismatched manifest")
    if manifest.get("splits") != ["dev", "devtest"] or manifest.get("seeds") != [42, 43, 44, 45, 46]:
        raise RuntimeError(f"{language}: unexpected split or seed provenance")
    if manifest.get("error_count") != 0:
        raise RuntimeError(f"{language}: manifest records errors")

    summary = pd.read_csv(directory / "per_language_summary.csv")
    if len(summary) != 1:
        raise RuntimeError(f"{language}: summary must contain exactly one row")
    row = summary.iloc[0]
    if row["language"] != language or int(row["total_sentences"]) != 2009:
        raise RuntimeError(f"{language}: invalid language or sentence count")
    if int(row["error_count"]) != 0 or int(row["seed_repeats"]) != 5:
        raise RuntimeError(f"{language}: summary records errors or wrong seed count")
    if not all(math.isfinite(float(row[column])) for column in summary.columns if pd.api.types.is_numeric_dtype(summary[column])):
        raise RuntimeError(f"{language}: summary contains non-finite values")

    vocabulary = pd.read_csv(directory / "tokenization_vocabulary_by_string.csv")
    if len(vocabulary) != int(row["unique_normalized_strings"]):
        raise RuntimeError(f"{language}: vocabulary-row count does not reconcile")
    if vocabulary["text"].duplicated().any() or not (vocabulary["bpe_unique_signatures"] == 1).all():
        raise RuntimeError(f"{language}: vocabulary has duplicate strings or non-deterministic BPE")
    parsed_seeds = vocabulary["seed_signatures"].map(json.loads)
    if not parsed_seeds.map(lambda values: set(values) == {"42", "43", "44", "45", "46"}).all():
        raise RuntimeError(f"{language}: seed signatures are incomplete")
    if vocabulary["seed_signatures"].str.contains("<ERROR>", regex=False).any():
        raise RuntimeError(f"{language}: invalid error signature found")
    varied = (vocabulary["has_fxt_seed_variation"] == True).sum()
    if varied != int(row["strings_with_fxt_seed_variation"]):
        raise RuntimeError(f"{language}: variation count does not reconcile")
    ambiguous = pd.read_csv(directory / "top_ambiguous_strings.csv")
    expected_ambiguous = vocabulary[vocabulary["has_fxt_seed_variation"]].sort_values(
        ["fxt_unique_signatures", "count", "text"], ascending=[False, False, True]
    ).head(200)
    if list(ambiguous.get("text", [])) != list(expected_ambiguous["text"]):
        raise RuntimeError(f"{language}: ambiguous-string ranking does not reconcile")

    signatures = json.loads((directory / "vocab_signatures.json").read_text())
    if signatures.get("metadata", {}).get("seeds") != [42, 43, 44, 45, 46]:
        raise RuntimeError(f"{language}: signature JSON seed provenance is invalid")
    if len(signatures.get("vocabulary", {})) != len(vocabulary):
        raise RuntimeError(f"{language}: signature JSON vocabulary count does not reconcile")
    checkpoint_dir = directory / ".seed_checkpoints"
    if checkpoint_dir.is_dir():
        for seed in range(42, 47):
            checkpoint = json.loads((checkpoint_dir / f"seed_{seed}.json").read_text())
            if checkpoint.get("seed") != seed or checkpoint.get("batching_strategy") != "dataset_order":
                raise RuntimeError(f"{language}: checkpoint {seed} has non-canonical provenance")
            if set(checkpoint.get("signatures", {})) != set(vocabulary["text"]):
                raise RuntimeError(f"{language}: checkpoint {seed} lacks vocabulary coverage")
    return summary


def main():
    root = Path(parse_args().results_root)
    summaries = [validate_language(root / language, language) for language in LANGUAGES]
    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(root / "per_language_summary.csv", index=False)
    print(f"Validated {len(combined)} canonical languages and wrote {root / 'per_language_summary.csv'}")


if __name__ == "__main__":
    main()
