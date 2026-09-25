#!/usr/bin/env python3
"""Analyze context-dependent FlexiTokens segmentations of FLORES words.

FlexiTokens always receives each complete FLORES sentence.  The word-level
analysis happens afterwards: GPT-2's byte-level pre-tokenizer supplies spans,
and deterministic FlexiTokens boundaries are assigned to those spans.

The evaluator is deliberately resumable.  It writes all artifacts for one
language only after that language completes, and it will refuse to overwrite a
completed language unless --overwrite_language is passed explicitly.
"""
import argparse
import hashlib
import json
import os
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from tokenizers.pre_tokenizers import ByteLevel
from transformers import AutoTokenizer

from src.model.fxt import FxTTransformerLM


FLORES_MAPPING = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "ru": "rus_Cyrl",
    "uk": "ukr_Cyrl",
    "hi": "hin_Deva",
    "te": "tel_Telu",
}
LANGUAGES = tuple(FLORES_MAPPING)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--languages", default="en")
    parser.add_argument("--splits", default="dev,devtest")
    parser.add_argument("--dataset_source", default="yash9439/flores200")
    parser.add_argument("--expected_sentences_per_lang", type=int, default=2009)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--max_examples_per_lang", type=int, default=0,
                        help="Optional smoke-test limit; 0 processes every sentence.")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Zero-based start index for an explicitly requested chunk.")
    parser.add_argument("--verify_repeat_samples", type=int, default=24,
                        help="Repeat this many inputs per language and require an exact match.")
    parser.add_argument("--top_words", type=int, default=20)
    parser.add_argument("--overwrite_language", action="store_true",
                        help="Replace completed output for a requested language.")
    return parser.parse_args()


def atomic_json_dump(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def atomic_csv_dump(frame, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def choose_device(requested):
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if requested == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_path, device):
    model_path = Path(model_path)
    config = json.loads((model_path / "config.json").read_text())
    import inspect

    params = inspect.signature(FxTTransformerLM.__init__).parameters
    model = FxTTransformerLM(**{
        name: config[name] for name in params if name != "self" and name in config
    })
    checkpoint = torch.load(model_path / "model.pth", map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    try:
        model.load_state_dict(state, assign=True)
    except TypeError:
        model.load_state_dict(state)
    model.to(device).eval()

    predictors = list(model.script_to_bp_layers.values())
    if len(predictors) != 1:
        raise RuntimeError(f"Expected one shared boundary predictor, found {len(predictors)}")
    predictor = predictors[0]
    predictor.force_deterministic_threshold = True
    if predictor.threshold != 0.5:
        raise RuntimeError(f"Expected boundary threshold 0.5, found {predictor.threshold}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    return model, tokenizer, config


def language_token_id(tokenizer, config, language):
    script_token = config["language_to_script"][language]
    token_id = tokenizer.convert_tokens_to_ids(script_token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        inverse = {value: int(key) for key, value in config["id_to_script"].items()}
        token_id = inverse[script_token]
    return int(token_id)


def cached_arrow_path(dataset_source, split):
    """Return a reusable local FLORES Arrow file when one is already cached.

    Loading the Arrow file directly is read-only. It avoids an unnecessary Hub
    request and avoids creating a cache lock outside this repository.
    """
    dataset_directory = dataset_source.replace("/", "___")
    cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / dataset_directory
    candidates = sorted(cache_root.glob(f"**/flores200-{split}.arrow"))
    return candidates[0] if candidates else None


def load_split(dataset_source, split):
    cached_path = cached_arrow_path(dataset_source, split)
    if cached_path is not None:
        return Dataset.from_file(str(cached_path)), str(cached_path)
    return load_dataset(dataset_source, split=split), None


def load_sentences(language, splits, dataset_source):
    column = FLORES_MAPPING[language]
    records, fingerprints = [], {}
    for split in splits:
        dataset, cached_path = load_split(dataset_source, split)
        if column not in dataset.column_names:
            raise RuntimeError(f"{split} has no {column} column")
        fingerprints[split] = {
            "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
            "cached_arrow_path": cached_path,
        }
        for row_index, text in enumerate(dataset[column]):
            if not isinstance(text, str) or not text:
                raise RuntimeError(f"{language}/{split}/{row_index}: empty or non-string sentence")
            records.append({"split": split, "row_index": row_index, "text": text})
    return records, fingerprints


def records_fingerprint(records):
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["split"].encode())
        digest.update(str(record["row_index"]).encode())
        digest.update(record["text"].encode())
        digest.update(b"\0")
    return digest.hexdigest()


def tokenize_batch(model, tokenizer, texts, language, config, device):
    """Return input IDs, their source-byte ends, and deterministic boundaries."""
    script_id = language_token_id(tokenizer, config, language)
    encoded = tokenizer(texts, add_special_tokens=False, padding=True,
                        return_attention_mask=True, return_tensors="pt")
    raw_ids, raw_mask = encoded["input_ids"], encoded["attention_mask"]
    model_ids = torch.cat([
        torch.full((raw_ids.size(0), 1), script_id, dtype=raw_ids.dtype), raw_ids
    ], dim=1).to(device)
    with torch.inference_mode():
        _, stats, _ = model({"input_ids": model_ids, "attention_mask": raw_mask.to(device)},
                            task="tokenization2")
    hard_boundaries = stats["hard_boundaries"].detach().cpu().tolist()
    output = []
    for text, ids, mask, boundaries in zip(texts, raw_ids.tolist(), raw_mask.tolist(), hard_boundaries):
        length = int(sum(mask))
        ids, boundaries = ids[:length], boundaries[:length]
        encoded_bytes = text.encode("utf-8")
        token_byte_ends, reconstructed = [], bytearray()
        for token_id in ids:
            # ByT5 uses IDs 3..258 for individual bytes. The checkpoint also
            # has language/control AddedTokens (for example ID 259 -> "en"),
            # which can appear inside ordinary text. Preserve the tokenizer's
            # actual input sequence instead of pretending every ID is a byte.
            if 3 <= token_id <= 258:
                token_bytes = bytes([token_id - 3])
            else:
                token = tokenizer.convert_ids_to_tokens(int(token_id))
                if not isinstance(token, str):
                    raise RuntimeError(f"Cannot recover source bytes for token ID {token_id}")
                token_bytes = token.encode("utf-8")
            reconstructed.extend(token_bytes)
            token_byte_ends.append(len(reconstructed))
        if bytes(reconstructed) != encoded_bytes:
            raise RuntimeError(
                "The checkpoint tokenizer input IDs cannot be aligned to the original UTF-8 text: "
                f"ids={ids!r}, reconstructed={bytes(reconstructed)!r}, expected={encoded_bytes!r}"
            )
        if len(boundaries) != len(ids):
            raise RuntimeError("Token/boundary length mismatch")
        output.append({
            "input_ids": ids,
            "token_byte_ends": token_byte_ends,
            "hard_boundaries": [int(value) for value in boundaries],
        })
    return output


def run_pass(model, tokenizer, records, language, config, device, batch_size, label):
    outputs = []
    texts = [record["text"] for record in records]
    for start in range(0, len(texts), batch_size):
        outputs.extend(tokenize_batch(model, tokenizer, texts[start:start + batch_size], language, config, device))
        if start == 0 or (start // batch_size + 1) % 25 == 0:
            print(f"{language}: {label} {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
        if device.type == "mps":
            torch.mps.empty_cache()
    if len(outputs) != len(records):
        raise RuntimeError("Incomplete tokenization pass")
    return outputs


def is_word_character(character):
    """Unicode letters, marks, and numbers form the word-like units we analyze."""
    category = unicodedata.category(character)
    return category[0] in {"L", "M", "N"}


def word_spans_from_gpt2_pretokenizer(text, pretokenizer):
    """Derive word-like spans from GPT-2 pre-token offsets.

    GPT-2's ByteLevel pre-tokenizer may separate combining marks in Indic text.
    Contiguous letter/mark/number pieces are therefore rejoined into one Unicode
    word span. Punctuation-only units are deliberately omitted from word plots.
    """
    candidate_spans = []
    for _, (piece_start, piece_end) in pretokenizer.pre_tokenize_str(text):
        index = piece_start
        while index < piece_end:
            while index < piece_end and not is_word_character(text[index]):
                index += 1
            start = index
            while index < piece_end and is_word_character(text[index]):
                index += 1
            if start < index:
                candidate_spans.append((start, index))
    # ByteLevel can emit separate adjacent pre-tokens for Unicode combining
    # marks. Merge only directly adjoining word-like pieces: a comma, hyphen,
    # apostrophe, or whitespace leaves a gap and therefore remains separate.
    spans = []
    for start, end in candidate_spans:
        if spans and spans[-1][1] == start:
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))
    # The pre-tokenizer produces non-overlapping offsets. This catches any
    # unexpected offset behavior before it can corrupt word counts.
    if any(end > next_start for (_, end), (next_start, _) in zip(spans, spans[1:])):
        raise RuntimeError(f"Overlapping GPT-2-derived word spans in {text!r}")
    return spans


def character_to_byte_offsets(text):
    offsets = [0]
    total = 0
    for character in text:
        total += len(character.encode("utf-8"))
        offsets.append(total)
    return offsets


def word_segmentation(text, char_start, char_end, token_byte_ends, hard_boundaries):
    """Describe exact byte boundaries within one Unicode word span."""
    byte_offsets = character_to_byte_offsets(text)
    byte_start, byte_end = byte_offsets[char_start], byte_offsets[char_end]
    sentence_bytes = text.encode("utf-8")
    word_bytes = sentence_bytes[byte_start:byte_end]
    # A boundary at byte_start belongs to the preceding space/punctuation, not
    # to this word. Only boundaries strictly inside the word change its split.
    model_cuts = [byte_end_position for byte_end_position, value in zip(token_byte_ends, hard_boundaries)
                  if value and byte_start < byte_end_position < byte_end]
    relative_cuts = tuple(position - byte_start for position in model_cuts)
    endpoints = list(relative_cuts) + [len(word_bytes)]
    pieces = []
    previous = 0
    for endpoint in endpoints:
        pieces.append(word_bytes[previous:endpoint])
        previous = endpoint
    if b"".join(pieces) != word_bytes:
        raise RuntimeError("Word segmentation does not reconstruct the original UTF-8 bytes")

    character_boundaries = set()
    running = 0
    for character in text[char_start:char_end]:
        running += len(character.encode("utf-8"))
        character_boundaries.add(running)
    has_intra_character_boundary = any(cut not in character_boundaries for cut in relative_cuts)
    surface = text[char_start:char_end]
    if has_intra_character_boundary:
        display = f"{surface} [byte cuts: {','.join(map(str, relative_cuts)) or 'none'}]"
        piece_display = json.dumps([piece.hex() for piece in pieces])
    else:
        decoded_pieces = [piece.decode("utf-8") for piece in pieces]
        display = "|".join(decoded_pieces)
        piece_display = json.dumps(decoded_pieces, ensure_ascii=False)
    return {
        "surface_word": surface,
        "char_start": char_start,
        "char_end": char_end,
        "byte_start": byte_start,
        "byte_end": byte_end,
        "boundary_signature": ",".join(map(str, relative_cuts)) or "none",
        "segmentation_display": display,
        "piece_bytes_or_text": piece_display,
        "piece_count": len(pieces),
        "has_intra_character_boundary": has_intra_character_boundary,
    }


def occurrence_rows(language, records, outputs, pretokenizer):
    rows, sentence_rows = [], []
    for sentence_id, (record, output) in enumerate(zip(records, outputs)):
        text = record["text"]
        spans = word_spans_from_gpt2_pretokenizer(text, pretokenizer)
        sentence_rows.append({
            "language": language,
            "sentence_id": sentence_id,
            "split": record["split"],
            "row_index": record["row_index"],
            "text": text,
            "utf8_byte_length": len(text.encode("utf-8")),
            "input_ids": json.dumps(output["input_ids"]),
            "input_token_byte_ends": json.dumps(output["token_byte_ends"]),
            "hard_boundaries": json.dumps(output["hard_boundaries"]),
            "gpt2_word_span_count": len(spans),
        })
        for char_start, char_end in spans:
            segmentation = word_segmentation(
                text, char_start, char_end, output["token_byte_ends"], output["hard_boundaries"]
            )
            rows.append({
                "language": language,
                "sentence_id": sentence_id,
                "split": record["split"],
                "row_index": record["row_index"],
                "sentence": text,
                "surface_word": segmentation["surface_word"],
                "lowercase_word": segmentation["surface_word"].lower(),
                "char_start": segmentation["char_start"],
                "char_end": segmentation["char_end"],
                "byte_start": segmentation["byte_start"],
                "byte_end": segmentation["byte_end"],
                "boundary_signature": segmentation["boundary_signature"],
                "segmentation_display": segmentation["segmentation_display"],
                "piece_bytes_or_text": segmentation["piece_bytes_or_text"],
                "piece_count": segmentation["piece_count"],
                "has_intra_character_boundary": segmentation["has_intra_character_boundary"],
            })
    return pd.DataFrame(rows), pd.DataFrame(sentence_rows)


def aggregate_words(occurrences, examples_per_pattern=3):
    word_rows, pattern_rows = [], []
    for word, group in occurrences.groupby("lowercase_word", sort=False):
        pattern_counts = group["boundary_signature"].value_counts(sort=True)
        total = len(group)
        dominant_signature, dominant_count = pattern_counts.index[0], int(pattern_counts.iloc[0])
        dominant_row = group[group["boundary_signature"] == dominant_signature].iloc[0]
        word_rows.append({
            "lowercase_word": word,
            "occurrence_count": total,
            "unique_tokenizations": len(pattern_counts),
            "dominant_boundary_signature": dominant_signature,
            "dominant_segmentation_display": dominant_row["segmentation_display"],
            "dominant_count": dominant_count,
            "dominant_percentage": 100 * dominant_count / total,
            "variation_score": 1 - dominant_count / total,
            "min_piece_count": int(group["piece_count"].min()),
            "mean_piece_count": float(group["piece_count"].mean()),
            "max_piece_count": int(group["piece_count"].max()),
            "surface_forms": json.dumps(sorted(group["surface_word"].unique()), ensure_ascii=False),
            "has_intra_character_boundary": bool(group["has_intra_character_boundary"].any()),
        })
        for signature, count in pattern_counts.items():
            pattern_group = group[group["boundary_signature"] == signature]
            representative = pattern_group.iloc[0]
            examples = pattern_group[["sentence_id", "sentence", "surface_word"]].head(examples_per_pattern)
            pattern_rows.append({
                "lowercase_word": word,
                "boundary_signature": signature,
                "segmentation_display": representative["segmentation_display"],
                "piece_count": int(representative["piece_count"]),
                "count": int(count),
                "percentage": 100 * int(count) / total,
                "has_intra_character_boundary": bool(pattern_group["has_intra_character_boundary"].any()),
                "example_occurrences": examples.to_json(orient="records", force_ascii=False),
            })
    return pd.DataFrame(word_rows), pd.DataFrame(pattern_rows)


def language_summary(language, occurrences, words, patterns, records, fingerprints, verification):
    repeated = words[words["occurrence_count"] >= 2]
    repeated_distribution = repeated["unique_tokenizations"].value_counts().sort_index()
    return {
        "language": language,
        "sentences": len(records),
        "word_occurrences": len(occurrences),
        "unique_lowercase_words": len(words),
        "repeated_words": len(repeated),
        "repeated_words_with_variation": int((repeated["unique_tokenizations"] > 1).sum()),
        "percent_repeated_words_with_variation": (
            100 * float((repeated["unique_tokenizations"] > 1).mean()) if len(repeated) else 0.0
        ),
        "intra_character_boundary_occurrences": int(occurrences["has_intra_character_boundary"].sum()),
        "intra_character_boundary_patterns": int(patterns["has_intra_character_boundary"].sum()),
        "unique_word_tokenization_patterns": len(patterns),
        "repeated_word_variant_distribution": {
            str(int(variants)): int(count) for variants, count in repeated_distribution.items()
        },
        "boundary_rule": "sigmoid(boundary_logit) > 0.5",
        "randomness": "none; Gumbel sampling disabled",
        "verification": verification,
        "dataset_fingerprints": fingerprints,
        "input_fingerprint": records_fingerprint(records),
    }


def save_plots(language_dir, language, words, patterns, top_words):
    repeated = words[words["occurrence_count"] >= 2].copy()
    if repeated.empty:
        return

    variant_counts = repeated["unique_tokenizations"].clip(upper=6).value_counts().sort_index()
    labels = [str(value) if value < 6 else "6+" for value in variant_counts.index]
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(labels, variant_counts.values, color="#4c78a8")
    axis.set_xlabel("unique tokenizations per repeated word")
    axis.set_ylabel("number of lowercase word groups")
    axis.set_title(f"{language}: contextual FlexiTokens variation")
    fig.tight_layout()
    fig.savefig(language_dir / "unique_tokenizations_per_word.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.5, 5))
    axis.scatter(repeated["occurrence_count"], repeated["unique_tokenizations"], alpha=0.55, s=20)
    axis.set_xscale("log")
    axis.set_xlabel("word occurrences (log scale)")
    axis.set_ylabel("unique tokenizations")
    axis.set_title(f"{language}: frequency versus tokenization variation")
    fig.tight_layout()
    fig.savefig(language_dir / "frequency_vs_variation.png", dpi=180)
    plt.close(fig)

    candidates = repeated[repeated["occurrence_count"] >= 5].sort_values(
        ["unique_tokenizations", "variation_score", "occurrence_count"], ascending=False
    ).head(top_words)
    if not candidates.empty:
        fig, axis = plt.subplots(figsize=(10, max(4.5, 0.35 * len(candidates))))
        labels = [f"{word} ({count})" for word, count in zip(candidates["lowercase_word"], candidates["occurrence_count"])]
        axis.barh(labels[::-1], candidates["unique_tokenizations"].iloc[::-1], color="#f58518")
        axis.set_xlabel("unique tokenizations")
        axis.set_title(f"{language}: most variable words with at least 5 occurrences")
        fig.tight_layout()
        fig.savefig(language_dir / "top_variable_words.png", dpi=180)
        plt.close(fig)

        selected = candidates.iloc[0]
        selected_patterns = patterns[patterns["lowercase_word"] == selected["lowercase_word"]].sort_values("count")
        fig, axis = plt.subplots(figsize=(9, max(3.5, 0.45 * len(selected_patterns))))
        axis.barh(selected_patterns["segmentation_display"], selected_patterns["count"], color="#54a24b")
        axis.set_xlabel("occurrences")
        axis.set_title(f"{language}: observed patterns for {selected['lowercase_word']!r}")
        fig.tight_layout()
        fig.savefig(language_dir / "top_word_pattern_frequencies.png", dpi=180)
        plt.close(fig)


def manual_inspection_template(words, patterns, occurrences, top_words):
    candidates = words[(words["occurrence_count"] >= 5) & (words["unique_tokenizations"] > 1)].sort_values(
        ["unique_tokenizations", "variation_score", "occurrence_count"], ascending=False
    ).head(top_words)
    rows = []
    for _, word_row in candidates.iterrows():
        word_patterns = patterns[patterns["lowercase_word"] == word_row["lowercase_word"]]
        for _, pattern_row in word_patterns.iterrows():
            example = occurrences[
                (occurrences["lowercase_word"] == word_row["lowercase_word"])
                & (occurrences["boundary_signature"] == pattern_row["boundary_signature"])
            ].iloc[0]
            rows.append({
                "lowercase_word": word_row["lowercase_word"],
                "occurrence_count": word_row["occurrence_count"],
                "unique_tokenizations": word_row["unique_tokenizations"],
                "boundary_signature": pattern_row["boundary_signature"],
                "segmentation_display": pattern_row["segmentation_display"],
                "pattern_count": pattern_row["count"],
                "pattern_percentage": pattern_row["percentage"],
                "sentence": example["sentence"],
                "manual_annotation": "unreviewed",
                "manual_notes": "",
            })
    return pd.DataFrame(rows)


def analyze_language(model, tokenizer, language, records, fingerprints, config, device, args, pretokenizer):
    outputs = run_pass(model, tokenizer, records, language, config, device, args.batch_size, "tokenizing")
    verification = "one deterministic thresholding pass; no boundary sampling is performed"
    repeat_count = min(args.verify_repeat_samples, len(records))
    if repeat_count:
        repeated = run_pass(model, tokenizer, records[:repeat_count], language, config, device,
                            args.batch_size, "repeat-check")
        if repeated != outputs[:repeat_count]:
            raise RuntimeError(f"{language}: deterministic repeat check failed for {repeat_count} sentences")
        verification = f"{repeat_count} representative sentences reproduced exactly"
    occurrences, sentence_boundaries = occurrence_rows(language, records, outputs, pretokenizer)
    if occurrences.empty:
        raise RuntimeError(f"{language}: no GPT-2-derived word occurrences")
    words, patterns = aggregate_words(occurrences)
    summary = language_summary(language, occurrences, words, patterns, records, fingerprints, verification)
    template = manual_inspection_template(words, patterns, occurrences, args.top_words)
    return occurrences, sentence_boundaries, words, patterns, template, summary


def update_progress(output_dir, language, status, detail=None):
    path = output_dir / "progress.json"
    payload = json.loads(path.read_text()) if path.exists() else {"languages": {}}
    payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["languages"][language] = {"status": status, "detail": detail}
    atomic_json_dump(payload, path)


def main():
    args = parse_args()
    languages = [item.strip() for item in args.languages.split(",") if item.strip()]
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    unknown = set(languages) - set(LANGUAGES)
    if unknown:
        raise ValueError(f"Unknown languages: {sorted(unknown)}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    model, tokenizer, config = load_model(args.model_path, device)
    pretokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    atomic_json_dump({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Full FLORES sentences are model inputs. GPT-2 ByteLevel pre-token offsets locate word-like spans; contiguous Unicode letters, marks, and numbers form each analyzed word.",
        "boundary_rule": "sigmoid(boundary_logit) > 0.5",
        "randomness": "none; Gumbel sampling explicitly disabled",
        "lowercase_grouping": True,
        "space_rule": "word-level segments are restricted to one non-whitespace word span",
        "languages_requested": languages,
        "splits": splits,
        "device": str(device),
        "checkpoint_boundaries_type": config.get("boundaries_type"),
    }, output_dir / "run_metadata.json")

    for language in languages:
        language_dir = output_dir / language
        summary_path = language_dir / "summary.json"
        if summary_path.exists() and not args.overwrite_language:
            raise FileExistsError(
                f"{language}: completed output already exists at {language_dir}; "
                "use --overwrite_language only after explicitly deciding to replace it"
            )
        language_dir.mkdir(parents=True, exist_ok=True)
        update_progress(output_dir, language, "running")
        records, fingerprints = load_sentences(language, splits, args.dataset_source)
        if args.expected_sentences_per_lang and not args.max_examples_per_lang and len(records) != args.expected_sentences_per_lang:
            raise RuntimeError(f"{language}: expected {args.expected_sentences_per_lang} sentences, found {len(records)}")
        if args.max_examples_per_lang:
            records = records[args.start_index:args.start_index + args.max_examples_per_lang]
        elif args.start_index:
            records = records[args.start_index:]
        print(f"{language}: starting {len(records)} complete sentences", flush=True)
        try:
            occurrences, sentence_boundaries, words, patterns, template, summary = analyze_language(
                model, tokenizer, language, records, fingerprints, config, device, args, pretokenizer
            )
            atomic_csv_dump(occurrences, language_dir / "word_occurrences.csv")
            atomic_csv_dump(sentence_boundaries, language_dir / "sentence_boundaries.csv")
            atomic_csv_dump(words, language_dir / "word_summary.csv")
            atomic_csv_dump(patterns, language_dir / "tokenization_patterns.csv")
            atomic_csv_dump(template, language_dir / "manual_inspection_template.csv")
            save_plots(language_dir, language, words, patterns, args.top_words)
            atomic_json_dump(summary, summary_path)
            update_progress(output_dir, language, "complete", {
                "sentences": summary["sentences"],
                "word_occurrences": summary["word_occurrences"],
                "repeated_words": summary["repeated_words"],
            })
            print(f"{language}: complete; {summary['word_occurrences']} word occurrences", flush=True)
        except Exception as error:
            update_progress(output_dir, language, "failed", str(error))
            raise


if __name__ == "__main__":
    main()
