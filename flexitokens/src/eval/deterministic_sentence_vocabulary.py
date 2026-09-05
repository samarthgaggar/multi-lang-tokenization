#!/usr/bin/env python3
"""Build deterministic FlexiTokens segment vocabularies from full FLORES sentences.

Each FLORES example is passed to the trained model as one complete sentence.
At inference, boundaries are exactly ``sigmoid(logit) > 0.5``.  In
particular, this script disables the checkpoint's Gumbel sampling path; it
does not set seeds or sample boundaries.  A second full pass verifies that the
same input sentences produce exactly the same segmentations.
"""
import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from src.model.fxt import FxTTransformerLM


FLORES_MAPPING = {
    "en": "eng_Latn", "es": "spa_Latn", "ru": "rus_Cyrl",
    "uk": "ukr_Cyrl", "hi": "hin_Deva", "te": "tel_Telu",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--languages", default="en,es,ru,uk,hi,te")
    parser.add_argument("--splits", default="dev,devtest")
    parser.add_argument("--dataset_source", default="yash9439/flores200")
    parser.add_argument("--expected_sentences_per_lang", type=int, default=2009)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--max_examples_per_lang", type=int, default=0,
                        help="Optional smoke-test limit; 0 evaluates every sentence.")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Zero-based start index for a resumable sentence chunk.")
    parser.add_argument("--examples_per_segment", type=int, default=3)
    parser.add_argument("--clear_mps_cache_each_batch", action="store_true")
    parser.add_argument("--verify_repeat", action="store_true",
                        help="Run a second complete deterministic pass and require an exact match.")
    return parser.parse_args()


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


def atomic_json_dump(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    os.replace(temporary, path)


def load_model(model_path, device):
    model_path = Path(model_path)
    config = json.loads((model_path / "config.json").read_text())
    import inspect
    params = inspect.signature(FxTTransformerLM.__init__).parameters
    model = FxTTransformerLM(**{
        name: config[name] for name in params if name != "self" and name in config
    })
    checkpoint = torch.load(model_path / "model.pth", map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    try:
        model.load_state_dict(state, assign=True)
    except TypeError:
        model.load_state_dict(state)
    model.to(device).eval()

    # The trained checkpoint uses bp_type="gumbel".  This evaluation flag
    # preserves its weights while selecting the requested deterministic rule.
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


def segments_from_boundaries(tokenizer, input_ids, boundaries):
    if len(input_ids) != len(boundaries):
        raise RuntimeError("Token/boundary length mismatch")
    segments, current = [], []
    for token_id, is_boundary in zip(input_ids, boundaries):
        current.append(int(token_id))
        if int(is_boundary) == 1:
            segments.append(tokenizer.decode(current, skip_special_tokens=False))
            current = []
    if current:
        segments.append(tokenizer.decode(current, skip_special_tokens=False))
    return segments


def tokenize_batch(model, tokenizer, texts, language, config, device):
    script_id = language_token_id(tokenizer, config, language)
    encoded = tokenizer(texts, add_special_tokens=False, padding=True,
                        return_attention_mask=True, return_tensors="pt")
    raw_ids, raw_mask = encoded["input_ids"], encoded["attention_mask"]
    model_ids = torch.cat([
        torch.full((raw_ids.size(0), 1), script_id, dtype=raw_ids.dtype), raw_ids
    ], dim=1).to(device)
    with torch.inference_mode():
        _, stats, _ = model({"input_ids": model_ids, "attention_mask": raw_mask.to(device)}, task="tokenization2")
    boundaries = stats["hard_boundaries"].detach().cpu().tolist()
    output = []
    for ids, mask, sentence_boundaries in zip(raw_ids.tolist(), raw_mask.tolist(), boundaries):
        length = sum(mask)
        output.append(segments_from_boundaries(tokenizer, ids[:length], sentence_boundaries[:length]))
    return output


def load_sentences(language, splits, dataset_source):
    column = FLORES_MAPPING[language]
    records, fingerprints = [], {}
    for split in splits:
        dataset = load_dataset(dataset_source, split=split)
        if column not in dataset.column_names:
            raise RuntimeError(f"{split} has no {column} column")
        fingerprints[split] = getattr(dataset, "_fingerprint", None)
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


def run_pass(model, tokenizer, records, language, config, device, batch_size, clear_cache, pass_name):
    outputs = []
    texts = [record["text"] for record in records]
    for start in range(0, len(texts), batch_size):
        outputs.extend(tokenize_batch(model, tokenizer, texts[start:start + batch_size], language, config, device))
        if start == 0 or (start // batch_size + 1) % 25 == 0:
            print(f"{language}: {pass_name} {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
        if clear_cache and device.type == "mps":
            torch.mps.empty_cache()
    if len(outputs) != len(records):
        raise RuntimeError("Incomplete tokenization pass")
    return outputs


def analyze_language(model, tokenizer, records, language, config, device, args, fingerprints):
    first_pass = run_pass(model, tokenizer, records, language, config, device,
                          args.batch_size, args.clear_mps_cache_each_batch, "pass 1")
    if args.verify_repeat:
        verification_pass = run_pass(model, tokenizer, records, language, config, device,
                                     args.batch_size, args.clear_mps_cache_each_batch, "pass 2")
        if first_pass != verification_pass:
            mismatches = sum(a != b for a, b in zip(first_pass, verification_pass))
            raise RuntimeError(f"{language}: deterministic verification failed for {mismatches} sentences")
        verification = "two complete passes matched exactly"
    else:
        verification = "one deterministic thresholding pass; no boundary sampling is performed"

    segment_counts = Counter()
    segment_examples = defaultdict(list)
    sentence_rows = []
    signature_counts = Counter()
    for sentence_id, (record, segments) in enumerate(zip(records, first_pass)):
        signature = json.dumps(segments, ensure_ascii=False, separators=(",", ":"))
        signature_counts[signature] += 1
        sentence_rows.append({
            "sentence_id": sentence_id, "split": record["split"], "row_index": record["row_index"],
            "text": record["text"], "segments": signature, "segment_count": len(segments),
        })
        for segment in segments:
            segment_counts[segment] += 1
            seen_sentence_ids = {example["sentence_id"] for example in segment_examples[segment]}
            if (len(segment_examples[segment]) < args.examples_per_segment
                    and sentence_id not in seen_sentence_ids):
                segment_examples[segment].append({"sentence_id": sentence_id, "text": record["text"]})

    vocabulary_rows = [{
        "segment": segment, "segment_display": repr(segment), "count": count,
        "example_occurrences": json.dumps(segment_examples[segment], ensure_ascii=False),
    } for segment, count in segment_counts.most_common()]
    sentence_frame = pd.DataFrame(sentence_rows)
    vocabulary_frame = pd.DataFrame(vocabulary_rows)
    summary = {
        "language": language,
        "sentences": len(records),
        "unique_sentence_segmentations": len(signature_counts),
        "unique_output_segments_vocabulary_size": len(segment_counts),
        "total_output_segment_occurrences": sum(segment_counts.values()),
        "mean_segments_per_sentence": sum(segment_counts.values()) / len(records),
        "deterministic_verification": verification,
        "dataset_fingerprints": fingerprints,
        "input_fingerprint": records_fingerprint(records),
    }
    return sentence_frame, vocabulary_frame, summary


def main():
    args = parse_args()
    languages = [item.strip() for item in args.languages.split(",") if item.strip()]
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    unknown = set(languages) - set(FLORES_MAPPING)
    if unknown:
        raise ValueError(f"Unknown languages: {sorted(unknown)}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    model, tokenizer, config = load_model(args.model_path, device)
    all_summaries = []
    for language in languages:
        records, fingerprints = load_sentences(language, splits, args.dataset_source)
        if args.expected_sentences_per_lang and not args.max_examples_per_lang and len(records) != args.expected_sentences_per_lang:
            raise RuntimeError(f"{language}: expected {args.expected_sentences_per_lang} sentences, found {len(records)}")
        if args.max_examples_per_lang:
            records = records[args.start_index:args.start_index + args.max_examples_per_lang]
        elif args.start_index:
            records = records[args.start_index:]
        mode = "plus an exact repeat check" if args.verify_repeat else "with deterministic thresholding"
        print(f"{language}: full-sentence tokenization {mode} ({len(records)} sentences)", flush=True)
        sentences, vocabulary, summary = analyze_language(
            model, tokenizer, records, language, config, device, args, fingerprints
        )
        sentences.to_csv(output_dir / f"{language}_sentence_tokenizations.csv", index=False)
        vocabulary.to_csv(output_dir / f"{language}_segment_vocabulary.csv", index=False)
        atomic_json_dump(summary, output_dir / f"{language}_summary.json")
        all_summaries.append(summary)
        print(f"{language}: {summary['unique_output_segments_vocabulary_size']} unique output segments", flush=True)
    pd.DataFrame(all_summaries).to_csv(output_dir / "summary.csv", index=False)
    atomic_json_dump({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Each original FLORES sentence is tokenized as a complete input. The vocabulary is the counted set of unique decoded output segments across those sentence tokenizations.",
        "boundary_rule": "sigmoid(boundary_logit) > 0.5",
        "randomness": "none; Gumbel sampling is explicitly disabled for this evaluation",
        "verification": "A second complete pass is performed only when --verify_repeat is requested; boundary decisions are deterministic in either mode.",
        "checkpoint_config_boundaries_type": config.get("boundaries_type"),
        "languages": languages, "splits": splits, "device": str(device),
    }, output_dir / "run_metadata.json")


if __name__ == "__main__":
    main()
