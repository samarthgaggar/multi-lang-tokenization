#!/usr/bin/env python3
"""Reproducible FLORES tokenization-consistency analysis for BPE and FlexiTokens.

Each normalized string is tokenized once by BPE and once for each requested
FlexiTokens seed. Inference errors deliberately stop the run: an error token
must never be mistaken for a valid tokenization signature.
"""
import argparse
import hashlib
import json
import os
import random
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
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


def parse_args():
    parser = argparse.ArgumentParser(description="FLORES tokenization consistency: FlexiTokens vs BPE")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--bpe_tokenizer", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--languages", default="en,es,ru,uk,hi,te")
    parser.add_argument("--splits", default="dev,devtest")
    parser.add_argument("--seed_repeats", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_examples_per_lang", type=int, default=0)
    parser.add_argument("--dataset_source", default="yash9439/flores200")
    parser.add_argument("--expected_sentences_per_lang", type=int, default=2009)
    parser.add_argument(
        "--length_bucketed",
        action="store_true",
        help="Batch normalized strings by encoded length to reduce padding; final result schema is unchanged.",
    )
    parser.add_argument(
        "--clear_mps_cache_each_batch",
        action="store_true",
        help="Release unused MPS allocations after each batch without changing the evaluation protocol.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """NFKC-normalize, trim, and collapse whitespace while preserving case."""
    return " ".join(unicodedata.normalize("NFKC", text).strip().split())


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_fxt_model(model_path: str, device: torch.device):
    model_path = Path(model_path)
    config = json.loads((model_path / "config.json").read_text())
    import inspect

    params = inspect.signature(FxTTransformerLM.__init__).parameters
    kwargs = {name: config[name] for name in params if name != "self" and name in config}
    model = FxTTransformerLM(**kwargs)
    checkpoint = torch.load(model_path / "model.pth", map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    try:
        model.load_state_dict(state, assign=True)
    except TypeError:  # compatibility with older PyTorch versions
        model.load_state_dict(state)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    return model, tokenizer, config


def language_token_id(tokenizer, config: dict, language: str) -> int:
    script_token = config.get("language_to_script", {}).get(language)
    if script_token is None:
        supported = ", ".join(sorted(config.get("language_to_script", {})))
        raise ValueError(f"Checkpoint does not support language '{language}'. Supported: {supported}")
    token_id = tokenizer.convert_tokens_to_ids(script_token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        id_to_script = {int(key): value for key, value in config.get("id_to_script", {}).items()}
        inverse = {value: key for key, value in id_to_script.items()}
        if script_token not in inverse:
            raise ValueError(f"No token id exists for checkpoint language token '{script_token}'")
        token_id = inverse[script_token]
    return int(token_id)


def signature_from_boundaries(tokenizer, input_ids, boundaries) -> str:
    """Build a signature from valid, unpadded token IDs and matching boundaries."""
    if len(input_ids) != len(boundaries):
        raise ValueError(f"Token/boundary length mismatch: {len(input_ids)} != {len(boundaries)}")
    segments, current = [], []
    for token_id, is_boundary in zip(input_ids, boundaries):
        current.append(int(token_id))
        if int(is_boundary) == 1:
            segments.append(tokenizer.decode(current, skip_special_tokens=False))
            current = []
    if current:
        segments.append(tokenizer.decode(current, skip_special_tokens=False))
    eos = tokenizer.eos_token or ""
    return "|".join(segment.replace(eos, "").strip() for segment in segments)


def fxt_signatures_for_batch(model, tokenizer, texts, language, config, device):
    if not texts:
        return []
    script_id = language_token_id(tokenizer, config, language)
    encoded = tokenizer(texts, add_special_tokens=False, padding=True, return_attention_mask=True, return_tensors="pt")
    raw_input_ids = encoded["input_ids"]
    raw_attention_mask = encoded["attention_mask"]
    model_input_ids = torch.cat(
        [torch.full((raw_input_ids.size(0), 1), script_id, dtype=raw_input_ids.dtype), raw_input_ids], dim=1
    ).to(device)
    # The model removes the language token internally, so its mask must align
    # with the unprefixed tokens rather than model_input_ids.
    model_attention_mask = raw_attention_mask.to(device)
    with torch.inference_mode():
        _, stats, _ = model({"input_ids": model_input_ids, "attention_mask": model_attention_mask}, task="tokenization2")
    hard_boundaries = stats["hard_boundaries"].detach().cpu().numpy()
    signatures = []
    for ids, mask, boundaries in zip(raw_input_ids.tolist(), raw_attention_mask.tolist(), hard_boundaries):
        valid_length = int(sum(mask))
        signatures.append(signature_from_boundaries(tokenizer, ids[:valid_length], boundaries[:valid_length]))
    return signatures


def bpe_signature(tokenizer, text: str) -> str:
    encoded = tokenizer(text, add_special_tokens=False)
    return "|".join(tokenizer.convert_ids_to_tokens(encoded["input_ids"]))


def load_language_texts(language, splits, dataset_source):
    column = FLORES_MAPPING.get(language)
    if column is None:
        raise KeyError(f"Unknown FLORES language code: {language}")
    texts, fingerprints = [], {}
    for split in splits:
        dataset = load_dataset(dataset_source, split=split)
        if column not in dataset.column_names:
            raise RuntimeError(f"Dataset {dataset_source} split {split} lacks column {column}")
        texts.extend(dataset[column])
        fingerprints[split] = getattr(dataset, "_fingerprint", None)
    return texts, fingerprints


def atomic_json_dump(payload, path: Path):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    os.replace(temporary_path, path)


def checkpoint_path(output_dir: Path, seed: int) -> Path:
    return output_dir / ".seed_checkpoints" / f"seed_{seed}.json"


def sequence_fingerprint(texts) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_seed_checkpoint(output_dir: Path, language: str, seed: int, texts, batching_strategy: str):
    path = checkpoint_path(output_dir, seed)
    if not path.is_file():
        return None
    with path.open() as handle:
        payload = json.load(handle)
    signatures = payload.get("signatures")
    if (
        payload.get("language") != language
        or payload.get("seed") != seed
        or payload.get("batching_strategy") != batching_strategy
        or payload.get("order_fingerprint") != sequence_fingerprint(texts)
        or not isinstance(signatures, dict)
    ):
        raise RuntimeError(f"Invalid checkpoint metadata in {path}")
    if set(signatures) != set(texts):
        raise RuntimeError(f"Checkpoint {path} does not match the current normalized vocabulary")
    return signatures


def save_seed_checkpoint(output_dir: Path, language: str, seed: int, signatures: dict, inference_texts, batching_strategy: str):
    path = checkpoint_path(output_dir, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump({
        "language": language,
        "seed": seed,
        "batching_strategy": batching_strategy,
        "order_fingerprint": sequence_fingerprint(inference_texts),
        "signatures": signatures,
    }, path)


def length_bucketed_texts(tokenizer, texts):
    lengths = {
        text: len(tokenizer(text, add_special_tokens=False)["input_ids"])
        for text in texts
    }
    return sorted(texts, key=lambda text: (lengths[text], text))


def analyze_language(
    model, fxt_tokenizer, bpe_tokenizer, config, language, texts, seeds, batch_size,
    device, output_dir, length_bucketed, clear_mps_cache_each_batch,
):
    normalized_counts = Counter(normalize_text(text) for text in texts)
    if "" in normalized_counts:
        raise ValueError(f"{language} contains an empty normalized string")
    normalized_texts = list(normalized_counts)
    bpe_signatures = {text: bpe_signature(bpe_tokenizer, text) for text in normalized_texts}
    inference_texts = length_bucketed_texts(fxt_tokenizer, normalized_texts) if length_bucketed else normalized_texts
    batching_strategy = "length_bucketed" if length_bucketed else "dataset_order"
    seed_signatures = {}
    for seed in seeds:
        restored = load_seed_checkpoint(output_dir, language, seed, inference_texts, batching_strategy)
        if restored is not None:
            print(f"{language}: restored checkpoint for seed={seed}", flush=True)
            seed_signatures[seed] = restored
            continue
        print(f"{language}: FlexiTokens seed={seed}", flush=True)
        set_seed(seed)
        signatures = []
        for start in range(0, len(inference_texts), batch_size):
            signatures.extend(fxt_signatures_for_batch(
                model, fxt_tokenizer, inference_texts[start:start + batch_size], language, config, device
            ))
            if clear_mps_cache_each_batch and device.type == "mps":
                torch.mps.empty_cache()
        if len(signatures) != len(inference_texts):
            raise RuntimeError(f"{language} seed {seed}: produced {len(signatures)} signatures for {len(inference_texts)} strings")
        seed_signatures[seed] = dict(zip(inference_texts, signatures))
        save_seed_checkpoint(output_dir, language, seed, seed_signatures[seed], inference_texts, batching_strategy)
        print(f"{language}: checkpointed seed={seed}", flush=True)

    records = []
    for text in normalized_texts:
        per_seed = {str(seed): seed_signatures[seed][text] for seed in seeds}
        distinct_fxt = sorted(set(per_seed.values()))
        records.append({
            "language": language,
            "text": text,
            "count": normalized_counts[text],
            "bpe_unique_signatures": 1,
            "fxt_unique_signatures": len(distinct_fxt),
            "fxt_seed_signature_count": len(distinct_fxt),
            "has_fxt_seed_variation": len(distinct_fxt) > 1,
            "most_common_bpe": bpe_signatures[text],
            "most_common_fxt": per_seed[str(seeds[0])],
            "seed_signatures": json.dumps(per_seed, ensure_ascii=False),
        })
    frame = pd.DataFrame.from_records(records).sort_values(["count", "text"], ascending=[False, True])
    if not (frame["bpe_unique_signatures"] == 1).all():
        raise RuntimeError(f"{language}: BPE signature multiplicity is not deterministic")
    return frame, normalized_counts, seed_signatures, bpe_signatures


def main():
    args = parse_args()
    languages = [language.strip() for language in args.languages.split(",") if language.strip()]
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    if len(languages) != 1:
        raise ValueError("Run exactly one language per process so each output directory is independently auditable")
    if not splits or args.seed_repeats < 1 or args.batch_size < 1:
        raise ValueError("splits, seed_repeats, and batch_size must be non-empty and positive")
    language = languages[0]
    seeds = list(range(42, 42 + args.seed_repeats))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device()
    print(f"Loading FlexiTokens model on {device}", flush=True)
    model, fxt_tokenizer, config = load_fxt_model(args.model_path, device)
    language_token_id(fxt_tokenizer, config, language)
    bpe_tokenizer = AutoTokenizer.from_pretrained(args.bpe_tokenizer, local_files_only=True)
    texts, fingerprints = load_language_texts(language, splits, args.dataset_source)
    if args.max_examples_per_lang > 0:
        texts = texts[:args.max_examples_per_lang]
    if args.expected_sentences_per_lang and args.max_examples_per_lang == 0 and len(texts) != args.expected_sentences_per_lang:
        raise RuntimeError(f"{language}: expected {args.expected_sentences_per_lang} sentences, received {len(texts)}")
    print(f"{language}: loaded {len(texts)} sentences; analyzing {len(set(map(normalize_text, texts)))} normalized strings", flush=True)
    frame, normalized_counts, seed_signatures, bpe_signatures = analyze_language(
        model, fxt_tokenizer, bpe_tokenizer, config, language, texts, seeds, args.batch_size, device,
        output_dir, args.length_bucketed, args.clear_mps_cache_each_batch,
    )
    varied_count = int(frame["has_fxt_seed_variation"].sum())
    unique_count = len(frame)
    summary = pd.DataFrame([{
        "language": language,
        "total_sentences": len(texts),
        "unique_normalized_strings": unique_count,
        "normalization_collision_count": len(texts) - unique_count,
        "normalization_collision_percent": (len(texts) - unique_count) / len(texts) * 100,
        "avg_bpe_signatures_per_string": float(frame["bpe_unique_signatures"].mean()),
        "avg_fxt_signatures_per_string": float(frame["fxt_unique_signatures"].mean()),
        "strings_with_fxt_seed_variation": varied_count,
        "percent_strings_with_fxt_seed_variation": varied_count / unique_count * 100,
        "seed_repeats": len(seeds),
        "error_count": 0,
    }])
    frame.to_csv(output_dir / "tokenization_vocabulary_by_string.csv", index=False)
    frame[frame["has_fxt_seed_variation"]].sort_values(
        ["fxt_unique_signatures", "count", "text"], ascending=[False, False, True]
    ).head(200).to_csv(output_dir / "top_ambiguous_strings.csv", index=False)
    summary.to_csv(output_dir / "per_language_summary.csv", index=False)
    with (output_dir / "vocab_signatures.json").open("w") as handle:
        json.dump({
            "metadata": {"language": language, "seeds": seeds},
            "vocabulary": {
                text: {
                    "count": normalized_counts[text],
                    "bpe_signature": bpe_signatures[text],
                    "seed_signatures": {str(seed): seed_signatures[seed][text] for seed in seeds},
                }
                for text in normalized_counts
            },
        }, handle, ensure_ascii=False, indent=2)
    manifest = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_source": args.dataset_source,
        "dataset_fingerprints": fingerprints,
        "language": language,
        "splits": splits,
        "seeds": seeds,
        "batch_size": args.batch_size,
        "max_examples_per_lang": args.max_examples_per_lang,
        "expected_sentences_per_lang": args.expected_sentences_per_lang,
        "device": str(device),
        "model_path": str(Path(args.model_path).resolve()),
        "bpe_tokenizer": str(Path(args.bpe_tokenizer).resolve()),
        "error_count": 0,
    }
    with (output_dir / "run_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Wrote validated results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
