#!/usr/bin/env python3
"""
Corpus-wide tokenization consistency analysis for FLORES and FlexiTokens/BPE.

Outputs JSON/CSV summaries reporting:
- vocabulary size (normalized-string -> signatures)
- per-language signature multiplicity and inconsistency rates
- top ambiguous strings for manual inspection

This script reuses the model and tokenizer loading conventions in the repo.
"""
import argparse
import json
import math
import os
from collections import defaultdict, Counter
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer

from src.model.fxt import FxTTransformerLM


FLORES_MAPPING = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "uk": "ukr_Cyrl",
    "ru": "rus_Cyrl",
    "be": "bel_Cyrl",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "te": "tel_Telu",
    "ur": "urd_Arab",
}


def parse_args():
    p = argparse.ArgumentParser(description="Tokenization consistency analysis: FlexiTokens vs BPE on FLORES")
    p.add_argument("--model_path", type=str, required=True, help="Path to FlexiTokens checkpoint directory (contains config.json and model.pth)")
    p.add_argument("--bpe_tokenizer", type=str, default="data/bpe_tokenizer_50000", help="Path to BPE tokenizer folder")
    p.add_argument("--output_dir", type=str, default="results/tokenization_consistency", help="Output directory for analysis artifacts")
    p.add_argument("--languages", type=str, default="en,es,ru,uk,hi,te", help="Comma-separated 2-letter languages to analyze (subset of FLORES mapping)")
    p.add_argument("--splits", type=str, default="dev,devtest", help="Comma-separated dataset splits to use")
    p.add_argument("--seed_repeats", type=int, default=3, help="Number of different seeds to run for stochastic consistency checks")
    p.add_argument("--max_examples_per_lang", type=int, default=0, help="If >0, limit per-language examples (useful for quick runs)")
    return p.parse_args()


def normalize_text(s: str) -> str:
    # Unicode normalize (NFKC), trim and collapse internal whitespace. Preserve case.
    s = unicodedata.normalize("NFKC", s)
    s = " ".join(s.strip().split())
    return s


def load_fxt_model(model_path: str, device="cpu"):
    model_path = Path(model_path)
    config = json.loads((model_path / "config.json").read_text())
    # construct model with config fields required by constructor
    # mirror notebook loader: select constructor args present in config
    import inspect

    parameters = inspect.signature(FxTTransformerLM.__init__).parameters
    kwargs = {name: config[name] for name in parameters if name != "self" and name in config}
    model = FxTTransformerLM(**kwargs)

    checkpoint = torch.load(model_path / "model.pth", map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    try:
        model.load_state_dict(state, assign=True)
    except TypeError:
        model.load_state_dict(state)
    # ensure device is a torch.device
    device_obj = torch.device(device)
    model.to(device_obj).eval()

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)

    return model, tokenizer, config


def fxt_tokenization_signature(model, tokenizer, text: str, lang: str, config: dict, seed: int = 42, device="cpu"):
    # prepare input: tokenizer returns byte-level ids; prepend language script token id
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    # map language to script token id using config["language_to_script"] and tokenizer
    script_token = config["language_to_script"].get(lang)
    # try resolve token id from tokenizer, else search id_to_script map
    token_id = tokenizer.convert_tokens_to_ids(script_token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        # fallback: find numeric key in config id_to_script
        id_to_script = {int(k): v for k, v in config.get("id_to_script", {}).items()}
        inv = {v: k for k, v in id_to_script.items()}
        token_id = int(inv[script_token])

    # batch with leading script token
    batch = {
        "input_ids": torch.tensor([[token_id, *input_ids]], device=device, dtype=torch.long),
        "attention_mask": torch.ones((1, len(input_ids) + 1), device=device, dtype=torch.long),
    }
    torch.manual_seed(seed)
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    with torch.inference_mode():
        _, stats, _ = model(batch, task="tokenization2")

    hard = stats["hard_boundaries"][0, : len(input_ids)].detach().cpu().numpy()
    # collect segments
    segments = []
    current = []
    for token_id, is_boundary in zip(input_ids, hard):
        current.append(token_id)
        if int(is_boundary) == 1:
            segments.append(tokenizer.decode(current))
            current = []
    if current:
        segments.append(tokenizer.decode(current))

    # canonical signature: join with | and also provide token id sequences
    signature = "|".join(s.replace(tokenizer.eos_token or "", "").strip() for s in segments)
    token_ids_signature = tuple(tuple([int(x) for x in seg_ids]) if isinstance(seg_ids, (list, tuple)) else tuple() for seg_ids in [])
    return signature


def bpe_tokenization_signature(bpe_tokenizer, text: str):
    enc = bpe_tokenizer(text, add_special_tokens=False)
    pieces = bpe_tokenizer.convert_ids_to_tokens(enc["input_ids"])
    signature = "|".join(pieces)
    return signature


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Prefer CUDA, else Apple MPS if available, else CPU
    if torch.cuda.is_available():
        device = "cuda"
    else:
        try:
            mps_available = getattr(torch.backends, "mps").is_available()
        except Exception:
            mps_available = False
        if mps_available:
            device = "mps"
        else:
            device = "cpu"
    # convert to torch.device for tensor creation and .to()
    device = torch.device(device)

    languages = args.languages.split(",")
    splits = args.splits.split(",")

    print(f"Loading FlexiTokens model from {args.model_path} on {device}")
    model, fxt_tokenizer, config = load_fxt_model(args.model_path, device=device)
    model.to(device)

    print(f"Loading BPE tokenizer from {args.bpe_tokenizer}")
    bpe_tokenizer = AutoTokenizer.from_pretrained(args.bpe_tokenizer)

    # corpora loader
    corpus = {lang: [] for lang in languages}
    for lang in languages:
        hf_lang = FLORES_MAPPING.get(lang, None)
        if hf_lang is None:
            raise KeyError(f"Unknown FLORES language code: {lang}")
        texts = []
        for split in splits:
            # Try the official facebook/flores (may be gated). If unavailable, fall back
            # to the public consolidated mirror 'yash9439/flores200' which exposes
            # aligned columns like 'eng_Latn', 'spa_Latn', etc.
            try:
                ds = load_dataset("facebook/flores", hf_lang, split=split, trust_remote_code=True)
                texts.extend(list(ds["sentence"]))
            except Exception:
                # fallback: load the consolidated mirror and read the language column
                mirror = load_dataset("yash9439/flores200", split=split)
                col = hf_lang
                if col not in mirror.column_names:
                    raise RuntimeError(f"Fallback dataset missing expected column {col}")
                texts.extend(list(mirror[col]))
        if args.max_examples_per_lang > 0:
            texts = texts[: args.max_examples_per_lang]
        corpus[lang] = texts
        print(f"Loaded {len(texts)} sentences for {lang}")

    # Main maps
    # normalized_string -> {"count": int, "langs": Counter, "bpe_signatures": Counter, "fxt_signatures": Counter}
    vocab = defaultdict(lambda: {"count": 0, "langs": Counter(), "bpe_signatures": Counter(), "fxt_signatures": Counter()})

    # First pass: collect signatures (single-seed deterministic pass for both tokenizers)
    for lang, texts in corpus.items():
        for text in texts:
            norm = normalize_text(text)
            vocab[norm]["count"] += 1
            vocab[norm]["langs"][lang] += 1
            # BPE signature
            try:
                bpe_sig = bpe_tokenization_signature(bpe_tokenizer, norm)
            except Exception:
                bpe_sig = "<ERROR>"
            vocab[norm]["bpe_signatures"][bpe_sig] += 1
            # FlexiTokens signature (single seed)
            try:
                fxt_sig = fxt_tokenization_signature(model, fxt_tokenizer, norm, lang, config, seed=42, device=device)
            except Exception:
                fxt_sig = "<ERROR>"
            vocab[norm]["fxt_signatures"][fxt_sig] += 1

    # Consistency/stochasticity checks for FlexiTokens across seeds
    seed_stats = {}
    seeds = [42 + i for i in range(args.seed_repeats)]
    for seed in seeds:
        print(f"Running stochastic pass seed={seed}")
        for lang, texts in corpus.items():
            for text in texts:
                norm = normalize_text(text)
                try:
                    fxt_sig = fxt_tokenization_signature(model, fxt_tokenizer, norm, lang, config, seed=seed, device=device)
                except Exception:
                    fxt_sig = "<ERROR>"
                key = f"seed_{seed}"
                seed_stats.setdefault(key, Counter())
                seed_stats[key][(norm, fxt_sig)] += 1

    # Compute per-normalized-string disagreement rate across seeds
    fxt_disagreement = {}
    for norm, entry in vocab.items():
        # collect the most frequent signature per seed
        sigs_per_seed = []
        for seed in seeds:
            key = f"seed_{seed}"
            # extract signatures for this norm from seed_stats
            matches = [sig for (n, sig), c in seed_stats[key].items() if n == norm]
            sigs_per_seed.append(matches[0] if matches else None)
        unique = set(sigs_per_seed)
        unique.discard(None)
        fxt_disagreement[norm] = {"n_signatures_across_seeds": len(unique), "seed_signatures": sigs_per_seed}

    # Summaries
    records = []
    for norm, entry in vocab.items():
        num_bpe = len(entry["bpe_signatures"])
        num_fxt = len(entry["fxt_signatures"])
        disagreement = fxt_disagreement.get(norm, {"n_signatures_across_seeds": 0})["n_signatures_across_seeds"]
        most_common_bpe = entry["bpe_signatures"].most_common(1)[0][0] if entry["bpe_signatures"] else ""
        most_common_fxt = entry["fxt_signatures"].most_common(1)[0][0] if entry["fxt_signatures"] else ""
        records.append(
            {
                "text": norm,
                "count": entry["count"],
                "langs": dict(entry["langs"]),
                "bpe_unique_signatures": num_bpe,
                "fxt_unique_signatures": num_fxt,
                "most_common_bpe": most_common_bpe,
                "most_common_fxt": most_common_fxt,
                "fxt_seed_signature_count": disagreement,
            }
        )

    df = pd.DataFrame.from_records(records)
    df.sort_values(["count"], ascending=False, inplace=True)
    df.to_csv(os.path.join(args.output_dir, "tokenization_vocabulary_by_string.csv"), index=False)

    # Per-language summary
    per_lang = []
    for lang in languages:
        langs_rows = df[df["langs"].apply(lambda d: lang in d)]
        per_lang.append(
            {
                "language": lang,
                "unique_normalized_strings": langs_rows.shape[0],
                "avg_bpe_signatures_per_string": langs_rows["bpe_unique_signatures"].mean(),
                "avg_fxt_signatures_per_string": langs_rows["fxt_unique_signatures"].mean(),
                "percent_strings_with_fxt_seed_variation": (
                    (langs_rows["fxt_seed_signature_count"] > 1).mean() * 100
                ),
            }
        )

    pd.DataFrame(per_lang).to_csv(os.path.join(args.output_dir, "per_language_summary.csv"), index=False)

    # Top ambiguous examples (by number of FXT signatures)
    amb = df.sort_values(["fxt_unique_signatures", "count"], ascending=[False, False]).head(200)
    amb.to_csv(os.path.join(args.output_dir, "top_ambiguous_strings.csv"), index=False)

    # Save full JSON
    with open(os.path.join(args.output_dir, "vocab_signatures.json"), "w") as f:
        json.dump({k: {"count": v["count"], "langs": v["langs"], "bpe_signatures": dict(v["bpe_signatures"]), "fxt_signatures": dict(v["fxt_signatures"])} for k, v in vocab.items()}, f, indent=2)

    print("Wrote results to", args.output_dir)


if __name__ == "__main__":
    main()
