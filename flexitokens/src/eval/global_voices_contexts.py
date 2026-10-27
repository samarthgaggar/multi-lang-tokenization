#!/usr/bin/env python3
"""Resumable sentence-level FlexiTokens analysis for Global Voices.

Each sentence is tokenized as a whole. Word spans and their byte-level
FlexiTokens boundaries are recorded afterwards, using the helpers from the
FLORES contextual analysis. Results are committed to SQLite in small batches
so a long language run can be stopped and resumed without losing completed
sentences.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import Dataset
from huggingface_hub import HfApi
from tokenizers.pre_tokenizers import ByteLevel

from src.eval.contextual_word_tokenization import (
    character_to_byte_offsets,
    choose_device,
    language_token_id,
    load_model,
    word_segmentation,
    word_spans_from_gpt2_pretokenizer,
    tokenize_batch,
)


DATASET = "sentence-transformers/parallel-sentences-global-voices"
LANGUAGE_CODES = {"en": "en", "es": "es", "ru": "ru", "hi": "hi"}
ENGLISH_CONFIGS = ("en-es", "en-ru", "en-hi")
METHOD_VERSION = "global-voices-contexts-v1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="model_ckpts/_2025-05-03_14-43-31")
    parser.add_argument("--output-dir", default="results/global_voices_context_analysis")
    parser.add_argument("--languages", default="en")
    parser.add_argument("--limit-per-language", type=int, default=0,
                        help="Process only the first N unique sentences per language; 0 means all.")
    parser.add_argument("--time-limit-minutes", type=float, default=25.0,
                        help="Save a checkpoint and stop cleanly after this many minutes; 0 means unlimited.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--commit-every", type=int, default=32,
                        help="Commit after this many complete sentences.")
    parser.add_argument("--verify-repeat-samples", type=int, default=24,
                        help="Re-tokenize this many saved sentences at startup and require exact equality.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Run the repeat check for the requested saved languages, without processing new sentences.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--revision", default="main",
                        help="Dataset revision. Record the resolved revision in the run manifest when available.")
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_text(language, text):
    digest = hashlib.sha256()
    digest.update(language.encode("utf-8"))
    digest.update(b"\0")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def atomic_json_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def connect_database(path):
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS sentences (
            language TEXT NOT NULL,
            sentence_id TEXT NOT NULL,
            source_config TEXT NOT NULL,
            source_row INTEGER NOT NULL,
            text TEXT NOT NULL,
            input_ids TEXT NOT NULL,
            token_byte_ends TEXT NOT NULL,
            hard_boundaries TEXT NOT NULL,
            gpt2_word_span_count INTEGER NOT NULL,
            PRIMARY KEY (language, sentence_id)
        );
        CREATE TABLE IF NOT EXISTS occurrences (
            language TEXT NOT NULL,
            sentence_id TEXT NOT NULL,
            char_start INTEGER NOT NULL,
            char_end INTEGER NOT NULL,
            surface_word TEXT NOT NULL,
            lowercase_word TEXT NOT NULL,
            byte_start INTEGER NOT NULL,
            byte_end INTEGER NOT NULL,
            boundary_signature TEXT NOT NULL,
            segmentation_display TEXT NOT NULL,
            piece_bytes_or_text TEXT NOT NULL,
            piece_count INTEGER NOT NULL,
            has_intra_character_boundary INTEGER NOT NULL,
            PRIMARY KEY (language, sentence_id, char_start, char_end),
            FOREIGN KEY (language, sentence_id)
                REFERENCES sentences (language, sentence_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS occurrence_word_idx
            ON occurrences(language, lowercase_word);
        CREATE INDEX IF NOT EXISTS occurrence_exact_word_idx
            ON occurrences(language, surface_word);
        CREATE INDEX IF NOT EXISTS occurrence_pattern_idx
            ON occurrences(language, lowercase_word, boundary_signature);
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    connection.commit()
    return connection


def source_configs(language):
    if language == "en":
        return ENGLISH_CONFIGS
    return (f"en-{LANGUAGE_CODES[language]}",)


def source_texts(config, revision):
    cache_root = Path.home() / ".cache/huggingface/datasets" / DATASET.replace("/", "___") / config
    candidates = sorted(cache_root.glob(f"*/{revision}/*-train.arrow"))
    if not candidates:
        raise FileNotFoundError(
            f"No local cached Arrow file for {config} at dataset revision {revision}. "
            "Download/cache this config before running offline."
        )
    arrow_path = candidates[-1]
    print(f"Reading {config} from memory-mapped local Arrow data: {arrow_path}", flush=True)
    dataset = Dataset.from_file(str(arrow_path))
    if not {"english", "non_english"}.issubset(set(dataset.column_names)):
        raise RuntimeError(f"{config}: expected english/non_english columns; found {dataset.column_names}")
    print(f"{config}: stream ready", flush=True)
    for row_index, row in enumerate(dataset):
        yield row_index, row


def get_dataset_text(language, row):
    column = "english" if language == "en" else "non_english"
    text = row.get(column)
    if not isinstance(text, str) or not text.strip():
        return None
    return text


def already_saved(connection, language, sentence_id):
    return connection.execute(
        "SELECT 1 FROM sentences WHERE language=? AND sentence_id=?",
        (language, sentence_id),
    ).fetchone() is not None


def build_sentence_payload(language, sentence_id, config, source_row, text, output, pretokenizer):
    spans = word_spans_from_gpt2_pretokenizer(text, pretokenizer)
    byte_offsets = character_to_byte_offsets(text)
    sentence = {
        "language": language,
        "sentence_id": sentence_id,
        "source_config": config,
        "source_row": source_row,
        "text": text,
        "input_ids": json.dumps(output["input_ids"]),
        "token_byte_ends": json.dumps(output["token_byte_ends"]),
        "hard_boundaries": json.dumps(output["hard_boundaries"]),
        "gpt2_word_span_count": len(spans),
    }
    occurrences = []
    for char_start, char_end in spans:
        segment = word_segmentation(
            text, char_start, char_end, output["token_byte_ends"], output["hard_boundaries"]
        )
        occurrences.append((
            language,
            sentence_id,
            segment["char_start"],
            segment["char_end"],
            segment["surface_word"],
            segment["surface_word"].lower(),
            segment["byte_start"],
            segment["byte_end"],
            segment["boundary_signature"],
            segment["segmentation_display"],
            segment["piece_bytes_or_text"],
            segment["piece_count"],
            int(segment["has_intra_character_boundary"]),
        ))
    if byte_offsets[-1] != len(text.encode("utf-8")):
        raise RuntimeError("Character-to-byte offsets do not cover the complete source sentence")
    return sentence, occurrences


def save_batch(connection, language, batch, outputs, pretokenizer):
    sentence_sql = """INSERT INTO sentences (
        language, sentence_id, source_config, source_row, text, input_ids,
        token_byte_ends, hard_boundaries, gpt2_word_span_count
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"""
    occurrence_sql = """INSERT INTO occurrences (
        language, sentence_id, char_start, char_end, surface_word, lowercase_word,
        byte_start, byte_end, boundary_signature, segmentation_display,
        piece_bytes_or_text, piece_count, has_intra_character_boundary
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
    for record, output in zip(batch, outputs):
        sentence, occurrences = build_sentence_payload(
            language, record["sentence_id"], record["source_config"], record["source_row"],
            record["text"], output, pretokenizer,
        )
        connection.execute(sentence_sql, (
            sentence["language"], sentence["sentence_id"], sentence["source_config"],
            sentence["source_row"], sentence["text"], sentence["input_ids"],
            sentence["token_byte_ends"], sentence["hard_boundaries"], sentence["gpt2_word_span_count"],
        ))
        connection.executemany(occurrence_sql, occurrences)
    connection.commit()


def database_counts(connection, languages):
    counts = {}
    for language in languages:
        sentences, occurrences = connection.execute(
            "SELECT (SELECT COUNT(*) FROM sentences WHERE language=?), "
            "(SELECT COUNT(*) FROM occurrences WHERE language=?)",
            (language, language),
        ).fetchone()
        counts[language] = {"sentences_saved": sentences, "word_occurrences_saved": occurrences}
    return counts


def verify_saved_samples(connection, model, tokenizer, model_config, device, language,
                         sample_count, batch_size):
    """Require exact repeated predictions for a stable, saved sentence sample."""
    rows = connection.execute(
        """SELECT sentence_id, text, input_ids, token_byte_ends, hard_boundaries
           FROM sentences WHERE language=? ORDER BY sentence_id LIMIT ?""",
        (language, sample_count),
    ).fetchall()
    if not rows:
        return "no saved examples available yet"
    records = [{"sentence_id": row[0], "text": row[1]} for row in rows]
    reproduced = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        outputs = tokenize_with_memory_fallback(
            model, tokenizer, [item["text"] for item in batch], language, model_config, device
        )
        for output, expected in zip(outputs, rows[start:start + batch_size]):
            actual = (output["input_ids"], output["token_byte_ends"], output["hard_boundaries"])
            saved = (json.loads(expected[2]), json.loads(expected[3]), json.loads(expected[4]))
            if actual != saved:
                raise RuntimeError(f"{language}: repeat check failed for sentence {expected[0]}")
            reproduced.append(expected[0])
    return f"{len(reproduced)} saved sentences reproduced exactly"


def tokenize_with_memory_fallback(model, tokenizer, texts, language, model_config, device):
    """Retry out-of-memory batches as smaller groups, eventually one by one."""
    try:
        return tokenize_batch(model, tokenizer, texts, language, model_config, device)
    except (RuntimeError, MemoryError) as error:
        message = str(error).lower()
        is_memory_error = isinstance(error, MemoryError) or "out of memory" in message or (
            "mps" in message and "memory" in message
        )
        if not is_memory_error or len(texts) <= 1:
            raise
        if device.type == "mps":
            torch.mps.empty_cache()
        midpoint = len(texts) // 2
        left = tokenize_with_memory_fallback(
            model, tokenizer, texts[:midpoint], language, model_config, device
        )
        right = tokenize_with_memory_fallback(
            model, tokenizer, texts[midpoint:], language, model_config, device
        )
        return left + right


def run_language(language, connection, model, tokenizer, config, device, args, start_time, progress_path):
    pretokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    target_count = args.limit_per_language or None
    unique_seen = 0
    skipped_saved = 0
    pending = []
    connection.execute("CREATE TEMP TABLE IF NOT EXISTS run_seen (sentence_id TEXT PRIMARY KEY)")
    connection.execute("DELETE FROM run_seen")
    configs = source_configs(language)
    print(f"{language}: scanning configs {', '.join(configs)}", flush=True)

    def flush():
        nonlocal pending
        if not pending:
            return
        outputs = []
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            outputs.extend(tokenize_with_memory_fallback(
                model, tokenizer, [record["text"] for record in batch], language, config, device
            ))
        save_batch(connection, language, pending, outputs, pretokenizer)
        print(f"{language}: checkpoint saved; {len(pending)} new sentences", flush=True)
        pending = []
        progress = json.loads(Path(progress_path).read_text())
        progress["languages"][language] = {
            "status": "running",
            **database_counts(connection, [language])[language],
            "last_checkpoint_at_utc": utc_now(),
        }
        progress["updated_at_utc"] = utc_now()
        atomic_json_write(progress_path, progress)

    stopped_for_time = False
    stopped_at_limit = False
    for source_config in configs:
        for source_row, row in source_texts(source_config, args.revision):
            if args.time_limit_minutes and (time.monotonic() - start_time) >= args.time_limit_minutes * 60:
                stopped_for_time = True
                break
            text = get_dataset_text(language, row)
            if text is None:
                continue
            sentence_id = sha256_text(language, text)
            inserted = connection.execute(
                "INSERT OR IGNORE INTO run_seen(sentence_id) VALUES (?)", (sentence_id,)
            ).rowcount
            if not inserted:
                continue
            unique_seen += 1
            if target_count and unique_seen > target_count:
                stopped_at_limit = True
                break
            if already_saved(connection, language, sentence_id):
                skipped_saved += 1
                if target_count and unique_seen >= target_count:
                    stopped_at_limit = True
                    break
                continue
            pending.append({
                "sentence_id": sentence_id,
                "source_config": source_config,
                "source_row": source_row,
                "text": text,
            })
            if len(pending) >= args.commit_every:
                flush()
            if target_count and unique_seen >= target_count:
                stopped_at_limit = True
                break
            if args.time_limit_minutes and (time.monotonic() - start_time) >= args.time_limit_minutes * 60:
                stopped_for_time = True
                break
        if stopped_for_time or stopped_at_limit:
            break
    flush()
    return {
        "unique_source_rows_seen_this_run": unique_seen,
        "previously_saved_rows_skipped": skipped_saved,
        "status": "pilot_complete" if stopped_at_limit else "paused" if stopped_for_time else "complete",
    }


def main():
    args = parse_args()
    languages = [value.strip().lower() for value in args.languages.split(",") if value.strip()]
    unknown = set(languages) - set(LANGUAGE_CODES)
    if unknown:
        raise ValueError(f"Unsupported languages {sorted(unknown)}; supported codes: {sorted(LANGUAGE_CODES)}")
    if len(set(languages)) != len(languages):
        raise ValueError("Each language may be requested only once per run")
    if args.commit_every < 1 or args.batch_size < 1:
        raise ValueError("--commit-every and --batch-size must be positive")
    if args.limit_per_language < 0 or args.time_limit_minutes < 0 or args.verify_repeat_samples < 0:
        raise ValueError("Sentence limit, time limit, and repeat sample count must be nonnegative")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "analysis.sqlite3"
    progress_path = output_dir / "progress.json"
    connection = connect_database(database_path)
    metadata_row = connection.execute("SELECT value FROM metadata WHERE key='method_version'").fetchone()
    if metadata_row and json.loads(metadata_row[0]) != METHOD_VERSION:
        raise RuntimeError("Existing database uses a different method version; choose a new output directory")
    connection.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('method_version',?)",
                       (json.dumps(METHOD_VERSION),))
    connection.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('dataset_source,revision',?)",
                       (json.dumps({"dataset": DATASET, "revision": args.revision}),))
    connection.commit()

    device = choose_device(args.device)
    model, tokenizer, model_config = load_model(args.model_path, device)
    model_path = Path(args.model_path)
    model_file = model_path / "model.pth"
    manifest_path = output_dir / "run_manifest.json"
    existing_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if existing_manifest:
        if existing_manifest.get("dataset_revision_requested") != args.revision:
            raise RuntimeError("Cannot resume: requested dataset revision differs from the saved manifest")
        resolved_revision = existing_manifest.get("dataset_revision_resolved")
        if not resolved_revision:
            raise RuntimeError("Saved manifest has no resolved dataset revision; use a new output directory")
    else:
        resolved_revision = HfApi().dataset_info(DATASET, revision=args.revision).sha
        if not resolved_revision:
            raise RuntimeError("Could not resolve the Global Voices dataset revision")
    manifest = {
        "created_at_utc": existing_manifest.get("created_at_utc") if existing_manifest else utc_now(),
        "dataset": DATASET,
        "dataset_revision_requested": args.revision,
        "dataset_revision_resolved": resolved_revision,
        "configs_by_language": {language: source_configs(language) for language in LANGUAGE_CODES},
        "dataset_columns": {"en": "english", "es": "non_english", "ru": "non_english", "hi": "non_english"},
        "deduplication": "exact sentence text within each language; English is the union of en-es, en-ru, en-hi",
        "model_path": str(model_path.resolve()),
        "model_file_size_bytes": model_file.stat().st_size,
        "model_file_mtime_ns": model_file.stat().st_mtime_ns,
        "model_config_sha256": hashlib.sha256((model_path / "config.json").read_bytes()).hexdigest(),
        "boundary_rule": "sigmoid(boundary_logit) > 0.5",
        "boundary_randomness": "none; deterministic threshold mode is forced",
        "word_span_method": "GPT-2 ByteLevel pre-tokenizer with contiguous Unicode letter/mark/number spans merged",
        "primary_grouping": "lowercase; original surface form retained",
        "environment": {
            "python": platform.python_version(),
            **{name: importlib.metadata.version(name) for name in
               ("torch", "transformers", "datasets", "tokenizers", "pandas")},
        },
        "method_version": METHOD_VERSION,
    }
    if existing_manifest:
        immutable_keys = ["dataset", "dataset_revision_requested", "model_path", "model_file_size_bytes",
                          "model_file_mtime_ns", "model_config_sha256", "boundary_rule", "method_version"]
        mismatches = [key for key in immutable_keys if existing_manifest.get(key) != manifest.get(key)]
        if mismatches:
            raise RuntimeError(f"Cannot resume database because run settings changed: {mismatches}")
    atomic_json_write(manifest_path, manifest)
    args.revision = resolved_revision

    session_start = time.monotonic()
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"languages": {}}
    progress["updated_at_utc"] = utc_now()
    progress["device"] = str(device)
    progress["database"] = str(database_path)
    if not args.verify_only:
        progress["session_started_at_utc"] = utc_now()
        progress["requested_minutes"] = args.time_limit_minutes
    atomic_json_write(progress_path, progress)
    try:
        for language in languages:
            if args.verify_repeat_samples:
                verification = verify_saved_samples(
                    connection, model, tokenizer, model_config, device, language,
                    args.verify_repeat_samples, args.batch_size,
                )
                current_progress = json.loads(progress_path.read_text())
                prior = current_progress.get("languages", {}).get(language, {})
                prior["repeat_verification"] = verification
                prior["repeat_verification_at_utc"] = utc_now()
                current_progress.setdefault("languages", {})[language] = prior
                atomic_json_write(progress_path, current_progress)
                print(f"{language}: repeat check: {verification}", flush=True)
            if args.verify_only:
                continue
            result = run_language(language, connection, model, tokenizer, model_config, device, args,
                                  session_start, progress_path)
            progress = json.loads(progress_path.read_text())
            progress["languages"][language] = {
                **result,
                **database_counts(connection, [language])[language],
                "updated_at_utc": utc_now(),
            }
            progress["updated_at_utc"] = utc_now()
            atomic_json_write(progress_path, progress)
            if result["status"] == "paused":
                print(f"{language}: time checkpoint reached; resume by rerunning the same command", flush=True)
                break
    except Exception as error:
        current_progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"languages": {}}
        failed_language = language if "language" in locals() else "unknown"
        current_progress.setdefault("languages", {})[failed_language] = {
            **current_progress.get("languages", {}).get(failed_language, {}),
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "updated_at_utc": utc_now(),
        }
        current_progress["updated_at_utc"] = utc_now()
        atomic_json_write(progress_path, current_progress)
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()
