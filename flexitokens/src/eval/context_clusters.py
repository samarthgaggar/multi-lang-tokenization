#!/usr/bin/env python3
"""Explore whether similar sentence contexts align with word split patterns."""

import argparse
import json
import os
import sqlite3
from array import array
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def load_language_occurrences(database_path, language):
    """Load saved word spans and their sentence text for one language."""
    with sqlite3.connect(database_path) as connection:
        return pd.read_sql_query(
            """SELECT o.language, o.sentence_id, s.source_config, s.source_row,
                      s.text AS sentence, o.char_start, o.char_end, o.surface_word,
                      o.lowercase_word, o.boundary_signature, o.segmentation_display,
                      o.piece_count, o.has_intra_character_boundary
               FROM occurrences AS o
               JOIN sentences AS s USING(language, sentence_id)
               WHERE o.language = ?
               ORDER BY o.sentence_id, o.char_start""",
            connection,
            params=(language,),
        )


def summarize_words(occurrences, min_frequency=1):
    """Count frequency and distinct observed split patterns for each word."""
    if occurrences.empty:
        return pd.DataFrame(columns=["word", "occurrence_frequency", "sentence_frequency",
                                     "unique_segmentations", "dominant_segmentation",
                                     "dominant_count", "dominant_share"])
    rows = []
    for word, group in occurrences.groupby("lowercase_word", sort=True):
        pattern_counts = group["boundary_signature"].value_counts()
        if len(group) < min_frequency:
            continue
        dominant_signature = pattern_counts.index[0]
        display = group.loc[group["boundary_signature"] == dominant_signature,
                            "segmentation_display"].iloc[0]
        rows.append({
            "word": word,
            "occurrence_frequency": len(group),
            "sentence_frequency": group["sentence_id"].nunique(),
            "unique_segmentations": len(pattern_counts),
            "dominant_segmentation": display,
            "dominant_count": int(pattern_counts.iloc[0]),
            "dominant_share": float(pattern_counts.iloc[0] / len(group)),
        })
    return pd.DataFrame(rows)


def summarize_language_words(database_path, language, min_frequency=1):
    """Return per-word occurrence and distinct-split counts for corpus charts."""
    with sqlite3.connect(database_path) as connection:
        frequencies = pd.read_sql_query(
            """SELECT lowercase_word AS word, COUNT(*) AS occurrence_frequency
               FROM occurrences WHERE language=?
               GROUP BY lowercase_word HAVING COUNT(*) >= ?
               ORDER BY lowercase_word""",
            connection, params=(language, min_frequency),
        )
        split_counts = pd.read_sql_query(
            """SELECT lowercase_word AS word, COUNT(*) AS unique_segmentations
               FROM (
                   SELECT lowercase_word, boundary_signature
                   FROM occurrences WHERE language=?
                   GROUP BY lowercase_word, boundary_signature
               )
               GROUP BY lowercase_word
               ORDER BY lowercase_word""",
            connection, params=(language,),
        )
    summary = frequencies.merge(split_counts, on="word", how="left")
    return summary


def load_target_occurrences(database_path, language, word, case_mode):
    """Load only the requested word occurrences and their source sentences."""
    if case_mode == "lower":
        field, target = "o.lowercase_word", word.lower()
    elif case_mode == "exact":
        field, target = "o.surface_word", word
    else:
        raise ValueError("case_mode must be 'lower' or 'exact'")
    query = f"""SELECT o.language, o.sentence_id, s.text AS sentence,
                      o.char_start, o.char_end, o.surface_word, o.lowercase_word,
                      o.boundary_signature, o.segmentation_display
               FROM occurrences o JOIN sentences s USING(language, sentence_id)
               WHERE o.language=? AND {field}=?
               ORDER BY o.sentence_id, o.char_start"""
    with sqlite3.connect(database_path) as connection:
        selected = pd.read_sql_query(query, connection, params=(language, target))
    if selected.empty:
        raise ValueError(f"No saved occurrences found for {word!r} using {case_mode} matching")
    patterns = (selected.groupby("boundary_signature", as_index=False)
                .agg(segmentation_display=("segmentation_display", "first"),
                     occurrence_count=("sentence_id", "size"),
                     sentence_count=("sentence_id", "nunique"))
                .sort_values(["occurrence_count", "segmentation_display"], ascending=[False, True])
                .reset_index(drop=True))
    patterns["occurrence_percentage"] = 100 * patterns["occurrence_count"] / len(selected)
    return selected, patterns, field, target


def _sentence_vectors_from_database(database_path, language, selected, target_field, target):
    """Build sparse sentence vectors from SQLite, leaving out the target word."""
    target_sentence_ids = sorted(selected["sentence_id"].unique())
    sentence_to_row = {sentence_id: index for index, sentence_id in enumerate(target_sentence_ids)}
    table = None
    vocabulary = []
    word_to_column = {}
    rows, columns, values = array("i"), array("i"), array("d")
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TEMP TABLE target_sentence_ids (sentence_id TEXT PRIMARY KEY)")
        connection.executemany("INSERT INTO target_sentence_ids VALUES (?)",
                               ((sentence_id,) for sentence_id in target_sentence_ids))
        exclusion = f"o.{target_field.split('.', 1)[1]} != ?"
        vocabulary = [row[0] for row in connection.execute(
            f"""SELECT DISTINCT o.lowercase_word
                FROM target_sentence_ids t CROSS JOIN occurrences o
                WHERE o.language=? AND o.sentence_id=t.sentence_id
                  AND {exclusion}
                ORDER BY o.lowercase_word""",
            (language, target),
        )]
        word_to_column = {word: index for index, word in enumerate(vocabulary)}
        cursor = connection.execute(
            f"""SELECT o.sentence_id, o.lowercase_word, COUNT(*)
                FROM target_sentence_ids t CROSS JOIN occurrences o
                WHERE o.language=? AND o.sentence_id=t.sentence_id
                  AND {exclusion}
                GROUP BY o.sentence_id, o.lowercase_word
                ORDER BY o.sentence_id, o.lowercase_word""",
            (language, target),
        )
        for sentence_id, word, count in cursor:
            rows.append(sentence_to_row[sentence_id])
            columns.append(word_to_column[word])
            values.append(count)
    count_matrix = csr_matrix((values, (rows, columns)),
                              shape=(len(target_sentence_ids), len(vocabulary)), dtype=np.float64)
    matrix = normalize(count_matrix.copy(), norm="l2", axis=1, copy=False)
    context_table = (selected.groupby("sentence_id", as_index=False)
                     .agg(sentence=("sentence", "first"),
                          target_occurrences=("surface_word", "size")))
    pattern_by_sentence = (selected.groupby("sentence_id")["segmentation_display"]
                           .apply(list).to_dict())
    spans_by_sentence = (selected.groupby("sentence_id")[["char_start", "char_end", "surface_word",
                                                           "segmentation_display", "boundary_signature"]]
                         .apply(lambda frame: frame.to_dict(orient="records")).to_dict())
    top_words = {}
    nonempty_by_sentence = {}
    for sentence_id, row_index in sentence_to_row.items():
        row = count_matrix.getrow(row_index)
        order = np.argsort(row.data)[::-1][:12]
        top_words[sentence_id] = [(vocabulary[row.indices[index]], int(row.data[index]))
                                  for index in order]
        nonempty_by_sentence[sentence_id] = bool(row.nnz)
    context_table["target_segmentations"] = context_table["sentence_id"].map(pattern_by_sentence)
    context_table["target_spans"] = context_table["sentence_id"].map(spans_by_sentence)
    context_table["top_context_words"] = context_table["sentence_id"].map(top_words)
    context_table["has_nonempty_context"] = context_table["sentence_id"].map(nonempty_by_sentence)
    context_table["sentence_vector_index"] = context_table["sentence_id"].map(sentence_to_row)
    context_table = context_table.sort_values("sentence_vector_index").reset_index(drop=True)
    return matrix, vocabulary, context_table


def _project_context_vectors(matrix):
    """Project sparse sentence vectors to 2D for visualization only."""
    coordinates = np.zeros((matrix.shape[0], 2), dtype=float)
    component_count = min(2, matrix.shape[0] - 1, matrix.shape[1] - 1)
    if component_count > 0:
        projection = TruncatedSVD(n_components=component_count, random_state=0)
        coordinates[:, :component_count] = projection.fit_transform(matrix)
    return coordinates


def _distinct_vector_count(matrix):
    return len({(matrix.getrow(index).indices.tobytes(), matrix.getrow(index).data.tobytes())
                for index in range(matrix.shape[0])})


def analyze_word_contexts(database_path, language, word, case_mode="lower",
                          n_clusters=None, similarity_sample_size=200):
    """Cluster one context per sentence and compare resulting split patterns.

    The model split labels are never used as vector features. By default the
    requested cluster count is the number of distinct split patterns observed
    for the word, as suggested by the professor.
    """
    selected, pattern_table, target_field, target = load_target_occurrences(
        database_path, language, word, case_mode
    )
    matrix, vocabulary, contexts = _sentence_vectors_from_database(
        database_path, language, selected, target_field, target
    )
    nonempty = contexts["has_nonempty_context"].to_numpy(dtype=bool)
    labels = np.full(len(contexts), -1, dtype=int)
    requested_k = int(n_clusters if n_clusters is not None else len(pattern_table))
    if requested_k < 1:
        raise ValueError("Cluster count must be at least 1")
    effective_k = min(requested_k, int(nonempty.sum()), _distinct_vector_count(matrix[nonempty]))
    reason = None
    if effective_k < requested_k:
        reason = (f"Reduced from {requested_k} to {effective_k}: only that many nonempty, "
                  "distinct sentence context vectors are available.")
    if effective_k >= 1:
        model = KMeans(n_clusters=effective_k, random_state=0, n_init=10, max_iter=300)
        labels[nonempty] = model.fit_predict(matrix[nonempty])
    contexts["cluster"] = [f"cluster {value + 1}" if value >= 0 else "empty context"
                            for value in labels]
    context_coordinates = _project_context_vectors(matrix)
    contexts["context_map_x"] = context_coordinates[:, 0]
    contexts["context_map_y"] = context_coordinates[:, 1]

    pattern_rows = []
    for cluster in sorted(contexts["cluster"].unique(), key=lambda value: (value == "empty context", value)):
        member_contexts = contexts[contexts["cluster"] == cluster]
        member_ids = set(member_contexts["sentence_id"])
        member_occurrences = selected[selected["sentence_id"].isin(member_ids)]
        grouped_patterns = (member_occurrences.groupby("boundary_signature")
                            .agg(segmentation=("segmentation_display", "first"),
                                 occurrence_count=("sentence_id", "size")))
        total = int(grouped_patterns["occurrence_count"].sum())
        for signature, pattern in grouped_patterns.iterrows():
            count = int(pattern["occurrence_count"])
            segmentation = pattern["segmentation"]
            pattern_rows.append({
                "cluster": cluster,
                "boundary_signature": signature,
                "segmentation": segmentation,
                "pattern_label": f"{segmentation} [{signature or 'no internal cut'}]",
                "occurrence_count": count,
                "occurrence_share": float(count / total) if total else 0.0,
                "sentence_count": int(member_contexts["sentence_id"].nunique()),
                "target_occurrences_in_cluster": total,
            })
    cluster_patterns = pd.DataFrame(pattern_rows)
    cluster_summary = (cluster_patterns.groupby("cluster", as_index=False)
                       .agg(sentence_count=("sentence_count", "first"),
                            target_occurrence_count=("target_occurrences_in_cluster", "first"),
                            distinct_segmentations=("boundary_signature", "nunique"),
                            dominant_segmentation_share=("occurrence_share", "max"))
                       if not cluster_patterns.empty else pd.DataFrame())
    if not cluster_summary.empty:
        cluster_summary["dominant_segmentation_percentage"] = (
            100 * cluster_summary["dominant_segmentation_share"]
        )

    selected_positions = np.flatnonzero(nonempty)
    if len(selected_positions) > similarity_sample_size:
        take = np.linspace(0, len(selected_positions) - 1, similarity_sample_size, dtype=int)
        selected_positions = selected_positions[take]
    similarity = cosine_similarity(matrix[selected_positions]) if len(selected_positions) else np.empty((0, 0))
    similarity_ids = contexts.iloc[selected_positions]["sentence_id"].tolist()
    similarity_frame = pd.DataFrame(similarity, index=similarity_ids, columns=similarity_ids)

    return {
        "language": language,
        "query_word": word,
        "case_mode": case_mode,
        "occurrence_count": len(selected),
        "sentence_context_count": int(selected["sentence_id"].nunique()),
        "unique_segmentation_count": int(len(pattern_table)),
        "requested_clusters": requested_k,
        "effective_clusters": effective_k,
        "cluster_count_note": reason,
        "feature_definition": "lowercase word counts from the sentence; all matched target word occurrences removed; L2-normalized",
        "clustering": "KMeans on normalized count vectors; random_state=0; n_init=10",
        "pattern_counts": pattern_table,
        "contexts": contexts,
        "cluster_patterns": cluster_patterns,
        "cluster_summary": cluster_summary,
        "similarity": similarity_frame,
        "vocabulary_size": len(vocabulary),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="results/global_voices_context_analysis/analysis.sqlite3")
    parser.add_argument("--language", required=True, choices=("en", "es", "ru", "hi"))
    parser.add_argument("--word", required=True)
    parser.add_argument("--case-mode", choices=("lower", "exact"), default="lower")
    parser.add_argument("--clusters", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    result = analyze_word_contexts(args.database, args.language, args.word, args.case_mode, args.clusters)
    print(json.dumps({key: value for key, value in result.items()
                      if not isinstance(value, pd.DataFrame)}, indent=2, ensure_ascii=False))
    if args.output_dir:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        for key in ("pattern_counts", "contexts", "cluster_patterns", "cluster_summary", "similarity"):
            result[key].to_csv(output / f"{args.language}_{args.word}_{key}.csv")


if __name__ == "__main__":
    main()
