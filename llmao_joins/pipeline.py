# file: llmao_joins/pipeline.py
import json
import os
import time
from typing import Dict, Tuple, List
import pandas as pd
import networkx as nx
import random

from .config import PipelineConfig
from .io_and_normalization import (
    build_lookup,
    load_column_values,
    ValueRecord
)
from .candidate_generation import (
    CandidatePair,
    generate_embedding_candidates,
    generate_rule_pairs,
    generate_string_candidates,
)
from .data_preprocessing import (
    DataNormalizer,
    CandidatePairGenerator,
    PairStatistics,
    StringSimilarity,
    NGramGenerator,
    MinHash
)
from .scoring_and_llm import score_all, run_llm_gate
from .graph_and_clusters import build_graph_and_clusters, bootstrap_pairs_from_existing_clusters

from dotenv import load_dotenv
load_dotenv()

def convert_to_tuple_format(pairs_dict: Dict[Tuple[str, str], CandidatePair]) -> List[Tuple[str, str, float]]:
    """Convert CandidatePair dict to list of (v1, v2, score) tuples"""
    result = []
    for pair in pairs_dict.values():
        # Use the combined score if available, otherwise use max of available scores
        score = pair.combined_score
        if score is None:
            score = max([s for s in [pair.rule_score, pair.string_sim, pair.embed_sim] if s is not None], default=0.0)
        result.append((pair.left_norm, pair.right_norm, score))
    return result


def extract_features_from_pair(pair: CandidatePair, string_sim: StringSimilarity, ngram_gen: NGramGenerator) -> Dict[str, float]:
    """Extract comprehensive features from a candidate pair"""
    features = {}
    
    # Existing scores
    features['rule_score'] = pair.rule_score or 0.0
    features['string_sim'] = pair.string_sim or 0.0
    features['embed_sim'] = pair.embed_sim or 0.0
    
    # String similarity features
    features['levenshtein_sim'] = string_sim.levenshtein_similarity(pair.left_norm, pair.right_norm)
    features['levenshtein_dist'] = string_sim.levenshtein_distance(pair.left_norm, pair.right_norm)
    features['jaro_winkler'] = string_sim.jaro_winkler_similarity(pair.left_norm, pair.right_norm)
    
    # N-gram features
    ngrams1 = ngram_gen.generate_ngrams(pair.left_norm, 3)
    ngrams2 = ngram_gen.generate_ngrams(pair.right_norm, 3)
    features['ngram_jaccard'] = ngram_gen.jaccard_similarity(ngrams1, ngrams2)
    
    # Length features
    features['length_diff'] = abs(len(pair.left_norm) - len(pair.right_norm))
    features['length_ratio'] = min(len(pair.left_norm), len(pair.right_norm)) / max(len(pair.left_norm), len(pair.right_norm)) if max(len(pair.left_norm), len(pair.right_norm)) > 0 else 0
    
    # Word-level features
    words1 = set(pair.left_norm.split())
    words2 = set(pair.right_norm.split())
    features['word_jaccard'] = ngram_gen.jaccard_similarity(words1, words2)
    features['word_count_diff'] = abs(len(words1) - len(words2))
    
    minhash = MinHash(num_hashes=200)

    # Calculate MinHash-based Jaccard similarity between two sets
    features['minhash']= minhash.jaccard_similarity(words1, words2)

    # Prefix/suffix features
    max_prefix = 0
    for i in range(min(len(pair.left_norm), len(pair.right_norm))):
        if pair.left_norm[i] == pair.right_norm[i]:
            max_prefix += 1
        else:
            break
    features['common_prefix_length'] = max_prefix
    features['common_prefix_ratio'] = max_prefix / min(len(pair.left_norm), len(pair.right_norm)) if min(len(pair.left_norm), len(pair.right_norm)) > 0 else 0
    
    return features


def run_pipeline(cfg: PipelineConfig, llm_api_key: str | None = None) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    metrics: Dict[str, float | int] = {}

    # Initialize components
    normalizer = DataNormalizer()
    string_sim = StringSimilarity()
    ngram_gen = NGramGenerator()

    # ---------------------------------------------------------
    # STEP 1 — LOAD + NORMALIZE INPUT TABLES
    # ---------------------------------------------------------
    t0 = time.perf_counter()
    left_records, left_df = load_column_values(cfg.left_csv, cfg.left_col, side="left")
    right_records, right_df = load_column_values(cfg.right_csv, cfg.right_col, side="right")
    metrics["n_left_unique_values"] = len(left_records)
    metrics["n_right_unique_values"] = len(right_records)
    
    if len(left_records) > cfg.max_unique_values or len(right_records) > cfg.max_unique_values:
        raise RuntimeError(
            f"Too many unique values: left={len(left_records)}, right={len(right_records)}, "
            f"max={cfg.max_unique_values}"
        )
    t1 = time.perf_counter()
    metrics["time_load_and_normalize_sec"] = t1 - t0

    # ---------------------------------------------------------
    # STEP 2 — CANDIDATE GENERATION
    # ---------------------------------------------------------

    # 2a — Bootstrap from existing graph clusters (knowledge reuse)
    pairs: Dict[Tuple[str, str], CandidatePair] = bootstrap_pairs_from_existing_clusters(
        left_records,
        right_records,
        cfg,
    )
    t_bootstrap_done = time.perf_counter()
    metrics["n_graph_prior_pairs"] = len(pairs)
    metrics["time_graph_prior_sec"] = t_bootstrap_done - t1

    # 2b — Rule-based
    pairs = generate_rule_pairs(left_records, right_records, pairs)
    t2 = time.perf_counter()
    metrics["time_rule_pairs_sec"] = t2 - t_bootstrap_done
    metrics["n_rule_pairs"] = len(pairs)

    # 2c — STRING SIMILARITY MATCHES
    pairs = generate_string_candidates(
        left_records,
        right_records,
        ngram_range=cfg.ngram_range,
        top_k=cfg.string_top_k,
        min_sim=cfg.string_min_sim,
        pairs=pairs,
    )
    t3 = time.perf_counter()
    metrics["time_string_candidates_sec"] = t3 - t2
    metrics["n_pairs_after_string"] = len(pairs)

    # 2d — EMBEDDING-BASED MATCHES (SentenceTransformers)
    pairs = generate_embedding_candidates(
        left_records,
        right_records,
        model_name=cfg.embed_model_name,
        top_k=cfg.embed_top_k,
        min_sim=cfg.embed_min_sim,
        pairs=pairs,
    )
    t4 = time.perf_counter()
    metrics["time_embedding_candidates_sec"] = t4 - t3
    metrics["n_pairs_after_embedding"] = len(pairs)

    # ---------------------------------------------------------
    # STEP 3 – FEATURE EXTRACTION & SCORING
    # ---------------------------------------------------------
    print("[STEP 3] Extracting features and scoring...")
    t5_start = time.perf_counter()
    
    for pair in pairs.values():
        # Extract comprehensive features
        pair.features = extract_features_from_pair(pair, string_sim, ngram_gen)
        
        # Calculate combined score (weighted average)
        scores = []
        weights = []
        
        if pair.rule_score is not None:
            scores.append(pair.rule_score)
            weights.append(cfg.w_rule)
        
        if pair.string_sim is not None:
            scores.append(pair.string_sim)
            weights.append(cfg.w_string)
        
        if pair.embed_sim is not None:
            scores.append(pair.embed_sim)
            weights.append(cfg.w_embed)
        if pair.features['jaro_winkler'] is not None:
            scores.append(pair.features['jaro_winkler'])
            weights.append(cfg.w_jaro_winkler)

        # Add Levenshtein similarity if available
        if pair.features['levenshtein_sim'] is not None:
            scores.append(pair.features['levenshtein_sim'])
            weights.append(cfg.w_levenshtein)

        # Add n-gram similarity if available
        if pair.features['ngram_jaccard'] is not None:
            scores.append(pair.features['ngram_jaccard'])
            weights.append(cfg.w_ngram_jaccard)
        if pair.features['minhash'] is not None:
            scores.append(pair.features['minhash'])
            weights.append(cfg.w_minhash)

        if scores:
            pair.combined_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        else:
            pair.combined_score = 0.0
    
    t5 = time.perf_counter()
    metrics["time_scoring_sec"] = t5 - t5_start
    print(f"  Extracted features for {len(pairs)} pairs")
    print(f"  Time: {t5-t5_start:.2f}s\n")
    store_random_pairs_in_file(pairs)
    # ---------------------------------------------------------
    # STEP 4 – CONFIDENCE CATEGORIZATION
    # ---------------------------------------------------------
    print("[STEP 4] Categorizing by confidence...")
    
    # Convert to tuple format for filtering
    tuple_pairs = convert_to_tuple_format(pairs)
    
    # Use PairStatistics for categorization
    categorized = PairStatistics.filter_by_confidence(
        tuple_pairs,
        high_threshold=cfg.w_high,
        low_threshold=cfg.w_low
    )
    
    high_conf = categorized['high_confidence']
    medium_conf = categorized['uncertain']
    low_conf = categorized['low_confidence']
    
    metrics["n_high_confidence"] = len(high_conf)
    metrics["n_medium_confidence"] = len(medium_conf)
    metrics["n_low_confidence"] = len(low_conf)
    
    print(f"  High confidence (auto-accept): {len(high_conf)}")
    print(f"  Medium confidence (→ LLM): {len(medium_conf)}")
    print(f"  Low confidence (reject): {len(low_conf)}\n")

    # ---------------------------------------------------------
    # STEP 5 – LLM GATE (OPTIONAL)
    # ---------------------------------------------------------
    t6 = time.perf_counter()
    llm_stats = run_llm_gate(pairs, cfg, api_key=llm_api_key)
    t7 = time.perf_counter()
    metrics["time_llm_gate_sec"] = t7 - t6
    metrics["llm_n_calls"] = llm_stats.n_calls
    metrics["llm_total_tokens"] = llm_stats.total_tokens
    metrics["llm_estimated_cost_usd"] = llm_stats.estimated_cost(cfg.llm_price_per_1k_tokens)

    # Re-score all pairs after injecting llm_score
    score_all(pairs, cfg)
    # ---------------------------------------------------------
    # STEP 5 — ACCEPT HIGH CONFIDENCE MATCHES
    # ---------------------------------------------------------
    accepted_pairs = [
        p for p in pairs.values()
        if (p.combined_score or 0.0) >= cfg.accept_threshold
    ]
    metrics["n_accepted_pairs"] = len(accepted_pairs)

    # ---------------------------------------------------------
    # STEP 6 — BUILD GRAPH IN NEO4J + EXTRACT CLUSTERS
    # ---------------------------------------------------------
    t8 = time.perf_counter()
    graph_stats = build_graph_and_clusters(pairs, cfg)
    t9 = time.perf_counter()
    metrics["time_graph_and_clusters_sec"] = t9 - t8
    metrics["graph_n_nodes"] = graph_stats.n_nodes
    metrics["graph_n_edges"] = graph_stats.n_edges
    metrics["graph_n_clusters"] = graph_stats.n_clusters

    # Use the cluster_map produced from the Neo4j backed graph
    cluster_map: Dict[str, int] = graph_stats.cluster_map

    # ---------------------------------------------------------
    # STEP 7 — MATERIALIZE THE FINAL SEMANTIC JOIN
    # ---------------------------------------------------------
    _, left_raw_to_norm = build_lookup(left_records)
    _, right_raw_to_norm = build_lookup(right_records)

    left_cluster_ids = left_df[cfg.left_col].fillna("").astype(str).map(
        lambda x: cluster_map.get(left_raw_to_norm.get(x, ""), None)
    )
    right_cluster_ids = right_df[cfg.right_col].fillna("").astype(str).map(
        lambda x: cluster_map.get(right_raw_to_norm.get(x, ""), None)
    )

    left_df_out = left_df.copy()
    right_df_out = right_df.copy()
    left_df_out["__cluster_id"] = left_cluster_ids
    right_df_out["__cluster_id"] = right_cluster_ids

    left_df_out = left_df_out[left_df_out["__cluster_id"].notna()]
    right_df_out = right_df_out[right_df_out["__cluster_id"].notna()]

    joined = pd.merge(
        left_df_out,
        right_df_out,
        on="__cluster_id",
        how="inner",
        suffixes=("_left", "_right"),
    )

    t10 = time.perf_counter()
    metrics["time_join_materialization_sec"] = t10 - t9
    metrics["total_runtime_sec"] = t10 - t0
    metrics["n_joined_rows"] = len(joined)

    # ---------------------------------------------------------
    # STEP 8 — SAVE OUTPUTS
    # ---------------------------------------------------------

    # Save matched candidate pairs
    pairs_rows = []
    for p in accepted_pairs:
        pairs_rows.append({
            "left_raw": p.left_raw,
            "right_raw": p.right_raw,
            "left_norm": p.left_norm,
            "right_norm": p.right_norm,
            "rule_score": p.rule_score,
            "string_sim": p.string_sim,
            "embed_sim": p.embed_sim,
            "llm_score": p.llm_score,
            "combined_score": p.combined_score,
            "sources": ",".join(p.sources),
        })
    pd.DataFrame(pairs_rows).to_csv(
        os.path.join(cfg.output_dir, "matched_pairs.csv"), index=False
    )

    # Save clusters
    cluster_rows = []
    for raw, norm in left_raw_to_norm.items():
        cid = cluster_map.get(norm)
        if cid:
            cluster_rows.append({"side": "left", "raw": raw, "norm": norm, "cluster_id": cid})

    for raw, norm in right_raw_to_norm.items():
        cid = cluster_map.get(norm)
        if cid:
            cluster_rows.append({"side": "right", "raw": raw, "norm": norm, "cluster_id": cid})

    pd.DataFrame(cluster_rows).drop_duplicates().to_csv(
        os.path.join(cfg.output_dir, "synonym_clusters.csv"), index=False
    )

    # Save semantic join output
    joined.to_csv(
        os.path.join(cfg.output_dir, "semantic_join.csv"),
        index=False
    )

    # Save metrics
    with open(os.path.join(cfg.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[LLMAO-JOINS] Finished pipeline. Outputs written to {cfg.output_dir}")
    print(f"[LLMAO-JOINS] Matched pairs: {len(accepted_pairs)}, joined rows: {len(joined)}")
    
    if llm_stats.n_calls > 0:
        print(f"[LLM GATE] LLM calls: {llm_stats.n_calls}, Tokens: {llm_stats.total_tokens}, Estimated cost: ${llm_stats.estimated_cost(cfg.llm_price_per_1k_tokens):.4f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LLMao-Joins semantic join pipeline")
    parser.add_argument("--left_csv", required=True)
    parser.add_argument("--right_csv", required=True)
    parser.add_argument("--left_col", required=True)
    parser.add_argument("--right_col", required=True)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--neo4j_uri", default=None)
    parser.add_argument("--neo4j_user", default=None)
    parser.add_argument("--neo4j_password", default=None)
    parser.add_argument("--embed_model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--llm_api_key", default=None)
    parser.add_argument("--abbrevation_master", default=None)

    args = parser.parse_args()

    cfg = PipelineConfig(
        left_csv=args.left_csv,
        right_csv=args.right_csv,
        left_col=args.left_col,
        right_col=args.right_col,
        output_dir=args.output_dir,
        neo4j_uri=args.neo4j_uri or os.getenv("NEO4J_URI"),
        neo4j_user=args.neo4j_user or os.getenv("NEO4J_USER"),
        neo4j_password=args.neo4j_password or os.getenv("NEO4J_PASSWORD"),
        llm_api_key=args.llm_api_key or os.getenv("LLM_API_KEY"),
        embed_model_name=args.embed_model_name,
        abbrevation_master= args.abbrevation_master or None
    )

    run_pipeline(cfg, llm_api_key=cfg.llm_api_key)


if __name__ == "__main__":
    main()
