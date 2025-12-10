# file: llmao_joins/pipeline.py
import json
import os
import time
from typing import Dict, Tuple

import pandas as pd
import networkx as nx

from .config import PipelineConfig
from .io_and_normalization import (
    build_lookup,
    load_column_values,
)
from .candidate_generation import (
    CandidatePair,
    generate_embedding_candidates,
    generate_rule_pairs,
    generate_string_candidates,
)
from .scoring_and_llm import score_all, run_llm_gate
from .graph_and_clusters import build_graph_and_clusters, bootstrap_pairs_from_existing_clusters

from dotenv import load_dotenv
load_dotenv()

def _fmt_score(x: float | None) -> str:
    return "None" if x is None else f"{x:.3f}"

def run_pipeline(cfg: PipelineConfig, llm_api_key: str | None = None) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    metrics: Dict[str, float | int] = {}

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
    # STEP 3 — SCORE ALL CANDIDATES
    # ---------------------------------------------------------
    score_all(pairs, cfg)          # Weighted combination scoring
    t5 = time.perf_counter()
    metrics["time_scoring_sec"] = t5 - t4

    # DEBUG: Top 10 pairs BEFORE LLM gate
    debug_before = sorted(
        pairs.values(),
        key=lambda p: p.combined_score or 0.0,
        reverse=True,
    )[:10]
    print("\n[DEBUG] Top 10 pairs BEFORE LLM gate:")
    for p in debug_before:
        print(
            f"  {p.left_raw!r} ↔ {p.right_raw!r} | "
            f"rule={_fmt_score(p.rule_score)} "
            f"str={_fmt_score(p.string_sim)} "
            f"embed={_fmt_score(p.embed_sim)} "
            f"llm={_fmt_score(p.llm_score)} "
            f"combined={_fmt_score(p.combined_score)} "
            f"sources={p.sources}"
        )

    # ---------------------------------------------------------
    # STEP 4 — LLM GATE FOR UNCERTAIN MATCHES
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

    # DEBUG: Top 10 pairs AFTER LLM gate
    debug_after = sorted(
        pairs.values(),
        key=lambda p: p.combined_score or 0.0,
        reverse=True,
    )[:10]
    print("\n[DEBUG] Top 10 pairs AFTER LLM gate:")
    for p in debug_after:
        print(
            f"  {p.left_raw!r} ↔ {p.right_raw!r} | "
            f"rule={_fmt_score(p.rule_score)} "
            f"str={_fmt_score(p.string_sim)} "
            f"embed={_fmt_score(p.embed_sim)} "
            f"llm={_fmt_score(p.llm_score)} "
            f"combined={_fmt_score(p.combined_score)} "
            f"sources={p.sources}"
        )
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

     # DEBUG: Save ALL candidate pairs (accepted + rejected)
    all_rows = []
    for p in pairs.values():
        all_rows.append({
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
    pd.DataFrame(all_rows).to_csv(
        os.path.join(cfg.output_dir, "all_candidate_pairs.csv"), index=False
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

    # optinal - i moved passwords to .env
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
