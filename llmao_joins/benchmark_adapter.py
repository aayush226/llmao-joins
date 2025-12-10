# file: llmao_joins/benchmark_adapter.py
"""
Wrapper to evaluate llmao_joins pipeline against AutoFJ benchmark datasets.
"""

import json
import os
import time
from typing import Dict

import pandas as pd

from .config import PipelineConfig
from .pipeline import run_pipeline


def run_llmao_benchmark(cfg: PipelineConfig, llm_api_key: str | None = None) -> Dict:
    """
    Execute llmao_joins pipeline and compute benchmark metrics
    against AutoFJ's ground-truth (gt.csv with id_l,id_r).
    """

    # --------- Run pipeline ----------
    t0 = time.perf_counter()
    run_pipeline(cfg, llm_api_key=llm_api_key)
    t1 = time.perf_counter()

    out_dir = cfg.output_dir
    pairs_path = os.path.join(out_dir, "matched_pairs.csv")

    # --------- Load predictions ----------
    pred = pd.read_csv(pairs_path)

    # --------- Load left/right/gt ----------
    dataset_dir = os.path.dirname(cfg.left_csv)

    left_df = pd.read_csv(os.path.join(dataset_dir, "left.csv"))
    right_df = pd.read_csv(os.path.join(dataset_dir, "right.csv"))
    gt = pd.read_csv(os.path.join(dataset_dir, "gt.csv"))

    # Map ID → textual value
    left_map = left_df.set_index("id")["title"].str.lower()
    right_map = right_df.set_index("id")["title"].str.lower()

    # Build ground-truth set of value pairs
    gt_pairs = set(
        (left_map[row["id_l"]], right_map[row["id_r"]])
        for _, row in gt.iterrows()
    )

    # Predicted pairs
    pred_pairs = set(
        zip(pred.left_norm.str.lower(), pred.right_norm.str.lower())
    )

    # --------- Compute metrics ----------
    tp = len(pred_pairs & gt_pairs)
    fp = len(pred_pairs - gt_pairs)
    fn = len(gt_pairs - pred_pairs)

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "runtime_sec": t1 - t0,
    }

    # Save benchmark metrics
    with open(os.path.join(out_dir, "benchmark_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[BENCHMARK] Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    return metrics