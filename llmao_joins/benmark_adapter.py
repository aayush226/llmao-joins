# file: llmao_joins/benchmark_adapter.py
"""
Wrapper to evaluate llmao_joins pipeline against AutoFJ benchmark datasets.
"""

import json
import os
import time
from typing import Dict

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from .config import PipelineConfig
from .pipeline import run_pipeline


def run_llmao_benchmark(cfg: PipelineConfig, llm_api_key: str | None = None) -> Dict:
    """
    Execute the semantic join pipeline and compute benchmark metrics
    against provided ground truth (gold standard) file.

    Expected benchmark directory layout:
        dataset/
          left.csv
          right.csv
          gt.csv   # gold pairs, columns: left_value,right_value
    """
    t0 = time.perf_counter()
    run_pipeline(cfg, llm_api_key=llm_api_key)
    t1 = time.perf_counter()

    out_dir = cfg.output_dir
    pairs_path = os.path.join(out_dir, "matched_pairs.csv")
    gt_path = os.path.join(os.path.dirname(cfg.left_csv), "gt.csv")

    pred = pd.read_csv(pairs_path)
    gt = pd.read_csv(gt_path)

    pred_pairs = set(zip(pred.left_raw.str.lower(), pred.right_raw.str.lower()))
    gt_pairs = set(zip(gt.iloc[:, 0].str.lower(), gt.iloc[:, 1].str.lower()))

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

    with open(os.path.join(out_dir, "benchmark_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[BENCHMARK] Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    return metrics
