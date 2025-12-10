# file: benchmark_runner.py
"""
Benchmark driver that compares llmao_joins vs AutoFJ on selected datasets.
Usage:
  python benchmark_runner.py --datasets_root /path/to/autofj/benchmark
"""

import argparse
import json
import os
import time

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from autofj import AutoFJ
from autofj.datasets import load_data

from llmao_joins.config import PipelineConfig
from llmao_joins.benchmark_adapter import run_llmao_benchmark


# -------------------------------------------------------------
# AutoFJ BASELINE RUNNER (direct, NO manual autofj_id)
# -------------------------------------------------------------
def run_autofj(dataset_name: str) -> dict:
    """
    Run AutoFJ on one of its built-in benchmark datasets.

    AutoFJ’s docs say:
      left, right, gt = load_data(dataset_name)
      fj = AutoFJ(precision_target=0.9)
      result = fj.join(left, right, "id")

    We follow that exactly. AutoFJ internally handles its own
    'autofj_id' column; we do NOT create or touch it.
    """
    print(f"[BASELINE] Running AutoFJ on {dataset_name}")

    # Load AutoFJ benchmark tables
    left, right, gt = load_data(dataset_name)

    start = time.perf_counter()
    fj = AutoFJ(precision_target=0.9)
    # join on the documented id column
    result = fj.join(left, right, "id")
    end = time.perf_counter()

    # ----- Build ground-truth value pairs (title_l, title_r) -----
    left_map = left.set_index("id")["title"].str.lower()
    right_map = right.set_index("id")["title"].str.lower()

    gt_pairs = set(
        (left_map[row["id_l"]], right_map[row["id_r"]])
        for _, row in gt.iterrows()
    )

    pred_pairs = set(
        zip(result["title_l"].str.lower(), result["title_r"].str.lower())
    )

    tp = len(pred_pairs & gt_pairs)
    fp = len(pred_pairs - gt_pairs)
    fn = len(gt_pairs - pred_pairs)

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    print(f"[BASELINE] AutoFJ  P={precision:.3f} R={recall:.3f} F1={f1:.3f}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "runtime_sec": end - start,
    }


# -------------------------------------------------------------
# GRAPH GENERATION
# -------------------------------------------------------------
def plot_benchmark_graphs(summary, output_root):
    os.makedirs(os.path.join(output_root, "graphs"), exist_ok=True)
    df = pd.DataFrame(summary)

    # ---- LLMao Precision / Recall / F1 ----
    plt.figure(figsize=(10, 6))
    df_plot = df.melt(
        id_vars="dataset",
        value_vars=["llmao_precision", "llmao_recall", "llmao_f1"],
        var_name="metric",
        value_name="value",
    )
    sns.barplot(data=df_plot, x="dataset", y="value", hue="metric")
    plt.title("LLMao Precision / Recall / F1")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_root, "graphs", "llmao_metrics.png"))
    plt.close()

    # ---- Runtime Comparison ----
    plt.figure(figsize=(10, 6))
    df_runtime = df[
        ["dataset", "autofj_runtime", "llmao_runtime"]
    ].melt(id_vars="dataset", var_name="system", value_name="runtime_sec")
    sns.barplot(data=df_runtime, x="dataset", y="runtime_sec", hue="system")
    plt.title("Runtime Comparison: AutoFJ vs LLMao")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_root, "graphs", "runtime_comparison.png"))
    plt.close()

    print(f"[GRAPHS] Saved graphs to {output_root}/graphs/")


# -------------------------------------------------------------
# MAIN ENTRY
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_root", required=True)
    parser.add_argument("--output_root", default="benchmark_results")
    parser.add_argument("--llm_api_key", default=None)
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    
    selected_datasets = ["TennisTournament"]  # or ["Case", "Amphibian", "Artwork"]

    # We only use datasets_root for our CSV paths (LLMao side);
    # AutoFJ itself uses load_data(dataset_name).
    dataset_dirs = [
        os.path.join(args.datasets_root, name) for name in selected_datasets
    ]

    summary = []

    for ds_path in dataset_dirs:
        name = os.path.basename(ds_path)
        print(f"\n=== DATASET: {name} ===")

        # -------- AutoFJ baseline --------
        autofj_metrics = run_autofj(name)
        with open(
            os.path.join(args.output_root, f"{name}_autofj.json"), "w"
        ) as f:
            json.dump(autofj_metrics, f, indent=2)

        # -------- LLMao-Joins run --------
        left_csv = os.path.join(ds_path, "left.csv")
        right_csv = os.path.join(ds_path, "right.csv")

        cfg = PipelineConfig(
            left_csv=left_csv,
            right_csv=right_csv,
            left_col="title",
            right_col="title",
            output_dir=os.path.join(args.output_root, f"{name}_llmao"),
            llm_api_key=args.llm_api_key,
        )

        llmao_metrics = run_llmao_benchmark(cfg, llm_api_key=args.llm_api_key)

        summary.append(
            {
                "dataset": name,
                "llmao_f1": llmao_metrics["f1"],
                "llmao_precision": llmao_metrics["precision"],
                "llmao_recall": llmao_metrics["recall"],
                "autofj_runtime": autofj_metrics["runtime_sec"],
                "llmao_runtime": llmao_metrics["runtime_sec"],
            }
        )

    summary_path = os.path.join(args.output_root, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nBenchmark summary saved to {summary_path}")

    plot_benchmark_graphs(summary, args.output_root)


if __name__ == "__main__":
    main()