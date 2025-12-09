# file: benchmark_runner.py
"""
Benchmark driver that compares llmao_joins vs AutoFJ on standard datasets.
Usage:
  python benchmark_runner.py --datasets_root ./autofj/benchmark/datasets
"""

import argparse
import glob
import json
import os
import subprocess
import time

from llmao_joins.config import PipelineConfig
from llmao_joins.benchmark_adapter import run_llmao_benchmark


def run_autofj(dataset_dir: str) -> dict:
    """
    Run AutoFJ baseline from chu-data-lab repository.
    Requires their code installed as 'autofj'.
    """
    print(f"[BASELINE] Running AutoFJ on {dataset_dir}")
    start = time.perf_counter()
    result = subprocess.run(
        ["python", "-m", "autofj.benchmark.run_benchmark", "--dataset", dataset_dir],
        capture_output=True,
        text=True,
        check=False,
    )
    end = time.perf_counter()
    metrics = {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "runtime_sec": end - start,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_root", required=True)
    parser.add_argument("--output_root", default="benchmark_results")
    parser.add_argument("--llm_api_key", default=None)
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    dataset_dirs = sorted(glob.glob(os.path.join(args.datasets_root, "*")))

    summary = []
    for ds_path in dataset_dirs:
        if not os.path.isdir(ds_path):
            continue
        name = os.path.basename(ds_path)
        print(f"\n=== DATASET: {name} ===")

        # AutoFJ baseline
        autofj_metrics = run_autofj(ds_path)
        with open(os.path.join(args.output_root, f"{name}_autofj.json"), "w") as f:
            json.dump(autofj_metrics, f, indent=2)

        # LLMao-Joins run
        left_csv = os.path.join(ds_path, "left.csv")
        right_csv = os.path.join(ds_path, "right.csv")

        cfg = PipelineConfig(
            left_csv=left_csv,
            right_csv=right_csv,
            left_col="left_attr",
            right_col="right_attr",
            output_dir=os.path.join(args.output_root, f"{name}_llmao"),
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
    print(f"\nBenchmark summary written to {summary_path}")


if __name__ == "__main__":
    main()
