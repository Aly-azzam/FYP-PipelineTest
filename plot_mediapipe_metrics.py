import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {
    "elapsed_seconds",
    "system_cpu_percent",
    "process_tree_rss_memory_mb",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot CPU and memory usage over time from monitor_existing_process.py CSV."
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to CSV file produced by monitor_existing_process.py",
    )
    parser.add_argument(
        "--title-prefix",
        default="MediaPipe (CPU)",
        help="Title prefix for the charts",
    )
    parser.add_argument(
        "--cpu-output",
        default="mediapipe_cpu_usage_over_time.png",
        help="Output PNG for CPU chart",
    )
    parser.add_argument(
        "--memory-output",
        default="mediapipe_memory_usage_over_time.png",
        help="Output PNG for memory chart",
    )
    args = parser.parse_args()

    csv_path = Path(args.input_csv)
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("CSV is empty. No data to plot.")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {', '.join(sorted(missing))}"
        )

    x = df["elapsed_seconds"]
    cpu = df["system_cpu_percent"]
    mem = df["process_tree_rss_memory_mb"]

    # CPU chart
    plt.figure(figsize=(12, 6))
    plt.plot(x, cpu, linewidth=2)
    plt.title(f"{args.title_prefix} CPU Usage Over Time", fontsize=16)
    plt.xlabel("Elapsed seconds", fontsize=12)
    plt.ylabel("CPU utilization (%)", fontsize=12)
    plt.grid(True, alpha=0.6)
    plt.tight_layout()
    plt.savefig(args.cpu_output, dpi=200)
    plt.close()

    # Memory chart
    plt.figure(figsize=(12, 6))
    plt.plot(x, mem, linewidth=2)
    plt.title(f"{args.title_prefix} Memory Usage Over Time", fontsize=16)
    plt.xlabel("Elapsed seconds", fontsize=12)
    plt.ylabel("RAM used (MB)", fontsize=12)
    plt.grid(True, alpha=0.6)
    plt.tight_layout()
    plt.savefig(args.memory_output, dpi=200)
    plt.close()

    print(f"Saved CPU plot to: {args.cpu_output}")
    print(f"Saved memory plot to: {args.memory_output}")


if __name__ == "__main__":
    main()
