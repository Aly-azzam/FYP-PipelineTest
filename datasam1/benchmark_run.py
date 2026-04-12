import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import psutil
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from pynvml import (
        nvmlInit,
        nvmlShutdown,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetUtilizationRates,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetName,
        nvmlDeviceGetCount,
    )
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False


def init_gpu(gpu_index=0):
    if not NVML_AVAILABLE:
        return None
    try:
        nvmlInit()
        count = nvmlDeviceGetCount()
        if count <= 0 or gpu_index >= count:
            return None
        return nvmlDeviceGetHandleByIndex(gpu_index)
    except Exception:
        return None


def shutdown_gpu():
    if not NVML_AVAILABLE:
        return
    try:
        nvmlShutdown()
    except Exception:
        pass


def get_gpu_stats(handle):
    if handle is None:
        return {
            "gpu_name": None,
            "gpu_util_percent": None,
            "gpu_mem_used_mb": None,
            "gpu_mem_total_mb": None,
        }

    try:
        util = nvmlDeviceGetUtilizationRates(handle)
        mem = nvmlDeviceGetMemoryInfo(handle)
        name = nvmlDeviceGetName(handle)

        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="ignore")

        return {
            "gpu_name": name,
            "gpu_util_percent": float(util.gpu),
            "gpu_mem_used_mb": float(mem.used / (1024 ** 2)),
            "gpu_mem_total_mb": float(mem.total / (1024 ** 2)),
        }
    except Exception:
        return {
            "gpu_name": None,
            "gpu_util_percent": None,
            "gpu_mem_used_mb": None,
            "gpu_mem_total_mb": None,
        }


def safe_mean(series):
    s = pd.Series(series).dropna()
    return float(s.mean()) if not s.empty else None


def safe_max(series):
    s = pd.Series(series).dropna()
    return float(s.max()) if not s.empty else None


def get_process_tree(root_proc):
    try:
        procs = [root_proc] + root_proc.children(recursive=True)
        alive = []
        for p in procs:
            try:
                if p.is_running():
                    alive.append(p)
            except Exception:
                pass
        return alive
    except Exception:
        return []


def prime_cpu_counters(processes, seen_pids):
    for p in processes:
        try:
            if p.pid not in seen_pids:
                p.cpu_percent(interval=None)
                seen_pids.add(p.pid)
        except Exception:
            pass


def collect_tree_metrics(root_proc, seen_pids):
    processes = get_process_tree(root_proc)
    prime_cpu_counters(processes, seen_pids)

    total_cpu = 0.0
    total_rss_mb = 0.0
    proc_count = 0

    for p in processes:
        try:
            total_cpu += float(p.cpu_percent(interval=None))
            total_rss_mb += float(p.memory_info().rss / (1024 ** 2))
            proc_count += 1
        except Exception:
            pass

    return {
        "tree_cpu_percent": total_cpu,
        "tree_rss_mb": total_rss_mb,
        "process_count": proc_count,
    }


def save_plot(df, output_png, title, gpu_name=None):
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    axes[0].plot(
        df["elapsed_sec"],
        df["tree_cpu_percent"],
        label="Process tree CPU %",
        color="tab:blue",
        linewidth=2,
    )
    axes[0].plot(
        df["elapsed_sec"],
        df["system_cpu_percent"],
        label="System CPU %",
        color="tab:orange",
        linewidth=2,
    )
    axes[0].set_ylabel("CPU %")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(
        df["elapsed_sec"],
        df["tree_rss_mb"],
        label="Process tree RAM (MB)",
        color="tab:blue",
        linewidth=2,
    )
    axes[1].plot(
        df["elapsed_sec"],
        df["system_ram_percent"],
        label="System RAM %",
        color="tab:orange",
        linewidth=2,
    )
    axes[1].set_ylabel("RAM")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    gpu_util_present = "gpu_util_percent" in df.columns and df["gpu_util_percent"].notna().any()
    gpu_mem_present = "gpu_mem_used_mb" in df.columns and df["gpu_mem_used_mb"].notna().any()

    ax_gpu = axes[2]

    if gpu_util_present and gpu_mem_present:
        ax_gpu_mem = ax_gpu.twinx()

        line1 = ax_gpu.plot(
            df["elapsed_sec"],
            df["gpu_util_percent"],
            label="GPU Util (%)",
            color="tab:blue",
            linewidth=2,
        )[0]

        line2 = ax_gpu_mem.plot(
            df["elapsed_sec"],
            df["gpu_mem_used_mb"],
            label="GPU Mem Used (MB)",
            color="tab:red",
            linewidth=2,
        )[0]

        ax_gpu.set_ylabel("GPU Util (%)", color="tab:blue")
        ax_gpu_mem.set_ylabel("GPU Mem Used (MB)", color="tab:red")
        ax_gpu.tick_params(axis="y", labelcolor="tab:blue")
        ax_gpu_mem.tick_params(axis="y", labelcolor="tab:red")
        ax_gpu.grid(True, alpha=0.3)

        ax_gpu.legend(
            handles=[line1, line2],
            labels=["GPU Util (%)", "GPU Mem Used (MB)"],
            loc="upper left",
        )
    else:
        if gpu_util_present:
            ax_gpu.plot(
                df["elapsed_sec"],
                df["gpu_util_percent"],
                label="GPU Util (%)",
                color="tab:blue",
                linewidth=2,
            )
        if gpu_mem_present:
            ax_gpu.plot(
                df["elapsed_sec"],
                df["gpu_mem_used_mb"],
                label="GPU Mem Used (MB)",
                color="tab:red",
                linewidth=2,
            )

        ax_gpu.set_ylabel("GPU")
        ax_gpu.grid(True, alpha=0.3)
        if gpu_util_present or gpu_mem_present:
            ax_gpu.legend(loc="upper left")

    ax_gpu.set_xlabel("Elapsed seconds")

    final_title = title
    if gpu_name:
        final_title += f" | GPU: {gpu_name}"

    fig.suptitle(final_title)
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def build_summary(df, label, command, frames):
    elapsed_sec = float(df["elapsed_sec"].max()) if not df.empty else 0.0
    effective_fps = (frames / elapsed_sec) if frames and elapsed_sec > 0 else None

    gpu_name = None
    if "gpu_name" in df.columns:
        gpu_name = next((x for x in df["gpu_name"].dropna().tolist() if x), None)

    return {
        "label": label,
        "command": command,
        "elapsed_sec": elapsed_sec,
        "frames_processed": frames,
        "effective_fps": effective_fps,
        "avg_tree_cpu_percent": safe_mean(df["tree_cpu_percent"]),
        "max_tree_cpu_percent": safe_max(df["tree_cpu_percent"]),
        "avg_tree_rss_mb": safe_mean(df["tree_rss_mb"]),
        "max_tree_rss_mb": safe_max(df["tree_rss_mb"]),
        "avg_system_cpu_percent": safe_mean(df["system_cpu_percent"]),
        "max_system_cpu_percent": safe_max(df["system_cpu_percent"]),
        "avg_system_ram_percent": safe_mean(df["system_ram_percent"]),
        "max_system_ram_percent": safe_max(df["system_ram_percent"]),
        "avg_process_count": safe_mean(df["process_count"]),
        "max_process_count": safe_max(df["process_count"]),
        "gpu_name": gpu_name,
        "avg_gpu_util_percent": safe_mean(df["gpu_util_percent"]) if "gpu_util_percent" in df else None,
        "max_gpu_util_percent": safe_max(df["gpu_util_percent"]) if "gpu_util_percent" in df else None,
        "avg_gpu_mem_used_mb": safe_mean(df["gpu_mem_used_mb"]) if "gpu_mem_used_mb" in df else None,
        "max_gpu_mem_used_mb": safe_max(df["gpu_mem_used_mb"]) if "gpu_mem_used_mb" in df else None,
    }


def print_summary_table(summary):
    rows = [
        ("label", summary["label"]),
        ("gpu_name", summary["gpu_name"]),
        ("elapsed_sec", summary["elapsed_sec"]),
        ("frames_processed", summary["frames_processed"]),
        ("effective_fps", summary["effective_fps"]),
        ("avg_tree_cpu_percent", summary["avg_tree_cpu_percent"]),
        ("max_tree_cpu_percent", summary["max_tree_cpu_percent"]),
        ("avg_tree_rss_mb", summary["avg_tree_rss_mb"]),
        ("max_tree_rss_mb", summary["max_tree_rss_mb"]),
        ("avg_system_cpu_percent", summary["avg_system_cpu_percent"]),
        ("max_system_cpu_percent", summary["max_system_cpu_percent"]),
        ("avg_system_ram_percent", summary["avg_system_ram_percent"]),
        ("max_system_ram_percent", summary["max_system_ram_percent"]),
        ("avg_process_count", summary["avg_process_count"]),
        ("max_process_count", summary["max_process_count"]),
        ("avg_gpu_util_percent", summary["avg_gpu_util_percent"]),
        ("max_gpu_util_percent", summary["max_gpu_util_percent"]),
        ("avg_gpu_mem_used_mb", summary["avg_gpu_mem_used_mb"]),
        ("max_gpu_mem_used_mb", summary["max_gpu_mem_used_mb"]),
    ]
    table = pd.DataFrame(rows, columns=["metric", "value"])
    print("\n=== BENCHMARK SUMMARY ===")
    print(table.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Benchmark CPU / RAM / GPU usage for a command.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--sample-sec", type=float, default=0.5)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--outdir", default="benchmark_outputs")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]

    if not command:
        print("Error: you must pass a command after --")
        sys.exit(1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / f"{args.label}_samples.csv"
    png_path = outdir / f"{args.label}_plot.png"
    json_path = outdir / f"{args.label}_summary.json"

    print("\nLaunching command:")
    print(" ".join(command))

    env = os.environ.copy()
    proc = subprocess.Popen(command, env=env)

    root_proc = psutil.Process(proc.pid)
    seen_pids = set()
    prime_cpu_counters(get_process_tree(root_proc), seen_pids)

    gpu_handle = init_gpu(args.gpu_index)
    if gpu_handle is None:
        print("GPU monitoring: unavailable")
    else:
        first_gpu_stats = get_gpu_stats(gpu_handle)
        print(f"GPU monitoring: enabled | GPU = {first_gpu_stats['gpu_name']}")

    samples = []
    start_time = time.time()
    last_status_print = 0.0
    return_code = None

    try:
        while proc.poll() is None:
            time.sleep(args.sample_sec)

            elapsed = time.time() - start_time
            tree_stats = collect_tree_metrics(root_proc, seen_pids)
            system_cpu = psutil.cpu_percent(interval=None)
            system_ram = psutil.virtual_memory().percent
            gpu_stats = get_gpu_stats(gpu_handle)

            samples.append(
                {
                    "elapsed_sec": elapsed,
                    "tree_cpu_percent": tree_stats["tree_cpu_percent"],
                    "tree_rss_mb": tree_stats["tree_rss_mb"],
                    "process_count": tree_stats["process_count"],
                    "system_cpu_percent": system_cpu,
                    "system_ram_percent": system_ram,
                    "gpu_name": gpu_stats["gpu_name"],
                    "gpu_util_percent": gpu_stats["gpu_util_percent"],
                    "gpu_mem_used_mb": gpu_stats["gpu_mem_used_mb"],
                    "gpu_mem_total_mb": gpu_stats["gpu_mem_total_mb"],
                }
            )

            if elapsed - last_status_print >= 5.0:
                print(
                    f"[benchmark] t={elapsed:.1f}s | "
                    f"tree_cpu={tree_stats['tree_cpu_percent']:.1f}% | "
                    f"ram={tree_stats['tree_rss_mb']:.1f} MB | "
                    f"gpu_util={gpu_stats['gpu_util_percent']} | "
                    f"gpu_mem_mb={gpu_stats['gpu_mem_used_mb']}"
                )
                last_status_print = elapsed

        return_code = proc.wait()

    finally:
        shutdown_gpu()

    df = pd.DataFrame(samples)
    if df.empty:
        print("No samples collected.")
        sys.exit(1)

    df.to_csv(csv_path, index=False)

    summary = build_summary(df, args.label, command, args.frames)
    summary["return_code"] = return_code
    summary["samples_csv"] = str(csv_path)
    summary["plot_png"] = str(png_path)
    summary["summary_json"] = str(json_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    save_plot(df, str(png_path), f"Benchmark: {args.label}", gpu_name=summary["gpu_name"])
    print_summary_table(summary)

    print("\nGenerated files:")
    print(csv_path)
    print(png_path)
    print(json_path)


if __name__ == "__main__":
    main()