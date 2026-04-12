import json
import time
from pathlib import Path

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
    )
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

from sam_runner import track_two_videos_from_selected_points


def init_gpu(gpu_index=0):
    if not NVML_AVAILABLE:
        return None
    try:
        nvmlInit()
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


def sample_system(proc, gpu_handle, elapsed_sec):
    process_ram_mb = proc.memory_info().rss / (1024 ** 2)
    process_cpu = proc.cpu_percent(interval=None)
    system_cpu = psutil.cpu_percent(interval=None)
    system_ram = psutil.virtual_memory().percent
    gpu = get_gpu_stats(gpu_handle)

    return {
        "elapsed_sec": elapsed_sec,
        "process_cpu_percent": process_cpu,
        "process_rss_mb": process_ram_mb,
        "system_cpu_percent": system_cpu,
        "system_ram_percent": system_ram,
        "gpu_name": gpu["gpu_name"],
        "gpu_util_percent": gpu["gpu_util_percent"],
        "gpu_mem_used_mb": gpu["gpu_mem_used_mb"],
        "gpu_mem_total_mb": gpu["gpu_mem_total_mb"],
    }


def save_plots(df, outdir, label, gpu_name=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base_title = f"Benchmark: {label}"
    if gpu_name:
        base_title += f" | GPU: {gpu_name}"

    # CPU
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["elapsed_sec"], df["process_cpu_percent"], label="Process CPU %", linewidth=2)
    ax.plot(df["elapsed_sec"], df["system_cpu_percent"], label="System CPU %", linewidth=2)
    ax.set_title(base_title + " - CPU")
    ax.set_xlabel("Elapsed seconds")
    ax.set_ylabel("CPU %")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"{label}_cpu.png", dpi=150)
    plt.close(fig)

    # RAM
    fig, ax1 = plt.subplots(figsize=(12, 5))
    l1 = ax1.plot(df["elapsed_sec"], df["process_rss_mb"], label="Process RAM (MB)", linewidth=2)[0]
    ax1.set_xlabel("Elapsed seconds")
    ax1.set_ylabel("Process RAM (MB)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    l2 = ax2.plot(df["elapsed_sec"], df["system_ram_percent"], label="System RAM %", linewidth=2)[0]
    ax2.set_ylabel("System RAM %")

    ax1.set_title(base_title + " - RAM")
    ax1.legend([l1, l2], ["Process RAM (MB)", "System RAM %"])
    fig.tight_layout()
    fig.savefig(outdir / f"{label}_ram.png", dpi=150)
    plt.close(fig)

    # GPU
    fig, ax1 = plt.subplots(figsize=(12, 5))
    lines = []
    labels = []

    if df["gpu_util_percent"].notna().any():
        l1 = ax1.plot(df["elapsed_sec"], df["gpu_util_percent"], label="GPU Util (%)", linewidth=2)[0]
        ax1.set_ylabel("GPU Util (%)")
        lines.append(l1)
        labels.append("GPU Util (%)")
    ax1.set_xlabel("Elapsed seconds")
    ax1.grid(True, alpha=0.3)

    if df["gpu_mem_used_mb"].notna().any():
        ax2 = ax1.twinx()
        l2 = ax2.plot(df["elapsed_sec"], df["gpu_mem_used_mb"], label="GPU Mem Used (MB)", linewidth=2)[0]
        ax2.set_ylabel("GPU Mem Used (MB)")
        lines.append(l2)
        labels.append("GPU Mem Used (MB)")

    ax1.set_title(base_title + " - GPU")
    if lines:
        ax1.legend(lines, labels)
    fig.tight_layout()
    fig.savefig(outdir / f"{label}_gpu.png", dpi=150)
    plt.close(fig)


def main():
    label = "gpu_sam_10s_stride3"
    outdir = Path("benchmark_outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    print("STEP 1: starting profiled SAM run")

    proc = psutil.Process()
    proc.cpu_percent(interval=None)

    gpu_handle = init_gpu(0)
    if gpu_handle is not None:
        print("GPU monitoring enabled")

    samples = []
    start_time = time.time()

    # background warm sample
    time.sleep(0.1)
    samples.append(sample_system(proc, gpu_handle, time.time() - start_time))

    print("STEP 2: running tracking")
    result = track_two_videos_from_selected_points(
        expert_video_path="test_videos/expert.mp4",
        learner_video_path="test_videos/learner.mp4",
        expert_time_seconds=1.0,
        learner_time_seconds=1.0,
        expert_point_xy=[297, 139],
        learner_point_xy=[302, 158],
        analysis_mode="first_n_seconds",
        max_seconds=10,
        frame_stride=3,
        output_root="outputs/benchmark_tracking",
    )

    samples.append(sample_system(proc, gpu_handle, time.time() - start_time))

    print("STEP 3: tracking finished")

    # save result json
    run_id = result["run"]["run_id"]
    run_json = Path("outputs/benchmark_tracking") / run_id / "json" / "main_result_bundle.json"
    run_json.parent.mkdir(parents=True, exist_ok=True)

    with open(run_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # build dataframe
    df = pd.DataFrame(samples)

    # if too few samples, duplicate timing checkpoints are not enough,
    # so add a simple summary-only graph note
    gpu_name = None
    if "gpu_name" in df.columns:
        names = [x for x in df["gpu_name"].dropna().tolist() if x]
        gpu_name = names[0] if names else None

    save_plots(df, outdir, label, gpu_name=gpu_name)

    summary = {
        "label": label,
        "gpu_name": gpu_name,
        "elapsed_sec": float(time.time() - start_time),
        "result_json": str(run_json),
        "cpu_plot": str(outdir / f"{label}_cpu.png"),
        "ram_plot": str(outdir / f"{label}_ram.png"),
        "gpu_plot": str(outdir / f"{label}_gpu.png"),
    }

    with open(outdir / f"{label}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    shutdown_gpu()

    print("DONE")
    print(summary)


if __name__ == "__main__":
    main()