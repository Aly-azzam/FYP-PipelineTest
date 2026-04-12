import time
import threading
import json
from pathlib import Path

import psutil
import pandas as pd
import matplotlib.pyplot as plt

from sam_runner import track_two_videos_from_selected_points
from sam_metrics import compute_metrics_for_two_videos

from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetName,
)

samples = []
running = True


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sampler():
    proc = psutil.Process()
    proc.cpu_percent(interval=None)

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)

    while running:
        cpu = proc.cpu_percent(interval=None)
        ram = proc.memory_info().rss / (1024 ** 2)

        sys_cpu = psutil.cpu_percent(interval=None)
        sys_ram = psutil.virtual_memory().percent

        util = nvmlDeviceGetUtilizationRates(handle)
        mem = nvmlDeviceGetMemoryInfo(handle)
        gpu_name = nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode("utf-8", errors="ignore")

        samples.append({
            "time": time.time(),
            "process_cpu_percent": cpu,
            "process_rss_mb": ram,
            "system_cpu_percent": sys_cpu,
            "system_ram_percent": sys_ram,
            "gpu_util_percent": float(util.gpu),
            "gpu_mem_used_mb": float(mem.used / (1024 ** 2)),
            "gpu_name": gpu_name,
        })

        time.sleep(0.1)

    nvmlShutdown()


def save_old_style_plot(df, output_png, title):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(df["elapsed_sec"], df["process_cpu_percent"], label="Process tree CPU %")
    axes[0].plot(df["elapsed_sec"], df["system_cpu_percent"], label="System CPU %")
    axes[0].set_ylabel("CPU %")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(df["elapsed_sec"], df["process_rss_mb"], label="Process tree RAM (MB)")
    axes[1].plot(df["elapsed_sec"], df["system_ram_percent"], label="System RAM %")
    axes[1].set_ylabel("RAM")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    gpu_has_util = df["gpu_util_percent"].notna().any()
    gpu_has_mem = df["gpu_mem_used_mb"].notna().any()

    if gpu_has_util:
        axes[2].plot(df["elapsed_sec"], df["gpu_util_percent"], label="GPU Util %")
    if gpu_has_mem:
        axes[2].plot(df["elapsed_sec"], df["gpu_mem_used_mb"], label="GPU Mem Used (MB)")

    axes[2].set_ylabel("GPU")
    axes[2].set_xlabel("Elapsed seconds")
    axes[2].grid(True, alpha=0.3)
    if gpu_has_util or gpu_has_mem:
        axes[2].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def main():
    global running

    print("STEP 1: starting profiler")
    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()

    print("STEP 2: running SAM tracking")
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

    print("STEP 3: tracking done")

    expert_raw_json_path = result["expert_video"]["raw_json_path"]
    learner_raw_json_path = result["learner_video"]["raw_json_path"]

    raw_json_bundle = {
        "expert_raw_json_path": expert_raw_json_path,
        "learner_raw_json_path": learner_raw_json_path,
        "expert_raw_json": load_json(expert_raw_json_path),
        "learner_raw_json": load_json(learner_raw_json_path),
    }

    print("STEP 4: computing derived metrics")
    metrics_result = compute_metrics_for_two_videos(
        expert_raw_json_path=expert_raw_json_path,
        learner_raw_json_path=learner_raw_json_path,
        output_root="outputs/benchmark_tracking",
    )

    expert_metrics_json_path = metrics_result["expert_metrics_json_path"]
    learner_metrics_json_path = metrics_result["learner_metrics_json_path"]

    derived_metrics_bundle = {
        "expert_metrics_json_path": expert_metrics_json_path,
        "learner_metrics_json_path": learner_metrics_json_path,
        "expert_metrics_json": load_json(expert_metrics_json_path),
        "learner_metrics_json": load_json(learner_metrics_json_path),
    }

    running = False
    thread.join()

    print("STEP 5: profiler stopped")

    df = pd.DataFrame(samples)
    if df.empty:
        raise RuntimeError("No samples collected")

    t0 = df["time"].iloc[0]
    df["elapsed_sec"] = df["time"] - t0

    run_id = result["run"]["run_id"]
    run_json_dir = Path("outputs/benchmark_tracking") / run_id / "json"
    run_json_dir.mkdir(parents=True, exist_ok=True)

    main_result_path = run_json_dir / "main_result_bundle.json"
    raw_bundle_path = run_json_dir / "raw_json_bundle.json"
    derived_bundle_path = run_json_dir / "derived_metrics_bundle.json"

    with open(main_result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with open(raw_bundle_path, "w", encoding="utf-8") as f:
        json.dump(raw_json_bundle, f, ensure_ascii=False, indent=2)

    with open(derived_bundle_path, "w", encoding="utf-8") as f:
        json.dump(derived_metrics_bundle, f, ensure_ascii=False, indent=2)

    outdir = Path("benchmark_outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    gpu_name = None
    names = [x for x in df["gpu_name"].dropna().tolist() if x]
    if names:
        gpu_name = names[0]

    title = "Benchmark: gpu_sam_10s_stride3"
    if gpu_name:
        title += f" | GPU: {gpu_name}"

    plot_path = outdir / "gpu_sam_10s_stride3_plot.png"
    save_old_style_plot(df, plot_path, title)

    samples_csv = outdir / "gpu_sam_10s_stride3_samples.csv"
    df.to_csv(samples_csv, index=False)

    summary = {
        "label": "gpu_sam_10s_stride3",
        "gpu_name": gpu_name,
        "elapsed_sec": float(df["elapsed_sec"].max()),
        "plot_png": str(plot_path),
        "samples_csv": str(samples_csv),
        "main_result_json": str(main_result_path),
        "raw_json_bundle": str(raw_bundle_path),
        "derived_metrics_bundle": str(derived_bundle_path),
    }

    summary_path = outdir / "gpu_sam_10s_stride3_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("DONE")
    print("Main result JSON:", main_result_path)
    print("RAW JSON SAM 2:", raw_bundle_path)
    print("Derived Metrics JSON:", derived_bundle_path)
    print("Plot:", plot_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()