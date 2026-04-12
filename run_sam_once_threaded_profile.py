import time
import threading
import json
from pathlib import Path

import psutil
import pandas as pd
import matplotlib.pyplot as plt

from sam_runner import track_two_videos_from_selected_points

# GPU
from pynvml import *
nvmlInit()
gpu_handle = nvmlDeviceGetHandleByIndex(0)


samples = []
running = True


def sampler():
    proc = psutil.Process()

    while running:
        cpu = proc.cpu_percent(interval=None)
        ram = proc.memory_info().rss / (1024 ** 2)

        sys_cpu = psutil.cpu_percent()
        sys_ram = psutil.virtual_memory().percent

        util = nvmlDeviceGetUtilizationRates(gpu_handle)
        mem = nvmlDeviceGetMemoryInfo(gpu_handle)

        samples.append({
            "time": time.time(),
            "cpu": cpu,
            "ram": ram,
            "sys_cpu": sys_cpu,
            "sys_ram": sys_ram,
            "gpu_util": util.gpu,
            "gpu_mem": mem.used / (1024 ** 2),
        })

        time.sleep(0.1)


def plot(df):
    t0 = df["time"].iloc[0]
    df["t"] = df["time"] - t0

    # CPU
    plt.figure(figsize=(10,4))
    plt.plot(df["t"], df["cpu"], label="Process CPU")
    plt.plot(df["t"], df["sys_cpu"], label="System CPU")
    plt.legend()
    plt.title("CPU")
    plt.savefig("benchmark_outputs/cpu.png")
    plt.close()

    # RAM
    plt.figure(figsize=(10,4))
    plt.plot(df["t"], df["ram"], label="Process RAM")
    plt.plot(df["t"], df["sys_ram"], label="System RAM")
    plt.legend()
    plt.title("RAM")
    plt.savefig("benchmark_outputs/ram.png")
    plt.close()

    # GPU
    plt.figure(figsize=(10,4))
    plt.plot(df["t"], df["gpu_util"], label="GPU Util")
    plt.plot(df["t"], df["gpu_mem"], label="GPU Mem")
    plt.legend()
    plt.title("GPU")
    plt.savefig("benchmark_outputs/gpu.png")
    plt.close()


def main():
    global running

    print("START SAM + PROFILER")

    thread = threading.Thread(target=sampler)
    thread.start()

    # RUN SAM
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

    running = False
    thread.join()

    print("SAM FINISHED")

    df = pd.DataFrame(samples)
    Path("benchmark_outputs").mkdir(exist_ok=True)

    plot(df)

    print("DONE")


if __name__ == "__main__":
    main()
    