import argparse
import json
import os
import sys
import time
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import matplotlib.pyplot as plt
import psutil

# Optional GPU monitoring
try:
    import pynvml  # type: ignore
    _PYNVML_AVAILABLE = True
except Exception:
    _PYNVML_AVAILABLE = False

# Your existing project functions
from sam_runner import track_two_videos_from_selected_points
from sam_metrics import load_json_file, export_derived_metrics_json


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_safe_write(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def now_iso() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="microseconds")


class GPUMonitor:
    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self.enabled = False
        self.handle = None
        self.gpu_name = None

        if _PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                self.gpu_name = pynvml.nvmlDeviceGetName(self.handle)
                if isinstance(self.gpu_name, bytes):
                    self.gpu_name = self.gpu_name.decode("utf-8", errors="ignore")
                self.enabled = True
            except Exception:
                self.enabled = False

    def sample(self) -> Dict[str, Optional[float]]:
        if not self.enabled or self.handle is None:
            return {
                "gpu_util_percent": None,
                "gpu_mem_used_mb": None,
                "gpu_mem_total_mb": None,
                "gpu_mem_percent": None,
            }

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)

            used_mb = mem.used / (1024 ** 2)
            total_mb = mem.total / (1024 ** 2)
            mem_percent = (used_mb / total_mb * 100.0) if total_mb > 0 else None

            return {
                "gpu_util_percent": float(util.gpu),
                "gpu_mem_used_mb": float(used_mb),
                "gpu_mem_total_mb": float(total_mb),
                "gpu_mem_percent": float(mem_percent) if mem_percent is not None else None,
            }
        except Exception:
            return {
                "gpu_util_percent": None,
                "gpu_mem_used_mb": None,
                "gpu_mem_total_mb": None,
                "gpu_mem_percent": None,
            }

    def shutdown(self) -> None:
        if self.enabled:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


class ResourceProfiler:
    def __init__(self, interval_sec: float = 0.5, gpu_index: int = 0):
        self.interval_sec = interval_sec
        self.gpu = GPUMonitor(gpu_index=gpu_index)
        self._stop_event = threading.Event()
        self._thread = None
        self.samples: List[Dict[str, Any]] = []
        self._t0 = None
        self._root_proc = psutil.Process(os.getpid())

    def _collect_process_tree(self) -> List[psutil.Process]:
        procs = [self._root_proc]
        try:
            children = self._root_proc.children(recursive=True)
            procs.extend(children)
        except Exception:
            pass
        return procs

    def _sample_once(self) -> None:
        elapsed = time.perf_counter() - self._t0

        # system
        system_cpu = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()

        # process tree RAM
        ram_bytes = 0
        proc_cpu_sum = 0.0
        proc_count = 0

        for proc in self._collect_process_tree():
            try:
                ram_bytes += proc.memory_info().rss
                proc_cpu_sum += proc.cpu_percent(interval=None)
                proc_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        gpu_stats = self.gpu.sample()

        self.samples.append({
            "elapsed_sec": elapsed,
            "process_tree_cpu_percent": proc_cpu_sum,
            "system_cpu_percent": system_cpu,
            "process_tree_ram_mb": ram_bytes / (1024 ** 2),
            "system_ram_percent": vm.percent,
            "gpu_util_percent": gpu_stats["gpu_util_percent"],
            "gpu_mem_used_mb": gpu_stats["gpu_mem_used_mb"],
            "gpu_mem_total_mb": gpu_stats["gpu_mem_total_mb"],
            "gpu_mem_percent": gpu_stats["gpu_mem_percent"],
            "process_count": proc_count,
        })

    def _run(self) -> None:
        # Warm-up cpu_percent counters
        for proc in self._collect_process_tree():
            try:
                proc.cpu_percent(interval=None)
            except Exception:
                pass
        psutil.cpu_percent(interval=None)

        while not self._stop_event.is_set():
            self._sample_once()
            time.sleep(self.interval_sec)

        # final sample
        self._sample_once()

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.gpu.shutdown()

    def save_csv(self, csv_path: Path) -> None:
        import csv

        if not self.samples:
            return

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.samples[0].keys()))
            writer.writeheader()
            writer.writerows(self.samples)

    def summary(self) -> Dict[str, Any]:
        if not self.samples:
            return {}

        def valid(vals):
            return [v for v in vals if v is not None]

        cpu_proc = [s["process_tree_cpu_percent"] for s in self.samples]
        cpu_sys = [s["system_cpu_percent"] for s in self.samples]
        ram_proc = [s["process_tree_ram_mb"] for s in self.samples]
        ram_sys = [s["system_ram_percent"] for s in self.samples]
        gpu_util = valid([s["gpu_util_percent"] for s in self.samples])
        gpu_mem = valid([s["gpu_mem_used_mb"] for s in self.samples])

        return {
            "sample_count": len(self.samples),
            "duration_sec_profiled": self.samples[-1]["elapsed_sec"],
            "process_tree_cpu_percent_avg": sum(cpu_proc) / len(cpu_proc),
            "process_tree_cpu_percent_max": max(cpu_proc),
            "system_cpu_percent_avg": sum(cpu_sys) / len(cpu_sys),
            "system_cpu_percent_max": max(cpu_sys),
            "process_tree_ram_mb_avg": sum(ram_proc) / len(ram_proc),
            "process_tree_ram_mb_max": max(ram_proc),
            "system_ram_percent_avg": sum(ram_sys) / len(ram_sys),
            "system_ram_percent_max": max(ram_sys),
            "gpu_util_percent_avg": (sum(gpu_util) / len(gpu_util)) if gpu_util else None,
            "gpu_util_percent_max": max(gpu_util) if gpu_util else None,
            "gpu_mem_used_mb_avg": (sum(gpu_mem) / len(gpu_mem)) if gpu_mem else None,
            "gpu_mem_used_mb_max": max(gpu_mem) if gpu_mem else None,
            "gpu_name": self.gpu.gpu_name,
        }


def make_profile_plot(samples: List[Dict[str, Any]], output_png: Path, title: str) -> None:
    if not samples:
        raise RuntimeError("No profiler samples were collected.")

    x = [s["elapsed_sec"] for s in samples]

    cpu_proc = [s["process_tree_cpu_percent"] for s in samples]
    cpu_sys = [s["system_cpu_percent"] for s in samples]

    ram_proc = [s["process_tree_ram_mb"] for s in samples]
    ram_sys = [s["system_ram_percent"] for s in samples]

    gpu_util = [0.0 if s["gpu_util_percent"] is None else s["gpu_util_percent"] for s in samples]
    gpu_mem = [0.0 if s["gpu_mem_used_mb"] is None else s["gpu_mem_used_mb"] for s in samples]

    fig = plt.figure(figsize=(16, 12))

    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(x, cpu_proc, label="Process tree CPU %")
    ax1.plot(x, cpu_sys, label="System CPU %")
    ax1.set_ylabel("CPU %")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(x, ram_proc, label="Process tree RAM (MB)")
    ax2.plot(x, ram_sys, label="System RAM %")
    ax2.set_ylabel("RAM")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(x, gpu_util, label="GPU Util %")
    ax3.plot(x, gpu_mem, label="GPU VRAM Used (MB)")
    ax3.set_ylabel("GPU")
    ax3.set_xlabel("Elapsed seconds")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_one_benchmark(
    expert_video: str,
    learner_video: str,
    expert_time_seconds: float,
    learner_time_seconds: float,
    expert_point_xy: List[int],
    learner_point_xy: List[int],
    analysis_mode: str,
    max_seconds: float,
    frame_stride: int,
    output_dir: Path,
    run_name: str,
    poll_interval: float,
    gpu_index: int,
) -> Dict[str, Any]:
    ensure_dir(output_dir)

    profiler = ResourceProfiler(interval_sec=poll_interval, gpu_index=gpu_index)

    started_at = now_iso()
    t0 = time.perf_counter()
    error_info = None
    result = None
    raw_json_bundle = None
    metrics_json_bundle = None

    try:
        profiler.start()

        result = track_two_videos_from_selected_points(
            expert_video_path=expert_video,
            learner_video_path=learner_video,
            expert_time_seconds=expert_time_seconds,
            learner_time_seconds=learner_time_seconds,
            expert_point_xy=expert_point_xy,
            learner_point_xy=learner_point_xy,
            analysis_mode=analysis_mode,
            max_seconds=max_seconds,
            frame_stride=frame_stride,
        )

        if "error" in result:
            raise RuntimeError(f"SAM2 pipeline returned error: {result['error']}")

        expert_raw_json_path = result["expert_video"]["raw_json_path"]
        learner_raw_json_path = result["learner_video"]["raw_json_path"]

        expert_metrics_json_path = export_derived_metrics_json(expert_raw_json_path)
        learner_metrics_json_path = export_derived_metrics_json(learner_raw_json_path)

        raw_json_bundle = {
            "expert_raw_json_path": expert_raw_json_path,
            "learner_raw_json_path": learner_raw_json_path,
            "expert_raw_json": load_json_file(expert_raw_json_path),
            "learner_raw_json": load_json_file(learner_raw_json_path),
        }

        metrics_json_bundle = {
            "expert_metrics_json_path": expert_metrics_json_path,
            "learner_metrics_json_path": learner_metrics_json_path,
            "expert_metrics_json": load_json_file(expert_metrics_json_path),
            "learner_metrics_json": load_json_file(learner_metrics_json_path),
        }

    except Exception as e:
        error_info = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
    finally:
        profiler.stop()

    elapsed = time.perf_counter() - t0
    ended_at = now_iso()

    profiler_csv = output_dir / f"{run_name}_resource_timeline.csv"
    profiler_png = output_dir / f"{run_name}_resource_plot.png"
    result_json_path = output_dir / f"{run_name}_result.json"
    raw_bundle_json_path = output_dir / f"{run_name}_raw_bundle.json"
    metrics_bundle_json_path = output_dir / f"{run_name}_metrics_bundle.json"
    summary_json_path = output_dir / f"{run_name}_benchmark_summary.json"

    profiler.save_csv(profiler_csv)
    make_profile_plot(profiler.samples, profiler_png, f"Benchmark: {run_name}")

    benchmark_summary = {
        "run_name": run_name,
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_time_sec": elapsed,
        "inputs": {
            "expert_video": expert_video,
            "learner_video": learner_video,
            "expert_time_seconds": expert_time_seconds,
            "learner_time_seconds": learner_time_seconds,
            "expert_point_xy": expert_point_xy,
            "learner_point_xy": learner_point_xy,
            "analysis_mode": analysis_mode,
            "max_seconds": max_seconds,
            "frame_stride": frame_stride,
            "gpu_index": gpu_index,
            "poll_interval": poll_interval,
        },
        "profiler_summary": profiler.summary(),
        "files": {
            "result_json": str(result_json_path),
            "raw_bundle_json": str(raw_bundle_json_path),
            "metrics_bundle_json": str(metrics_bundle_json_path),
            "resource_csv": str(profiler_csv),
            "resource_plot_png": str(profiler_png),
        },
        "pipeline_result_available": result is not None,
        "error": error_info,
    }

    if result is not None:
        json_safe_write(result_json_path, result)
    else:
        json_safe_write(result_json_path, {"error": error_info or "Unknown error"})

    if raw_json_bundle is not None:
        json_safe_write(raw_bundle_json_path, raw_json_bundle)
    else:
        json_safe_write(raw_bundle_json_path, {"error": error_info or "RAW JSON was not generated"})

    if metrics_json_bundle is not None:
        json_safe_write(metrics_bundle_json_path, metrics_json_bundle)
    else:
        json_safe_write(metrics_bundle_json_path, {"error": error_info or "Metrics JSON was not generated"})

    json_safe_write(summary_json_path, benchmark_summary)

    return benchmark_summary


def parse_xy(text: str) -> List[int]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Point must be in format: x,y")
    return [int(parts[0]), int(parts[1])]


def main():
    parser = argparse.ArgumentParser(
        description="Run SAM2 video tracking benchmark with CPU/RAM/GPU profiling and save the 3 JSON files."
    )

    parser.add_argument("--expert_video", required=True, help="Path to expert video")
    parser.add_argument("--learner_video", required=True, help="Path to learner video")

    parser.add_argument("--expert_time", type=float, default=1.0, help="Expert selected time in seconds")
    parser.add_argument("--learner_time", type=float, default=1.0, help="Learner selected time in seconds")

    parser.add_argument("--expert_point", type=parse_xy, required=True, help="Expert point as x,y")
    parser.add_argument("--learner_point", type=parse_xy, required=True, help="Learner point as x,y")

    parser.add_argument("--analysis_mode", choices=["first_n_seconds", "full"], default="first_n_seconds")
    parser.add_argument("--max_seconds", type=float, default=10.0)
    parser.add_argument("--frame_stride", type=int, default=3)

    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--poll_interval", type=float, default=0.5)

    parser.add_argument("--run_name", default="gpu_sam2_10s_stride3")
    parser.add_argument("--output_dir", default="benchmark_outputs")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    summary = run_one_benchmark(
        expert_video=args.expert_video,
        learner_video=args.learner_video,
        expert_time_seconds=args.expert_time,
        learner_time_seconds=args.learner_time,
        expert_point_xy=args.expert_point,
        learner_point_xy=args.learner_point,
        analysis_mode=args.analysis_mode,
        max_seconds=args.max_seconds,
        frame_stride=args.frame_stride,
        output_dir=output_dir,
        run_name=args.run_name,
        poll_interval=args.poll_interval,
        gpu_index=args.gpu_index,
    )

    print("\n=== BENCHMARK DONE ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()