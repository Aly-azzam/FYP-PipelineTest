import argparse
import csv
import time
from pathlib import Path
from typing import List

import psutil


def get_process_tree(root: psutil.Process) -> List[psutil.Process]:
    try:
        return [root] + root.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []


def safe_name(proc: psutil.Process) -> str:
    try:
        return proc.name()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor an existing process by PID. Logs SYSTEM CPU and PROCESS-TREE RAM over time."
    )
    parser.add_argument("--pid", type=int, required=True, help="PID of the target process")
    parser.add_argument(
        "--output-csv",
        default="mediapipe_monitor_log.csv",
        help="Path to output CSV file",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds",
    )
    args = parser.parse_args()

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    try:
        root_proc = psutil.Process(args.pid)
        root_create_time = root_proc.create_time()
    except psutil.NoSuchProcess as exc:
        raise SystemExit(f"No process exists with PID {args.pid}") from exc
    cpu_count = psutil.cpu_count(logical=True) or 1

    print(f"Monitoring PID {args.pid} ({root_proc.name()})")
    print(f"Logical CPU count: {cpu_count}")
    print("CPU metric = SYSTEM-WIDE CPU usage (%)")
    print("Memory metric = TARGET PROCESS TREE RAM (MB)")
    print("Press Ctrl+C to stop.\n")

    # Prime system CPU measurement
    psutil.cpu_percent(interval=None)

    start_time = time.time()

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "elapsed_seconds",
                "root_pid",
                "process_count",
                "system_cpu_percent",
                "process_tree_rss_memory_mb",
                "logical_cpu_count",
                "process_names",
            ]
        )

        try:
            while True:
                time.sleep(args.interval)

                now = time.time()
                elapsed = now - start_time
                try:
                    if (
                        not root_proc.is_running()
                        or root_proc.create_time() != root_create_time
                    ):
                        print("\nTarget process exited. Monitoring stopped.")
                        break
                except psutil.NoSuchProcess:
                    print("\nTarget process exited. Monitoring stopped.")
                    break

                # Correct CPU measurement: system-wide usage over the last interval
                system_cpu = psutil.cpu_percent(interval=None)

                processes = get_process_tree(root_proc)
                if not processes:
                    print("\nTarget process exited. Monitoring stopped.")
                    break

                rss_total_mb = 0.0
                names = []

                for proc in processes:
                    try:
                        rss_total_mb += proc.memory_info().rss / (1024 * 1024)
                        names.append(f"{safe_name(proc)}:{proc.pid}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                writer.writerow(
                    [
                        round(elapsed, 3),
                        args.pid,
                        len(processes),
                        round(system_cpu, 2),
                        round(rss_total_mb, 2),
                        cpu_count,
                        ";".join(names),
                    ]
                )
                f.flush()

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")

    print(f"CSV saved to: {output_csv}")


if __name__ == "__main__":
    main()
