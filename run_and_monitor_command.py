import argparse
import csv
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil


def get_process_tree(root: psutil.Process):
    try:
        return [root] + root.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []


def safe_name(proc: psutil.Process) -> str:
    try:
        return proc.name()
    except Exception:
        return "unknown"


def command_for_popen(command: str, use_shell: bool):
    if use_shell:
        return command
    if os.name == "nt":
        # Windows CreateProcess can parse the command line itself. This keeps
        # quoted paths with spaces intact without adding a cmd.exe wrapper.
        return command
    return shlex.split(command)


def drain_pipe(pipe, label: str, output_queue: queue.Queue[tuple[str, str]]) -> None:
    try:
        for line in iter(pipe.readline, ""):
            output_queue.put((label, line))
    finally:
        pipe.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch a command and monitor system CPU + process-tree RAM over time."
    )
    parser.add_argument("--command", required=True, help="Command to launch")
    parser.add_argument("--output-csv", default="run_monitor_log.csv", help="Output CSV path")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds")
    parser.add_argument(
        "--shell",
        action="store_true",
        help="Run the command through the shell. Only use this for shell builtins/operators.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Stop the command after this many seconds.",
    )
    args = parser.parse_args()

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        command_for_popen(args.command, args.shell),
        shell=args.shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    root_proc = psutil.Process(proc.pid)
    cpu_count = psutil.cpu_count(logical=True) or 1
    output_queue: queue.Queue[tuple[str, str]] = queue.Queue()
    stdout_tail: list[str] = []
    stderr_tail: list[str] = []

    stdout_thread = threading.Thread(
        target=drain_pipe, args=(proc.stdout, "STDOUT", output_queue), daemon=True
    )
    stderr_thread = threading.Thread(
        target=drain_pipe, args=(proc.stderr, "STDERR", output_queue), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()

    print(f"Launched PID {proc.pid}")
    print(f"Monitoring command: {args.command}")
    print("CPU metric = SYSTEM-WIDE CPU usage (%)")
    print("Memory metric = TARGET PROCESS TREE RAM (MB)\n")

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

        while True:
            time.sleep(args.interval)

            elapsed = time.time() - start_time
            while True:
                try:
                    label, line = output_queue.get_nowait()
                except queue.Empty:
                    break
                print(f"[{label}] {line}", end="")
                if label == "STDOUT":
                    stdout_tail.append(line)
                    stdout_tail = stdout_tail[-50:]
                else:
                    stderr_tail.append(line)
                    stderr_tail = stderr_tail[-50:]

            if args.timeout is not None and elapsed >= args.timeout and proc.poll() is None:
                print(f"\nTimeout reached after {elapsed:.1f}s. Terminating process tree...")
                for p in reversed(get_process_tree(root_proc)):
                    try:
                        p.terminate()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

            system_cpu = psutil.cpu_percent(interval=None)

            processes = get_process_tree(root_proc)
            rss_total_mb = 0.0
            names = []

            for p in processes:
                try:
                    rss_total_mb += p.memory_info().rss / (1024 * 1024)
                    names.append(f"{safe_name(p)}:{p.pid}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            writer.writerow(
                [
                    round(elapsed, 3),
                    proc.pid,
                    len(processes),
                    round(system_cpu, 2),
                    round(rss_total_mb, 2),
                    cpu_count,
                    ";".join(names),
                ]
            )
            f.flush()

            if proc.poll() is not None:
                break

    stdout_thread.join(timeout=2)
    stderr_thread.join(timeout=2)

    while True:
        try:
            label, line = output_queue.get_nowait()
        except queue.Empty:
            break
        print(f"[{label}] {line}", end="")
        if label == "STDOUT":
            stdout_tail.append(line)
            stdout_tail = stdout_tail[-50:]
        else:
            stderr_tail.append(line)
            stderr_tail = stderr_tail[-50:]

    print(f"CSV saved to: {output_csv}")
    print(f"Return code: {proc.returncode}")

    if stdout_tail:
        print("\n=== STDOUT tail ===")
        print("".join(stdout_tail))

    if stderr_tail:
        print("\n=== STDERR tail ===")
        print("".join(stderr_tail))

    if proc.returncode != 0:
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
