"""
Compute derived wrist metrics from MediaPipe raw hand JSON.

Reads an existing ``mediapipe_<run_id>_raw_hand_data.json`` and writes
``mediapipe_<run_id>_derived_metrics.json`` under ``outputs/json/``.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _extract_run_id(raw_doc: dict[str, Any], raw_path: Path) -> str:
    run_id = str(raw_doc.get("run", {}).get("run_id", "")).strip()
    if run_id:
        return run_id

    stem = raw_path.stem
    prefix = "mediapipe_"
    suffix = "_raw_hand_data"
    if stem.startswith(prefix) and stem.endswith(suffix):
        return stem[len(prefix) : -len(suffix)]
    return "unknown_run"


def _euclidean(a: list[float], b: list[float]) -> float:
    dims = min(len(a), len(b))
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(dims)))


def _wrist_from_hand(hand: dict[str, Any]) -> list[float] | None:
    cam_t_full = hand.get("cam_t_full")
    if isinstance(cam_t_full, list) and len(cam_t_full) >= 3:
        return [float(cam_t_full[0]), float(cam_t_full[1]), float(cam_t_full[2])]

    joints_3d = hand.get("joints_3d_cam")
    if isinstance(joints_3d, list) and joints_3d and isinstance(joints_3d[0], list):
        wrist = joints_3d[0]
        if len(wrist) >= 3:
            return [float(wrist[0]), float(wrist[1]), float(wrist[2])]

    return None


def _coverage_metrics(frames: list[dict[str, Any]]) -> dict[str, Any]:
    total_frames = len(frames)
    detected_indices: list[int] = []
    longest_streak = 0
    current_streak = 0

    for idx, frame in enumerate(frames):
        detected = int(frame.get("hands_detected", 0)) > 0
        if detected:
            detected_indices.append(idx)
            current_streak += 1
            if current_streak > longest_streak:
                longest_streak = current_streak
        else:
            current_streak = 0

    frames_with_detection = len(detected_indices)
    frames_without_detection = total_frames - frames_with_detection
    detection_rate = (
        (frames_with_detection / total_frames) if total_frames > 0 else 0.0
    )

    return {
        "total_frames": total_frames,
        "frames_with_detection": frames_with_detection,
        "frames_without_detection": frames_without_detection,
        "detection_rate": round(detection_rate, 6),
        "longest_detection_streak": longest_streak,
        "first_detected_frame": detected_indices[0] if detected_indices else None,
        "last_detected_frame": detected_indices[-1] if detected_indices else None,
    }


def _compute_per_hand_and_per_frame(
    frames: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, float | None]], list[dict[str, Any]]]:
    per_frame_metrics: list[dict[str, Any]] = []
    sides = ("left", "right")

    previous_position: dict[str, list[float] | None] = {s: None for s in sides}
    previous_timestamp: dict[str, float | None] = {s: None for s in sides}
    previous_speed: dict[str, float | None] = {s: None for s in sides}

    speed_values: dict[str, list[float]] = {s: [] for s in sides}
    accel_values: dict[str, list[float]] = {s: [] for s in sides}
    trajectory_lengths: dict[str, float] = {s: 0.0 for s in sides}

    for frame in frames:
        timestamp = frame.get("timestamp_sec")
        timestamp_float = float(timestamp) if isinstance(timestamp, (int, float)) else None
        hands = frame.get("hands", []) or []

        side_to_wrist: dict[str, list[float] | None] = {s: None for s in sides}
        for hand in hands:
            side = str(hand.get("hand_side", "")).strip().lower()
            if side not in side_to_wrist:
                continue
            side_to_wrist[side] = _wrist_from_hand(hand)

        for side in sides:
            wrist_position = side_to_wrist[side]
            wrist_velocity: list[float] | None = None
            wrist_speed: float | None = None
            wrist_acceleration: float | None = None

            prev_pos = previous_position[side]
            prev_time = previous_timestamp[side]
            prev_speed = previous_speed[side]

            dt = None
            if timestamp_float is not None and prev_time is not None:
                candidate_dt = timestamp_float - prev_time
                if candidate_dt > 0:
                    dt = candidate_dt

            if wrist_position is not None and prev_pos is not None and dt is not None:
                dims = min(len(wrist_position), len(prev_pos))
                wrist_velocity = [
                    float((wrist_position[i] - prev_pos[i]) / dt) for i in range(dims)
                ]
                wrist_speed = _euclidean(wrist_position, prev_pos) / dt
                trajectory_lengths[side] += _euclidean(wrist_position, prev_pos)
                speed_values[side].append(wrist_speed)

                if prev_speed is not None:
                    wrist_acceleration = abs((wrist_speed - prev_speed) / dt)
                    accel_values[side].append(wrist_acceleration)

            per_frame_metrics.append(
                {
                    "frame_index": int(frame.get("frame_index", 0)),
                    "hand_side": side,
                    "timestamp_sec": timestamp_float,
                    "wrist_position": wrist_position,
                    "wrist_velocity": wrist_velocity,
                    "wrist_speed": wrist_speed,
                    "wrist_acceleration": wrist_acceleration,
                }
            )

            previous_position[side] = wrist_position
            previous_timestamp[side] = timestamp_float
            previous_speed[side] = wrist_speed

    per_hand_metrics: dict[str, dict[str, float | None]] = {}
    for side in sides:
        speeds = speed_values[side]
        accels = accel_values[side]
        per_hand_metrics[side] = {
            "mean_speed": round(sum(speeds) / len(speeds), 6) if speeds else None,
            "max_speed": round(max(speeds), 6) if speeds else None,
            "mean_acceleration": round(sum(accels) / len(accels), 6) if accels else None,
            "max_acceleration": round(max(accels), 6) if accels else None,
            "trajectory_length": round(trajectory_lengths[side], 6),
        }

    return per_hand_metrics, per_frame_metrics


def build_derived_metrics_document(raw_doc: dict[str, Any], raw_path: Path) -> dict[str, Any]:
    frames = raw_doc.get("frames", []) or []
    coverage = _coverage_metrics(frames)
    per_hand_metrics, per_frame_metrics = _compute_per_hand_and_per_frame(frames)
    run_id = _extract_run_id(raw_doc, raw_path)

    return {
        "run": {
            "run_id": run_id,
            "pipeline_name": "mediapipe",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "derived_schema_version": "1.0",
        },
        "source_files": {
            "raw_hand_data_json": str(raw_path.resolve()),
        },
        "summary_metrics": {
            "coverage": coverage,
            "per_hand_wrist_metrics": per_hand_metrics,
        },
        "per_frame_metrics": per_frame_metrics,
    }


def export_mediapipe_derived_metrics_json(
    raw_json_path: str,
    *,
    output_dir: Path | None = None,
) -> Path:
    raw_path = Path(raw_json_path)
    raw_doc = json.loads(raw_path.read_text(encoding="utf-8"))
    derived_doc = build_derived_metrics_document(raw_doc, raw_path)

    run_id = _extract_run_id(raw_doc, raw_path)
    out_dir = output_dir or (_project_root() / "outputs" / "json")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mediapipe_{run_id}_derived_metrics.json"
    out_path.write_text(json.dumps(derived_doc, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export derived wrist metrics from MediaPipe raw hand JSON.",
    )
    parser.add_argument(
        "raw_json",
        help="Path to mediapipe_<run_id>_raw_hand_data.json",
    )
    args = parser.parse_args()

    out_path = export_mediapipe_derived_metrics_json(args.raw_json)
    print("Wrote:", out_path)

    derived_doc = json.loads(out_path.read_text(encoding="utf-8"))
    example = next(
        (item for item in derived_doc.get("per_frame_metrics", []) if item.get("wrist_position")),
        None,
    )
    if example is None:
        print("\nNo per-frame wrist sample with detection found.")
    else:
        print("\n--- Example per_frame_metrics entry ---")
        print(json.dumps(example, indent=2))


if __name__ == "__main__":
    main()
