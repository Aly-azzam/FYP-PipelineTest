"""
Build a MediaPipe summary JSON similar to Fast-HaMeR summary output.

Input: derived metrics JSON (and referenced raw JSON)
Output: mediapipe_<run_id>.json under outputs/json/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _extract_run_id(derived_doc: dict[str, Any], derived_path: Path) -> str:
    run_id = str(derived_doc.get("run", {}).get("run_id", "")).strip()
    if run_id:
        return run_id

    stem = derived_path.stem
    prefix = "mediapipe_"
    suffix = "_derived_metrics"
    if stem.startswith(prefix) and stem.endswith(suffix):
        return stem[len(prefix) : -len(suffix)]
    return "unknown_run"


def _default_output_video(input_video: dict[str, Any]) -> dict[str, Any]:
    return {
        "filename": None,
        "path": None,
        "duration_sec": input_video.get("duration_sec"),
        "fps": input_video.get("fps"),
        "width": input_video.get("width"),
        "height": input_video.get("height"),
    }


def build_mediapipe_summary_document(
    derived_doc: dict[str, Any],
    derived_path: Path,
    raw_doc: dict[str, Any],
) -> dict[str, Any]:
    run_id = _extract_run_id(derived_doc, derived_path)
    raw_run = raw_doc.get("run", {}) or {}
    raw_video = raw_doc.get("video_metadata", {}) or {}

    coverage = (
        derived_doc.get("summary_metrics", {})
        .get("coverage", {})
        or {}
    )
    total_frames = coverage.get("total_frames")
    detected_frames = coverage.get("frames_with_detection")
    detection_rate = coverage.get("detection_rate")

    explanation_text = (
        "MediaPipe processed "
        f"{total_frames if total_frames is not None else 'unknown'} frames. "
        f"{detected_frames if detected_frames is not None else 'unknown'} frames had hand detections "
        f"(detection rate: {detection_rate if detection_rate is not None else 'unknown'}). "
        "Comparison metrics are not computed yet."
    )

    return {
        "run": {
            "run_id": run_id,
            "pipeline_name": "mediapipe",
            "processing_time_sec": None,
            "created_at": raw_run.get("created_at"),
            "component_notes": {
                "raw_source": "mediapipe_raw_hand_data",
                "derived_source": "mediapipe_derived_metrics",
                "frames_processed": total_frames,
                "frames_with_detection": detected_frames,
                "detection_rate": detection_rate,
            },
        },
        "input_video": {
            "filename": raw_video.get("filename"),
            "path": raw_video.get("path"),
            "duration_sec": raw_video.get("duration_sec"),
            "fps": raw_video.get("fps"),
            "width": raw_video.get("width"),
            "height": raw_video.get("height"),
        },
        "output_video": _default_output_video(raw_video),
        "overall_score": None,
        "metrics": {
            "joint_angle_deviation": None,
            "trajectory_deviation": None,
            "velocity_difference": None,
            "tool_alignment_deviation": None,
            "dtw_cost": None,
            "semantic_similarity": None,
            "optical_flow_similarity": None,
            "extra": None,
        },
        "confidences": {
            "overall": None,
            "same_task": None,
            "score": None,
            "explanation": None,
        },
        "explanation": {
            "text": explanation_text,
            "strengths": [
                "Provides frame-level hand detection coverage statistics",
                "Includes wrist kinematic metrics in derived outputs",
            ],
            "weaknesses": [
                "No comparison score is computed yet",
                "Task-level semantic explanation is not implemented",
            ],
            "raw_vlm_output": None,
            "structured_notes": {
                "frames_processed": total_frames,
                "frames_with_detection": detected_frames,
                "detection_rate": detection_rate,
            },
        },
        "warnings": [
            "overall_score is null — comparison scoring is not implemented",
            "All metrics fields are null — comparison metrics are not computed yet",
            "No semantic or VLM-based analysis was performed",
            "output_video path is null — summary built from raw/derived JSON only",
        ],
    }


def export_mediapipe_summary_json(
    derived_json_path: str,
    *,
    output_dir: Path | None = None,
) -> Path:
    derived_path = Path(derived_json_path)
    derived_doc = json.loads(derived_path.read_text(encoding="utf-8"))

    raw_path_str = (
        derived_doc.get("source_files", {})
        .get("raw_hand_data_json")
    )
    if not raw_path_str:
        raise ValueError(
            "Derived JSON does not contain source_files.raw_hand_data_json"
        )

    raw_path = Path(raw_path_str)
    raw_doc = json.loads(raw_path.read_text(encoding="utf-8"))

    summary_doc = build_mediapipe_summary_document(
        derived_doc=derived_doc,
        derived_path=derived_path,
        raw_doc=raw_doc,
    )

    run_id = _extract_run_id(derived_doc, derived_path)
    out_dir = output_dir or (_project_root() / "outputs" / "json")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mediapipe_{run_id}.json"
    out_path.write_text(json.dumps(summary_doc, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export MediaPipe summary JSON from derived metrics JSON."
    )
    parser.add_argument(
        "derived_json",
        help="Path to mediapipe_<run_id>_derived_metrics.json"
    )
    args = parser.parse_args()

    out_path = export_mediapipe_summary_json(args.derived_json)
    print("Wrote:", out_path)

    summary_doc = json.loads(out_path.read_text(encoding="utf-8"))
    print("\n--- Summary explanation ---")
    print(summary_doc.get("explanation", {}).get("text"))


if __name__ == "__main__":
    main()
