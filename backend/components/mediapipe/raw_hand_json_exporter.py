"""
Export MediaPipe Hands per-frame data to JSON matching the Fast-HaMeR raw hand schema.

Runs its own MediaPipe Hands pass — does NOT modify existing pipeline logic.
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp

from backend.components.mediapipe.utils import bgr_to_rgb, normalize_handedness_label

mp_hands = mp.solutions.hands

_NUM_JOINTS = 21


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _side_label(raw_label: str) -> str:
    normalized = normalize_handedness_label(raw_label)
    return {"Left": "left", "Right": "right"}.get(normalized, "unknown")


def _joints_2d_pixels(hand_landmarks, w: int, h: int) -> list[list[float]]:
    return [
        [round(float(lm.x * w), 2), round(float(lm.y * h), 2)]
        for lm in hand_landmarks.landmark
    ]


def _bbox_from_landmarks(hand_landmarks, w: int, h: int) -> list[float]:
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]
    return [round(min(xs), 2), round(min(ys), 2),
            round(max(xs), 2), round(max(ys), 2)]


def _joints_3d_world(world_landmarks) -> list[list[float]]:
    return [
        [float(lm.x), float(lm.y), float(lm.z)]
        for lm in world_landmarks.landmark
    ]


def _wrist_3d(world_landmarks) -> list[float]:
    w = world_landmarks.landmark[0]
    return [float(w.x), float(w.y), float(w.z)]


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_raw_hand_document(
    video_path: str,
    run_id: str,
    *,
    max_frames: int | None = None,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    model_complexity: int = 1,
) -> dict[str, Any]:
    """Process *video_path* with MediaPipe Hands and return the HaMeR-shaped dict."""

    path_obj = Path(video_path)
    cap = cv2.VideoCapture(str(path_obj))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = (total / fps) if fps > 0 else 0.0

    frames_out: list[dict[str, Any]] = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as hands:
        idx = 0
        while True:
            if max_frames is not None and idx >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break

            ts = (idx / fps) if fps > 0 else None
            rgb = bgr_to_rgb(frame)
            results = hands.process(rgb)

            multi_lm = results.multi_hand_landmarks or []
            multi_cls = results.multi_handedness or []
            multi_world = results.multi_hand_world_landmarks or []

            hands_payload: list[dict[str, Any]] = []
            for hi, hand_lm in enumerate(multi_lm):
                cls = multi_cls[hi].classification[0]
                conf = float(cls.score)

                world = multi_world[hi] if hi < len(multi_world) else None

                hands_payload.append({
                    "hand_index": hi,
                    "hand_side": _side_label(cls.label or ""),
                    "bbox_xyxy": _bbox_from_landmarks(hand_lm, w, h),
                    "bbox_confidence": round(conf, 4),
                    "joints_2d": _joints_2d_pixels(hand_lm, w, h),
                    "joints_2d_scores": [round(conf, 4)] * _NUM_JOINTS,
                    "joints_3d_cam": _joints_3d_world(world) if world else None,
                    "cam_t_full": _wrist_3d(world) if world else None,
                })

            frames_out.append({
                "frame_index": idx,
                "timestamp_sec": round(ts, 6) if ts is not None else None,
                "hands_detected": len(hands_payload),
                "hands": hands_payload,
            })
            idx += 1

    cap.release()

    return {
        "run": {
            "run_id": run_id,
            "pipeline_name": "mediapipe",
            "model_name": "mediapipe_hands",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fps_used_for_processing": fps,
            "total_frames": len(frames_out),
            "raw_schema_version": "1.0",
        },
        "video_metadata": {
            "filename": path_obj.name,
            "path": str(path_obj.resolve()),
            "width": w,
            "height": h,
            "fps": fps,
            "duration_sec": round(duration, 3),
        },
        "frames": frames_out,
    }


# ---------------------------------------------------------------------------
# Public export function
# ---------------------------------------------------------------------------

def export_mediapipe_raw_hand_json(
    video_path: str,
    run_id: str | None = None,
    *,
    output_dir: Path | None = None,
    max_frames: int | None = None,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    model_complexity: int = 1,
) -> Path:
    """
    Build the raw-hand document and write
    ``mediapipe_<run_id>_raw_hand_data.json`` under ``outputs/json/``.
    """
    rid = run_id or uuid.uuid4().hex
    out_dir = output_dir or (_project_root() / "outputs" / "json")
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = build_raw_hand_document(
        video_path,
        rid,
        max_frames=max_frames,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        model_complexity=model_complexity,
    )

    dest = out_dir / f"mediapipe_{rid}_raw_hand_data.json"
    dest.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return dest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export MediaPipe raw hand JSON (HaMeR-compatible schema).",
    )
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Process only the first N frames (useful for quick tests)",
    )
    args = parser.parse_args()

    out = export_mediapipe_raw_hand_json(
        args.video, run_id=args.run_id, max_frames=args.max_frames,
    )
    print("Wrote:", out)

    doc = json.loads(out.read_text(encoding="utf-8"))
    example = next(
        (f for f in doc.get("frames", []) if f.get("hands_detected", 0) > 0),
        None,
    )
    if example is None:
        print("\nNo frame with detected hands found.")
    else:
        print("\n--- Example frame (first with hands detected) ---")
        print(json.dumps(example, indent=2))


if __name__ == "__main__":
    main()
