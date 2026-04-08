from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp

from backend.components.mediapipe.utils import (
    bgr_to_rgb,
    interpolate_single_frame_wrist_gaps,
    landmark_to_list,
    normalize_handedness_label,
    reject_impossible_wrist_jumps,
)


mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

_WRIST_SMOOTHING_ALPHA = 0.35
_DEFAULT_POSE_VISIBILITY_THRESHOLD = 0.5
_ANNOTATION_OUTPUT_CANDIDATES = (
    (".mp4", "avc1"),
    (".mp4", "H264"),
    (".webm", "VP80"),
    (".webm", "VP90"),
    (".mp4", "mp4v"),
    (".avi", "MJPG"),
    (".avi", "XVID"),
)


def _draw_wrist_marker(frame, wrist_coords: list[float], label: str, color) -> None:
    height, width = frame.shape[:2]
    x_px = int(wrist_coords[0] * width)
    y_px = int(wrist_coords[1] * height)
    cv2.circle(frame, (x_px, y_px), 8, color, -1)
    cv2.putText(
        frame,
        label,
        (x_px + 10, y_px - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def _create_video_writer(
    output_path: str,
    output_fps: float,
    width: int,
    height: int,
):
    requested_path = Path(output_path)
    for extension, codec in _ANNOTATION_OUTPUT_CANDIDATES:
        candidate_path = requested_path.with_suffix(extension)
        writer = cv2.VideoWriter(
            str(candidate_path),
            cv2.VideoWriter_fourcc(*codec),
            output_fps,
            (width, height),
        )
        if writer.isOpened():
            return writer, codec, str(candidate_path)
        writer.release()
    return None, None, None


def _smooth_wrist_coords(
    wrist_coords: list[float],
    previous_coords: list[float] | None,
) -> list[float]:
    if previous_coords is None:
        return wrist_coords

    smoothed: list[float] = []
    for current, previous in zip(wrist_coords, previous_coords):
        value = (_WRIST_SMOOTHING_ALPHA * current) + (
            (1 - _WRIST_SMOOTHING_ALPHA) * previous
        )
        smoothed.append(float(value))
    return smoothed


def _extract_wrist_from_hand_results(results) -> dict[str, Any]:
    frame_result: dict[str, Any] = {
        "left_wrist": None,
        "right_wrist": None,
        "left_wrist_visible": False,
        "right_wrist_visible": False,
    }

    if not results.multi_hand_landmarks or not results.multi_handedness:
        return frame_result

    for hand_landmarks, handedness in zip(
        results.multi_hand_landmarks,
        results.multi_handedness,
    ):
        label = normalize_handedness_label(handedness.classification[0].label)
        wrist_landmark = hand_landmarks.landmark[0]
        wrist_coords = landmark_to_list(wrist_landmark)

        if label == "Left":
            frame_result["left_wrist"] = wrist_coords
            frame_result["left_wrist_visible"] = True
        elif label == "Right":
            frame_result["right_wrist"] = wrist_coords
            frame_result["right_wrist_visible"] = True

    return frame_result


def _extract_wrist_from_pose_results(
    results,
    visibility_threshold: float,
) -> dict[str, Any]:
    frame_result: dict[str, Any] = {
        "left_wrist": None,
        "right_wrist": None,
        "left_wrist_visible": False,
        "right_wrist_visible": False,
    }

    if not getattr(results, "pose_landmarks", None):
        return frame_result

    landmarks = results.pose_landmarks.landmark
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    left_visibility = float(getattr(left_wrist, "visibility", 0.0))
    right_visibility = float(getattr(right_wrist, "visibility", 0.0))

    if left_visibility > visibility_threshold:
        frame_result["left_wrist"] = landmark_to_list(left_wrist)
        frame_result["left_wrist_visible"] = True
    if right_visibility > visibility_threshold:
        frame_result["right_wrist"] = landmark_to_list(right_wrist)
        frame_result["right_wrist_visible"] = True

    return frame_result


def _merge_pose_and_hand_wrist_results(
    pose_result: dict[str, Any],
    hand_result: dict[str, Any],
) -> dict[str, Any]:
    frame_result: dict[str, Any] = {
        "left_wrist": None,
        "right_wrist": None,
        "left_wrist_visible": False,
        "right_wrist_visible": False,
        "left_wrist_source": "none",
        "right_wrist_source": "none",
        "wrist_source": "none",
    }

    for side in ("left", "right"):
        wrist_key = f"{side}_wrist"
        visible_key = f"{side}_wrist_visible"
        source_key = f"{side}_wrist_source"

        if pose_result.get(visible_key) and pose_result.get(wrist_key):
            frame_result[wrist_key] = pose_result[wrist_key]
            frame_result[visible_key] = True
            frame_result[source_key] = "pose"
        elif hand_result.get(visible_key) and hand_result.get(wrist_key):
            frame_result[wrist_key] = hand_result[wrist_key]
            frame_result[visible_key] = True
            frame_result[source_key] = "hands"

    if (
        frame_result["left_wrist_source"] == "pose"
        or frame_result["right_wrist_source"] == "pose"
    ):
        frame_result["wrist_source"] = "pose"
    elif (
        frame_result["left_wrist_source"] == "hands"
        or frame_result["right_wrist_source"] == "hands"
    ):
        frame_result["wrist_source"] = "hands"

    return frame_result


def extract_wrist_from_frame(
    frame,
    hands_processor,
    pose_processor,
    pose_visibility_threshold: float = _DEFAULT_POSE_VISIBILITY_THRESHOLD,
) -> dict[str, Any]:
    """
    Extract left/right wrist coordinates from a single frame.

    MediaPipe Hands landmark index 0 = wrist.
    Returns a dictionary with:
    - left_wrist
    - right_wrist
    - left_wrist_visible
    - right_wrist_visible
    """
    rgb_frame = bgr_to_rgb(frame)
    hand_results = hands_processor.process(rgb_frame)
    pose_results = pose_processor.process(rgb_frame)
    hand_frame_result = _extract_wrist_from_hand_results(hand_results)
    pose_frame_result = _extract_wrist_from_pose_results(
        pose_results,
        pose_visibility_threshold,
    )
    return _merge_pose_and_hand_wrist_results(
        pose_result=pose_frame_result,
        hand_result=hand_frame_result,
    )


def _draw_hand_landmarks(frame, results) -> None:
    if not results.multi_hand_landmarks:
        return

    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )


def extract_wrist_from_video(
    video_path: str,
    annotated_output_path: str | None = None,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    model_complexity: int = 1,
    pose_visibility_threshold: float = _DEFAULT_POSE_VISIBILITY_THRESHOLD,
) -> dict[str, Any]:
    """
    Extract wrist data frame by frame from a video.

    Returns a structured dictionary containing:
    - filename
    - path
    - fps
    - frame_count
    - duration_sec
    - frames: list of per-frame wrist results
    """
    path_obj = Path(video_path)

    cap = cv2.VideoCapture(str(path_obj))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    duration_sec = 0.0
    if fps > 0:
        duration_sec = frame_count / fps

    frames_output: list[dict[str, Any]] = []
    writer = None
    selected_codec = None
    previous_wrists = {
        "Left": None,
        "Right": None,
    }
    source_counts = {
        "pose": 0,
        "hands": 0,
        "none": 0,
    }

    if annotated_output_path and width > 0 and height > 0:
        output_fps = fps if fps > 0 else 30.0
        writer, selected_codec, selected_output_path = _create_video_writer(
            str(annotated_output_path),
            output_fps,
            width,
            height,
        )
        if writer is None:
            annotated_output_path = None
        else:
            annotated_output_path = selected_output_path

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as hands_processor, mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as pose_processor:
        frame_index = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            timestamp_sec = frame_index / fps if fps > 0 else None

            rgb_frame = bgr_to_rgb(frame)
            hand_results = hands_processor.process(rgb_frame)
            pose_results = pose_processor.process(rgb_frame)
            hand_frame_result = _extract_wrist_from_hand_results(hand_results)
            pose_frame_result = _extract_wrist_from_pose_results(
                pose_results,
                pose_visibility_threshold,
            )
            frame_result = _merge_pose_and_hand_wrist_results(
                pose_result=pose_frame_result,
                hand_result=hand_frame_result,
            )

            source_counts[frame_result["wrist_source"]] += 1

            if frame_result["left_wrist_visible"] and frame_result["left_wrist"]:
                frame_result["left_wrist"] = _smooth_wrist_coords(
                    frame_result["left_wrist"],
                    previous_wrists["Left"],
                )
                previous_wrists["Left"] = frame_result["left_wrist"]
            else:
                previous_wrists["Left"] = None

            if frame_result["right_wrist_visible"] and frame_result["right_wrist"]:
                frame_result["right_wrist"] = _smooth_wrist_coords(
                    frame_result["right_wrist"],
                    previous_wrists["Right"],
                )
                previous_wrists["Right"] = frame_result["right_wrist"]
            else:
                previous_wrists["Right"] = None

            frame_result["frame_index"] = frame_index
            frame_result["timestamp_sec"] = timestamp_sec

            if writer is not None:
                annotated_frame = frame.copy()
                _draw_hand_landmarks(annotated_frame, hand_results)
                if frame_result["left_wrist_visible"] and frame_result["left_wrist"]:
                    _draw_wrist_marker(
                        annotated_frame,
                        frame_result["left_wrist"],
                        "Left wrist",
                        (80, 200, 120),
                    )
                if frame_result["right_wrist_visible"] and frame_result["right_wrist"]:
                    _draw_wrist_marker(
                        annotated_frame,
                        frame_result["right_wrist"],
                        "Right wrist",
                        (80, 120, 240),
                    )
                writer.write(annotated_frame)

            frames_output.append(frame_result)
            frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()

    rejected_left_wrist_jumps = reject_impossible_wrist_jumps(
        frames_output,
        wrist_key="left_wrist",
        visible_key="left_wrist_visible",
        max_jump_distance=0.15,
    )
    rejected_right_wrist_jumps = reject_impossible_wrist_jumps(
        frames_output,
        wrist_key="right_wrist",
        visible_key="right_wrist_visible",
        max_jump_distance=0.15,
    )

    interpolation_counts = interpolate_single_frame_wrist_gaps(frames_output)

    return {
        "filename": path_obj.name,
        "path": str(path_obj),
        "annotated_path": annotated_output_path,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "width": width,
        "height": height,
        "mediapipe_config": {
            "min_detection_confidence": min_detection_confidence,
            "min_tracking_confidence": min_tracking_confidence,
            "model_complexity": model_complexity,
            "pose_visibility_threshold": pose_visibility_threshold,
            "wrist_source_priority": ["pose", "hands", "none"],
            "annotation_codec": selected_codec if annotated_output_path else None,
            "annotation_extension": (
                Path(annotated_output_path).suffix if annotated_output_path else None
            ),
        },
        "wrist_source_counts": source_counts,
        "rejected_left_wrist_jumps": rejected_left_wrist_jumps,
        "rejected_right_wrist_jumps": rejected_right_wrist_jumps,
        "interpolated_left_wrist_frames": interpolation_counts[
            "interpolated_left_wrist_frames"
        ],
        "interpolated_right_wrist_frames": interpolation_counts[
            "interpolated_right_wrist_frames"
        ],
        "frames": frames_output,
    }
