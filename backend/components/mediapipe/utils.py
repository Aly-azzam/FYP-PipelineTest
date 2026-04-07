from __future__ import annotations

import math


def bgr_to_rgb(frame):
    """
    Convert an OpenCV BGR frame to RGB.
    MediaPipe expects RGB input.
    """
    return frame[:, :, ::-1]


def landmark_to_list(landmark) -> list[float]:
    """
    Convert a MediaPipe landmark object to a simple [x, y, z] list.
    """
    return [float(landmark.x), float(landmark.y), float(landmark.z)]


def normalize_handedness_label(label: str) -> str:
    """
    Normalize MediaPipe handedness labels to stable values.
    Expected values are usually 'Left' or 'Right'.
    """
    label = (label or "").strip().lower()

    if label == "left":
        return "Left"
    if label == "right":
        return "Right"

    return "Unknown"


def reject_impossible_wrist_jumps(
    frames: list[dict],
    wrist_key: str,
    visible_key: str,
    max_jump_distance: float = 0.15,
) -> int:
    """
    Reject impossible frame-to-frame wrist jumps for one hand.

    If the Euclidean distance between the previous valid wrist and current
    wrist is greater than max_jump_distance, current wrist is rejected.
    """
    rejected_count = 0
    previous_valid_wrist: list[float] | None = None

    for frame in frames:
        if not frame.get(visible_key) or not frame.get(wrist_key):
            continue

        current_wrist = frame[wrist_key]
        if previous_valid_wrist is None:
            previous_valid_wrist = current_wrist
            continue

        dimensions = min(len(previous_valid_wrist), len(current_wrist))
        if dimensions == 0:
            frame[wrist_key] = None
            frame[visible_key] = False
            rejected_count += 1
            continue

        distance = math.sqrt(
            sum(
                (float(current_wrist[i]) - float(previous_valid_wrist[i])) ** 2
                for i in range(dimensions)
            )
        )

        if distance > max_jump_distance:
            frame[wrist_key] = None
            frame[visible_key] = False
            rejected_count += 1
            continue

        previous_valid_wrist = current_wrist

    return rejected_count


def interpolate_single_frame_wrist_gaps(
    frames: list[dict],
) -> dict[str, int]:
    """
    Fill only single-frame wrist gaps.

    For each side (left/right):
    - previous frame has wrist visible
    - current frame wrist is missing
    - next frame has wrist visible
    Then current frame wrist is interpolated as the average of previous and next.
    """
    if len(frames) < 3:
        return {
            "interpolated_left_wrist_frames": 0,
            "interpolated_right_wrist_frames": 0,
        }

    counts = {
        "interpolated_left_wrist_frames": 0,
        "interpolated_right_wrist_frames": 0,
    }

    side_specs = [
        ("left_wrist", "left_wrist_visible", "interpolated_left_wrist_frames"),
        ("right_wrist", "right_wrist_visible", "interpolated_right_wrist_frames"),
    ]

    for i in range(1, len(frames) - 1):
        prev_frame = frames[i - 1]
        current_frame = frames[i]
        next_frame = frames[i + 1]

        for wrist_key, visible_key, count_key in side_specs:
            prev_visible = bool(prev_frame.get(visible_key))
            curr_visible = bool(current_frame.get(visible_key))
            next_visible = bool(next_frame.get(visible_key))

            if curr_visible or not (prev_visible and next_visible):
                continue

            prev_wrist = prev_frame.get(wrist_key)
            next_wrist = next_frame.get(wrist_key)
            if not prev_wrist or not next_wrist:
                continue

            interpolated = [
                float((prev_wrist[j] + next_wrist[j]) / 2.0)
                for j in range(min(len(prev_wrist), len(next_wrist)))
            ]
            if len(interpolated) < 3:
                continue

            current_frame[wrist_key] = interpolated
            current_frame[visible_key] = True
            counts[count_key] += 1

    return counts
