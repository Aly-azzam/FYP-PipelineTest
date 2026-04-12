import cv2
import numpy as np
from pathlib import Path


def sample_uniform_frames(
    video_path: str,
    num_frames: int = 64,
    start_frame: int | None = None,
    end_frame: int | None = None,
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(str(Path(video_path)))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        raise ValueError(f"Could not read video: {video_path}")

    start = 0 if start_frame is None else max(0, int(start_frame))
    end = total - 1 if end_frame is None else min(total - 1, int(end_frame))
    if end < start:
        cap.release()
        raise ValueError(
            f"Invalid frame range for {video_path}: start={start}, end={end}"
        )

    indices = np.linspace(start, end, num_frames).astype(int)
    wanted = set(indices.tolist())

    frames = []
    current = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if current in wanted:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            if len(frames) == num_frames:
                break

        current += 1

    cap.release()

    if len(frames) != num_frames:
        raise ValueError(
            f"Expected {num_frames} sampled frames, got {len(frames)} for {video_path}"
        )

    sampled_frames = np.stack(frames, axis=0)  # [T, H, W, C]
    if return_indices:
        return sampled_frames, indices.astype(int)
    return sampled_frames