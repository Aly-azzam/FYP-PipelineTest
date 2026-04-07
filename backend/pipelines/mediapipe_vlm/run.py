from __future__ import annotations

import time
import uuid
from pathlib import Path, PurePosixPath, PureWindowsPath

from backend.components.mediapipe.wrist_extractor import extract_wrist_from_video
from backend.schemas.pipeline_schema import PipelineInput
from backend.schemas.result_schema import (
    Confidences,
    Explanation,
    Metrics,
    PipelineResult,
    RunMeta,
    VideoMeta,
)

_PIPELINE_NAME = "mediapipe_vlm"


def _filename(path: str) -> str:
    for cls in (PureWindowsPath, PurePosixPath):
        name = cls(path).name
        if name:
            return name
    return path


def _count_detected_wrist_frames(video_result: dict) -> int:
    count = 0
    for frame in video_result.get("frames", []):
        if frame.get("left_wrist_visible") or frame.get("right_wrist_visible"):
            count += 1
    return count


def run(pipeline_input: PipelineInput) -> PipelineResult:
    start_time = time.perf_counter()

    output_dir = Path(__file__).resolve().parents[2] / ".tmp_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    expert_output_path = output_dir / f"{uuid.uuid4().hex}_expert_annotated.mp4"
    learner_output_path = output_dir / f"{uuid.uuid4().hex}_learner_annotated.mp4"

    expert_result = extract_wrist_from_video(
        pipeline_input.expert_video_path,
        annotated_output_path=str(expert_output_path),
    )
    learner_result = extract_wrist_from_video(
        pipeline_input.learner_video_path,
        annotated_output_path=str(learner_output_path),
    )

    processing_time_sec = time.perf_counter() - start_time

    expert_detected_frames = _count_detected_wrist_frames(expert_result)
    learner_detected_frames = _count_detected_wrist_frames(learner_result)

    warnings: list[str] = [
        "vlm_not_implemented",
        "score_not_computed",
        "wrist_only_extraction",
    ]

    if expert_detected_frames == 0:
        warnings.append("no_wrist_detected_in_expert_video")
    if learner_detected_frames == 0:
        warnings.append("no_wrist_detected_in_learner_video")

    component_notes = {
        "mediapipe": (
            f"mode=wrist_only;"
            f" expert_detected_wrist_frames={expert_detected_frames};"
            f" learner_detected_wrist_frames={learner_detected_frames}"
        ),
        "vlm": "implemented=False",
    }

    return PipelineResult(
        run=RunMeta(
            pipeline_name=_PIPELINE_NAME,
            processing_time_sec=processing_time_sec,
            component_notes=component_notes,
        ),
        expert_video=VideoMeta(
            filename=_filename(expert_result.get("annotated_path") or pipeline_input.expert_video_path),
            path=expert_result.get("annotated_path") or pipeline_input.expert_video_path,
            duration_sec=expert_result.get("duration_sec"),
            fps=expert_result.get("fps"),
            width=expert_result.get("width"),
            height=expert_result.get("height"),
        ),
        learner_video=VideoMeta(
            filename=_filename(learner_result.get("annotated_path") or pipeline_input.learner_video_path),
            path=learner_result.get("annotated_path") or pipeline_input.learner_video_path,
            duration_sec=learner_result.get("duration_sec"),
            fps=learner_result.get("fps"),
            width=learner_result.get("width"),
            height=learner_result.get("height"),
        ),
        overall_score=None,
        metrics=Metrics(),
        confidences=Confidences(),
        explanation=Explanation(
            text="MediaPipe wrist extraction completed. VLM is not implemented yet.",
            strengths=[],
            weaknesses=[],
            structured_notes={
                "expert_detected_wrist_frames": expert_detected_frames,
                "learner_detected_wrist_frames": learner_detected_frames,
            },
        ),
        warnings=warnings,
    )