"""VLM-only benchmark pipeline — raw-video baseline using Gemini."""

import time
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Dict, Optional

from backend.components.vlm import VLMComparator, VLMComparisonResult
from backend.schemas.pipeline_schema import PipelineInput
from backend.schemas.result_schema import (
    Confidences,
    Explanation,
    Metrics,
    PipelineResult,
    RunMeta,
    VideoMeta,
)

_PIPELINE_NAME = "vlm_only"


def _filename(path: str) -> str:
    for cls in (PureWindowsPath, PurePosixPath):
        name = cls(path).name
        if name:
            return name
    return path


def _build_warnings(vlm: VLMComparisonResult) -> list[str]:
    warnings: list[str] = []
    if vlm.same_task_label is False:
        warnings.append("same_task_label_false")
    if vlm.estimated_score is None:
        warnings.append("missing_estimated_score")
    if not vlm.full_explanation:
        warnings.append("missing_full_explanation")
    return warnings


def _build_structured_notes(vlm: VLMComparisonResult) -> Optional[Dict[str, Any]]:
    notes: Dict[str, Any] = {}
    if vlm.same_task_label is not None:
        notes["same_task_label"] = vlm.same_task_label
    if vlm.key_differences:
        notes["key_differences"] = vlm.key_differences
    if vlm.final_verdict:
        notes["final_verdict"] = vlm.final_verdict
    return notes or None


def run(pipeline_input: PipelineInput) -> PipelineResult:
    """Execute the VLM-only baseline: upload both raw videos to Gemini and
    return a standardised PipelineResult."""

    t0 = time.monotonic()

    comparator = VLMComparator()
    vlm = comparator.compare_videos(
        expert_video_path=pipeline_input.expert_video_path,
        learner_video_path=pipeline_input.learner_video_path,
    )

    elapsed = round(time.monotonic() - t0, 3)

    return PipelineResult(
        run=RunMeta(
            pipeline_name=_PIPELINE_NAME,
            processing_time_sec=elapsed,
        ),
        expert_video=VideoMeta(
            filename=_filename(pipeline_input.expert_video_path),
            path=pipeline_input.expert_video_path,
        ),
        learner_video=VideoMeta(
            filename=_filename(pipeline_input.learner_video_path),
            path=pipeline_input.learner_video_path,
        ),
        overall_score=vlm.estimated_score,
        metrics=Metrics(),
        confidences=Confidences(
            overall=vlm.overall_confidence,
            same_task=vlm.same_task_confidence,
        ),
        explanation=Explanation(
            text=vlm.full_explanation,
            strengths=vlm.strengths,
            weaknesses=vlm.weaknesses,
            raw_vlm_output=vlm.raw_vlm_output,
            structured_notes=_build_structured_notes(vlm),
        ),
        warnings=_build_warnings(vlm),
    )
