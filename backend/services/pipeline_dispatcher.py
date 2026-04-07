"""Central pipeline dispatcher for the AugMentor benchmark demo."""

from pathlib import PurePosixPath, PureWindowsPath
from typing import Callable, Dict

from backend.pipelines.mediapipe_vlm import run as run_mediapipe_vlm
from backend.schemas.pipeline_schema import PipelineInput, PipelineName
from backend.schemas.result_schema import (
    Explanation,
    PipelineResult,
    RunMeta,
    VideoMeta,
)

Runner = Callable[[PipelineInput], PipelineResult]


def _filename_from_path(path: str) -> str:
    """Extract the filename component from a local or posix-style path."""
    for cls in (PureWindowsPath, PurePosixPath):
        name = cls(path).name
        if name:
            return name
    return path


def _make_placeholder_runner(pipeline_name: PipelineName) -> Runner:
    """Return a runner that produces a valid but not-implemented result."""

    def _run(pipeline_input: PipelineInput) -> PipelineResult:
        return PipelineResult(
            run=RunMeta(pipeline_name=pipeline_name.value),
            expert_video=VideoMeta(
                filename=_filename_from_path(pipeline_input.expert_video_path),
                path=pipeline_input.expert_video_path,
            ),
            learner_video=VideoMeta(
                filename=_filename_from_path(pipeline_input.learner_video_path),
                path=pipeline_input.learner_video_path,
            ),
            overall_score=None,
            explanation=Explanation(
                text=(
                    f"Pipeline '{pipeline_name.value}' is registered "
                    f"but not implemented yet."
                ),
            ),
            warnings=["pipeline_not_implemented"],
        )

    return _run


# ---------------------------------------------------------------------------
# Pipeline registry
# ---------------------------------------------------------------------------
# Each entry maps a PipelineName to a runner callable.
# Replace individual placeholders with real imports as pipelines are built,
# e.g.:
#   from backend.pipelines.vlm_only import run as vlm_only_run
#   REGISTRY[PipelineName.VLM_ONLY] = vlm_only_run
# ---------------------------------------------------------------------------

REGISTRY: Dict[PipelineName, Runner] = {
    member: _make_placeholder_runner(member) for member in PipelineName
}
REGISTRY[PipelineName.MEDIAPIPE_VLM] = run_mediapipe_vlm


# ---------------------------------------------------------------------------
# Public dispatch entry point
# ---------------------------------------------------------------------------

def dispatch_pipeline(pipeline_input: PipelineInput) -> PipelineResult:
    """Look up the selected pipeline in the registry and execute it."""
    runner = REGISTRY.get(pipeline_input.pipeline_name)
    if runner is None:
        raise ValueError(
            f"Pipeline '{pipeline_input.pipeline_name.value}' is not in the "
            f"registry. Available: {[m.value for m in REGISTRY]}"
        )
    return runner(pipeline_input)
