"""Pipeline identity, selection, input contract, and execution metadata schemas."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PipelineName(str, Enum):
    VLM_ONLY = "vlm_only"
    MEDIAPIPE_VLM = "mediapipe_vlm"
    MEDIAPIPE_DTW_VLM = "mediapipe_dtw_vlm"
    MEDIAPIPE_VJEPA_VLM = "mediapipe_vjepa_vlm"
    MEDIAPIPE_VJEPA_DTW_VLM = "mediapipe_vjepa_dtw_vlm"
    MEDIAPIPE_SAM2_DTW_VLM = "mediapipe_sam2_dtw_vlm"
    MEDIAPIPE_VJEPA_SAM2_DTW_VLM = "mediapipe_vjepa_sam2_dtw_vlm"
    MEDIAPIPE_VJEPA_GROUNDED_SAM2_DTW_VLM = "mediapipe_vjepa_grounded_sam2_dtw_vlm"
    MEDIAPIPE_OPTICAL_FLOW_DTW_VLM = "mediapipe_optical_flow_dtw_vlm"


class PipelineSelection(BaseModel):
    pipeline_name: PipelineName


class PipelineInput(BaseModel):
    pipeline_name: PipelineName
    expert_video_path: str
    learner_video_path: str
    config: Optional[Dict[str, Any]] = None


class PipelineExecutionMeta(BaseModel):
    pipeline_name: PipelineName
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None
    processing_time_sec: Optional[float] = None


class PipelineDescriptor(BaseModel):
    name: PipelineName
    description: str = ""
    components_used: List[str] = Field(default_factory=list)
