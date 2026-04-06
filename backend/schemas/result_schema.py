"""Unified result schema for the AugMentor pipeline comparison demo."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class VideoMeta(BaseModel):
    filename: str
    path: Optional[str] = None
    duration_sec: Optional[float] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None


class Metrics(BaseModel):
    joint_angle_deviation: Optional[float] = None
    trajectory_deviation: Optional[float] = None
    velocity_difference: Optional[float] = None
    tool_alignment_deviation: Optional[float] = None
    dtw_cost: Optional[float] = None
    semantic_similarity: Optional[float] = None
    optical_flow_similarity: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None


class Confidences(BaseModel):
    overall: Optional[float] = None
    same_task: Optional[float] = None
    score: Optional[float] = None
    explanation: Optional[float] = None


class Explanation(BaseModel):
    text: str = ""
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    raw_vlm_output: Optional[str] = None
    structured_notes: Optional[Dict[str, Any]] = None


class RunMeta(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    pipeline_name: str
    processing_time_sec: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    component_notes: Optional[Dict[str, str]] = None


class PipelineResult(BaseModel):
    run: RunMeta
    expert_video: VideoMeta
    learner_video: VideoMeta
    overall_score: Optional[float] = None
    metrics: Metrics = Field(default_factory=Metrics)
    confidences: Confidences = Field(default_factory=Confidences)
    explanation: Explanation = Field(default_factory=Explanation)
    warnings: List[str] = Field(default_factory=list)
