"""Reusable VLM comparison component using the Gemini API (raw-video path)."""

import json
import re
import time
from pathlib import Path
from typing import Any

from google import genai
from pydantic import BaseModel, Field

from backend.services.config_service import get_gemini_api_key

_DEFAULT_MODEL = "gemini-2.5-flash"
_UPLOAD_POLL_INTERVAL = 2
_UPLOAD_TIMEOUT = 300


# ---------------------------------------------------------------------------
# Internal result model
# ---------------------------------------------------------------------------

class VLMComparisonResult(BaseModel):
    estimated_score: float | None = None
    same_task_label: bool | None = None
    same_task_confidence: float | None = None
    overall_confidence: float | None = None
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    key_differences: list[str] = Field(default_factory=list)
    final_verdict: str = ""
    full_explanation: str = ""
    raw_vlm_output: str = ""


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------

class VLMComparator:
    """Upload two raw videos to Gemini and return a structured comparison."""

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model = model_name
        self._client = genai.Client(api_key=get_gemini_api_key())

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _file_state(uploaded: Any) -> str:
        """Normalize the file state to a lowercase string."""
        raw = getattr(uploaded, "state", None)
        if raw is None:
            return "unknown"
        return str(raw).rsplit(".", 1)[-1].strip().lower()

    def _upload_video(self, video_path: str) -> genai.types.File:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        uploaded = self._client.files.upload(file=str(path))

        deadline = time.monotonic() + _UPLOAD_TIMEOUT
        while self._file_state(uploaded) == "processing":
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Gemini file processing timed out after "
                    f"{_UPLOAD_TIMEOUT}s for: {video_path}"
                )
            time.sleep(_UPLOAD_POLL_INTERVAL)
            uploaded = self._client.files.get(name=uploaded.name)

        if self._file_state(uploaded) == "failed":
            raise RuntimeError(
                f"Gemini rejected the uploaded file: {video_path}"
            )
        return uploaded

    @staticmethod
    def build_compare_prompt() -> str:
        return (
            "You are an expert movement analyst evaluating two egocentric craft / skill videos.\n"
            "\n"
            "Video 1 is the EXPERT REFERENCE demonstrating the correct execution.\n"
            "Video 2 is the LEARNER IMITATION attempting to replicate the same task.\n"
            "\n"
            "Your job:\n"
            "1. Determine whether both videos appear to represent the same task "
            "and set same_task_label to true or false accordingly.\n"
            "2. Estimate how well the learner matches the expert in terms of technique, "
            "motion quality, timing, tool handling, and overall execution.\n"
            "3. Identify specific strong points of the learner's performance.\n"
            "4. Identify specific weak points or errors.\n"
            "5. List key differences in motion, posture, tool use, or execution flow.\n"
            "6. If you are uncertain about any judgment, lower the relevant confidence value "
            "rather than inventing certainty.\n"
            "\n"
            "Return ONLY a single JSON object with exactly two top-level keys:\n"
            '  "structured_result" — contains the benchmark-style structured fields.\n'
            '  "full_explanation"  — a rich, detailed explanation (multiple paragraphs if needed) '
            "suitable for research documentation and pedagogical feedback. "
            "Do not write only one or two sentences; provide thorough analysis.\n"
            "\n"
            "No markdown. No code fences. No extra text outside the JSON.\n"
            "\n"
            "Required JSON schema:\n"
            "{\n"
            '  "structured_result": {\n'
            '    "estimated_score": <float 0-100, overall learner similarity/performance>,\n'
            '    "same_task_label": <boolean, true if both videos show the same task>,\n'
            '    "same_task_confidence": <float 0-1, confidence in same_task_label>,\n'
            '    "overall_confidence": <float 0-1>,\n'
            '    "strengths": ["string", ...],\n'
            '    "weaknesses": ["string", ...],\n'
            '    "key_differences": ["string", ...],\n'
            '    "final_verdict": "string"\n'
            "  },\n"
            '  "full_explanation": "string"\n'
            "}\n"
        )

    # -- main entry ----------------------------------------------------------

    def compare_videos(
        self,
        expert_video_path: str,
        learner_video_path: str,
    ) -> VLMComparisonResult:
        prompt = self.build_compare_prompt()

        expert_file = self._upload_video(expert_video_path)
        learner_file = self._upload_video(learner_video_path)

        response = self._client.models.generate_content(
            model=self._model,
            contents=[expert_file, learner_file, prompt],
        )

        raw_text = response.text or ""
        return self._parse_response(raw_text)

    # -- parser --------------------------------------------------------------

    @staticmethod
    def _safe_float(data: dict, key: str) -> float | None:
        val = data.get(key)
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Field '{key}' must be a number, got {type(val).__name__}: {val!r}"
            ) from exc

    @staticmethod
    def _safe_bool(data: dict, key: str) -> bool | None:
        val = data.get(key)
        if val is None:
            return None
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            low = val.strip().lower()
            if low in ("true", "1", "yes"):
                return True
            if low in ("false", "0", "no"):
                return False
        raise ValueError(
            f"Field '{key}' must be a boolean, got {type(val).__name__}: {val!r}"
        )

    @staticmethod
    def _safe_str_list(data: dict, key: str) -> list[str]:
        val = data.get(key)
        if val is None:
            return []
        if isinstance(val, list):
            return [str(item) for item in val]
        if isinstance(val, str):
            return [val]
        raise ValueError(
            f"Field '{key}' must be a list of strings, "
            f"got {type(val).__name__}: {val!r}"
        )

    @classmethod
    def _parse_response(cls, raw: str) -> VLMComparisonResult:
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse VLM JSON response: {exc}\n"
                f"Raw output:\n{raw}"
            ) from exc

        sr = data.get("structured_result")
        if sr is None:
            raise ValueError(
                "VLM response missing 'structured_result' key.\n"
                f"Raw output:\n{raw}"
            )
        if not isinstance(sr, dict):
            raise ValueError(
                f"'structured_result' must be a JSON object, "
                f"got {type(sr).__name__}.\nRaw output:\n{raw}"
            )

        full_explanation = str(data.get("full_explanation", ""))

        return VLMComparisonResult(
            estimated_score=cls._safe_float(sr, "estimated_score"),
            same_task_label=cls._safe_bool(sr, "same_task_label"),
            same_task_confidence=cls._safe_float(sr, "same_task_confidence"),
            overall_confidence=cls._safe_float(sr, "overall_confidence"),
            strengths=cls._safe_str_list(sr, "strengths"),
            weaknesses=cls._safe_str_list(sr, "weaknesses"),
            key_differences=cls._safe_str_list(sr, "key_differences"),
            final_verdict=str(sr.get("final_verdict", "")),
            full_explanation=full_explanation,
            raw_vlm_output=raw,
        )
