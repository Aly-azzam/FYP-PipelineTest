import json
import numpy as np
from pathlib import Path, PurePosixPath, PureWindowsPath
from scipy.optimize import linear_sum_assignment
from time import perf_counter

from backend.components.vjepa.service import VJepa21Service
from backend.components.vjepa.similarity import cosine_similarity
from backend.schemas.pipeline_schema import PipelineName
from backend.schemas.result_schema import (
    PipelineResult,
    RunMeta,
    Metrics,
    Confidences,
    Explanation,
    VideoMeta,
)


def _filename(path: str) -> str:
    for cls in (PureWindowsPath, PurePosixPath):
        name = cls(path).name
        if name:
            return name
    return path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _embedding_preview(embedding: np.ndarray, size: int = 8) -> list[float]:
    return [float(x) for x in embedding[:size].tolist()]


def _read_video_dimensions(video_path: str) -> tuple[int | None, int | None]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()

    return (
        width if width > 0 else None,
        height if height > 0 else None,
    )


def _build_video_meta(path: str, video_debug: dict) -> VideoMeta:
    width, height = _read_video_dimensions(path)
    return VideoMeta(
        filename=_filename(path),
        path=path,
        duration_sec=video_debug.get("duration_sec"),
        fps=video_debug.get("fps"),
        width=width,
        height=height,
    )


_BOTTOM_K_RATIO = 0.5
_BOTTOM_K_WEIGHT = 0.6
_MEAN_WEIGHT = 0.4
_GAP_PENALTY_WEIGHT = 0.3
_STD_PENALTY_WEIGHT = 0.2
_SCORE_GAMMA = 3.0
_SIM_BASELINE = 0.96
_SIM_RANGE = 1.0 - _SIM_BASELINE
_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "json" / "vjepa_only"


def run(pipeline_input):
    run_started = perf_counter()
    expert_path = pipeline_input.expert_video_path
    learner_path = pipeline_input.learner_video_path

    vjepa = VJepa21Service()
    num_segments = vjepa.config.num_segments

    expert_segment_embs = vjepa.extract_segment_embeddings(
        expert_path,
        num_segments=num_segments,
    )
    expert_fallback = vjepa.used_chunk_fallback
    expert_segment_debug = [dict(item) for item in vjepa.last_segment_debug]
    expert_video_debug = dict(vjepa.last_video_debug)

    learner_segment_embs = vjepa.extract_segment_embeddings(
        learner_path,
        num_segments=num_segments,
    )
    learner_fallback = vjepa.used_chunk_fallback
    learner_segment_debug = [dict(item) for item in vjepa.last_segment_debug]
    learner_video_debug = dict(vjepa.last_video_debug)

    similarity_matrix = [
        [
            cosine_similarity(expert_emb, learner_emb)
            for learner_emb in learner_segment_embs
        ]
        for expert_emb in expert_segment_embs
    ]
    matrix_np = np.asarray(similarity_matrix, dtype=np.float32)

    # Optimal one-to-one matching via the Hungarian algorithm.
    cost_matrix = 1.0 - matrix_np
    expert_indices, learner_indices = linear_sum_assignment(cost_matrix)

    chosen_matches = []
    segment_similarities = []
    for expert_idx, learner_idx in zip(expert_indices, learner_indices):
        similarity = float(matrix_np[expert_idx, learner_idx])
        chosen_matches.append(
            {
                "expert_segment": int(expert_idx),
                "learner_segment": int(learner_idx),
                "similarity": similarity,
            }
        )
        segment_similarities.append(similarity)

    if not segment_similarities:
        raise ValueError("No segment similarities were produced for vjepa_only")

    # ── Similarity re-centering (temporary calibration heuristic) ────────────
    # V-JEPA cosine similarities cluster near 1.0 for any input pair.  To give
    # the scoring meaningful dynamic range we subtract a fixed empirical baseline
    # (0.96) and rescale into [0, 1]:
    #
    #   adjusted = clip( (raw - 0.96) / (1.0 - 0.96), 0, 1 )
    #
    # 0.96 was chosen as the approximate floor observed for genuinely unrelated
    # V-JEPA ViT-Base-384 video pairs.  Using a fixed baseline avoids the problem
    # of a per-run median that shifts upward for similar-video pairs and zeroes
    # out legitimately good matches.
    # This is a temporary heuristic, not a validated calibration.
    sim_baseline = _SIM_BASELINE
    sim_range = _SIM_RANGE

    def _recenter(raw: float) -> float:
        return float(np.clip((raw - sim_baseline) / sim_range, 0.0, 1.0))

    # Raw (uncalibrated) per-segment stats — kept for diagnostics.
    raw_segment_similarities = list(segment_similarities)
    mean_segment_similarity_raw = float(np.mean(raw_segment_similarities))
    min_segment_similarity_raw = float(np.min(raw_segment_similarities))

    # Adjusted per-segment similarities used for all downstream scoring.
    adj_segment_similarities = [_recenter(s) for s in segment_similarities]

    mean_segment_similarity = float(np.mean(adj_segment_similarities))
    min_segment_similarity = float(np.min(adj_segment_similarities))
    segment_variance = float(np.var(adj_segment_similarities))
    segment_std = float(np.sqrt(segment_variance))
    consistency = max(0.0, 1.0 - segment_std)
    similarity_gap = max(0.0, mean_segment_similarity - min_segment_similarity)

    sorted_sims = sorted(adj_segment_similarities)
    bottom_k = max(1, int(len(sorted_sims) * _BOTTOM_K_RATIO))
    bottom_k_mean = float(np.mean(sorted_sims[:bottom_k]))

    # Temporary heuristic (not a validated rubric): score is driven by the
    # average of the worst half of matched segments (adjusted), blended with
    # the overall adjusted mean, then penalized by gap and spread.
    weighted_similarity = (
        (_BOTTOM_K_WEIGHT * bottom_k_mean)
        + (_MEAN_WEIGHT * mean_segment_similarity)
    )
    mismatch_penalty = (
        (_GAP_PENALTY_WEIGHT * similarity_gap)
        + (_STD_PENALTY_WEIGHT * segment_std)
    )
    # final_similarity is in [0, 1] (adjusted scale).
    final_similarity = float(np.clip(weighted_similarity - mismatch_penalty, 0.0, 1.0))

    worst_match = min(chosen_matches, key=lambda item: item["similarity"])
    per_segment_metrics = []
    for match in chosen_matches:
        expert_idx = int(match["expert_segment"])
        learner_idx = int(match["learner_segment"])
        expert_meta = expert_segment_debug[expert_idx]
        learner_meta = learner_segment_debug[learner_idx]
        per_segment_metrics.append(
            {
                "segment_index": expert_idx,
                "expert_segment_start_time_sec": expert_meta.get("start_time_sec"),
                "expert_segment_end_time_sec": expert_meta.get("end_time_sec"),
                "learner_segment_start_time_sec": learner_meta.get("start_time_sec"),
                "learner_segment_end_time_sec": learner_meta.get("end_time_sec"),
                "frames_used": {
                    "expert": expert_meta.get("frames_used"),
                    "learner": learner_meta.get("frames_used"),
                },
                "cosine_similarity_raw": float(match["similarity"]),
                "cosine_similarity_adjusted": _recenter(match["similarity"]),
                "expert_embedding_norm": float(np.linalg.norm(expert_segment_embs[expert_idx])),
                "learner_embedding_norm": float(np.linalg.norm(learner_segment_embs[learner_idx])),
                "worst_segment": expert_idx == int(worst_match["expert_segment"]),
            }
        )

    # semantic_similarity is exposed on the adjusted [0, 1] scale so the
    # frontend reflects the re-centred value, not the raw cosine.
    semantic_similarity = final_similarity
    linear_score = final_similarity * 100.0  # direct percentage on adjusted scale
    # Temporary calibrated heuristic: power-law mapping spreads the [0,1]
    # adjusted range so near-perfect matches score near 100 and moderate
    # matches score proportionally lower.
    calibrated_score = final_similarity**_SCORE_GAMMA
    overall_score = float(np.clip(calibrated_score * 100.0, 0.0, 100.0))

    used_fallback = expert_fallback or learner_fallback

    if used_fallback:
        explanation_text = (
            "This run used a temporary chunk-based embedding fallback instead of the full "
            "V-JEPA 2.1 encoder for at least one segment (weights failed to load or "
            "inference errored). Treat the stricter segment-wise similarity output as a "
            "rough diagnostic only."
        )
        strengths = [
            "Pipeline runs end to end",
            "A segment-wise similarity signal is still produced when the encoder is unavailable",
        ]
        weaknesses = [
            "Chunk-based fallback embeddings are not semantically comparable to real V-JEPA",
            "Fix checkpoint download or set VJepaConfig.checkpoint_path for full quality",
        ]
        warnings = ["temporary_embedding_fallback"]
        embedding_note = "temporary_embedding_fallback"
        conf_overall = 0.55
    else:
        explanation_text = (
            "This comparison uses real V-JEPA 2.1 encoder embeddings on multiple temporal "
            "segments from the expert and learner clips. Segment matching uses the Hungarian "
            "algorithm for optimal one-to-one assignment (no segment reuse). The final "
            "similarity is driven by the worst-half of matched segments plus gap and spread "
            "penalties, and the 0-100 score uses a steep nonlinear calibration (gamma=8). "
            "This remains a temporary heuristic."
        )
        strengths = [
            "Uses real V-JEPA 2.1 encoder embeddings",
            "Segment-wise semantic similarity based comparison with no external VLM used",
        ]
        weaknesses = [
            "Temporal alignment is still coarse because matching is segment-by-segment",
            "This remains a semantic similarity signal rather than a calibrated skill rubric",
        ]
        warnings = []
        embedding_note = "vjepa2_1_encoder"
        conf_overall = 0.72

    processing_time_sec = float(perf_counter() - run_started)
    structured_notes = {
        "real_encoder_used": not used_fallback,
        "fallback_used": used_fallback,
        "worst_segment_index": int(worst_match["expert_segment"]),
        "segment_count": len(segment_similarities),
        "frames_per_clip": vjepa.config.frames_per_clip,
    }

    result = PipelineResult(
        run=RunMeta(
            pipeline_name=PipelineName.VJEPA_ONLY,
            processing_time_sec=processing_time_sec,
            component_notes={
                "pipeline": "vjepa_only_pipeline",
                "scoring": "hungarian_segment_match_semantic_similarity_scoring",
                "embedding": embedding_note,
            },
        ),
        expert_video=_build_video_meta(expert_path, expert_video_debug),
        learner_video=_build_video_meta(learner_path, learner_video_debug),
        overall_score=overall_score,
        metrics=Metrics(
            semantic_similarity=semantic_similarity,
            extra={
                "segment_count": len(raw_segment_similarities),
                "similarity_matrix": similarity_matrix,
                "chosen_segment_matches": chosen_matches,
                "worst_matched_segment": worst_match,
                "worst_segment_index": int(worst_match["expert_segment"]),
                # Raw (uncalibrated) cosine similarities from V-JEPA.
                "segment_similarities_raw": [float(x) for x in raw_segment_similarities],
                "mean_segment_similarity_raw": mean_segment_similarity_raw,
                "min_segment_similarity_raw": min_segment_similarity_raw,
                # Re-centering parameters (temporary calibration heuristic).
                "sim_baseline": sim_baseline,
                "sim_range": sim_range,
                # Adjusted similarities after re-centering to [0, 1].
                "segment_similarities_adjusted": [float(x) for x in adj_segment_similarities],
                "mean_segment_similarity": mean_segment_similarity,
                "min_segment_similarity": min_segment_similarity,
                "bottom_k": bottom_k,
                "bottom_k_mean": bottom_k_mean,
                "segment_similarity_variance": segment_variance,
                "segment_similarity_std": segment_std,
                "segment_consistency": consistency,
                "weighted_similarity": float(weighted_similarity),
                "similarity_gap": float(similarity_gap),
                "mismatch_penalty": float(mismatch_penalty),
                "final_similarity": float(final_similarity),
                "linear_score": float(linear_score),
                "score_gamma": _SCORE_GAMMA,
                "calibrated_score": float(overall_score),
                "score_bottom_k_weight": _BOTTOM_K_WEIGHT,
                "score_mean_weight": _MEAN_WEIGHT,
                "score_gap_penalty_weight": _GAP_PENALTY_WEIGHT,
                "score_std_penalty_weight": _STD_PENALTY_WEIGHT,
                "scoring_note": "temporary heuristic, not a validated rubric",
            },
        ),
        confidences=Confidences(
            overall=conf_overall,
            score=conf_overall,
        ),
        explanation=Explanation(
            text=explanation_text,
            strengths=strengths,
            weaknesses=weaknesses,
            structured_notes=structured_notes,
        ),
        warnings=warnings,
    )

    run_id = result.run.run_id
    result_json_path = _OUTPUT_DIR / f"vjepa_{run_id}.json"
    derived_json_path = _OUTPUT_DIR / f"vjepa_{run_id}_derived_metrics.json"
    raw_debug_json_path = _OUTPUT_DIR / f"vjepa_{run_id}_raw_embeddings.json"

    result_payload = result.model_dump(mode="json")
    _write_json(result_json_path, result_payload)

    derived_payload = {
        "run": result_payload["run"],
        "source_files": {
            "expert_video_path": expert_path,
            "learner_video_path": learner_path,
            "main_result_json": str(result_json_path),
        },
        "summary_metrics": {
            "overall_score": float(overall_score),
            "semantic_similarity": float(semantic_similarity),
            # Raw cosine values before re-centering.
            "mean_segment_similarity_raw": mean_segment_similarity_raw,
            "min_segment_similarity_raw": min_segment_similarity_raw,
            # Re-centering parameters.
            "sim_baseline": sim_baseline,
            "sim_range": sim_range,
            # Adjusted values used for scoring.
            "mean_segment_similarity": float(mean_segment_similarity),
            "min_segment_similarity": float(min_segment_similarity),
            "bottom_k": bottom_k,
            "bottom_k_mean": float(bottom_k_mean),
            "segment_similarity_variance": float(segment_variance),
            "segment_similarity_std": float(segment_std),
            "segment_consistency": float(consistency),
            "similarity_gap": float(similarity_gap),
            "mismatch_penalty": float(mismatch_penalty),
            "weighted_similarity": float(weighted_similarity),
            "final_similarity": float(final_similarity),
            "linear_score": float(linear_score),
            "calibrated_score": float(overall_score),
            "score_gamma": _SCORE_GAMMA,
        },
        "embedding_mode": {
            "real_encoder_used": not used_fallback,
            "fallback_used": used_fallback,
            "embedding_note": embedding_note,
            "model_name": vjepa.config.model_name,
            "frames_per_clip": vjepa.config.frames_per_clip,
            "num_segments": num_segments,
            "device": vjepa.device,
            "last_error": vjepa.last_error,
        },
        "segment_similarity_summary": {
            "segment_count": len(raw_segment_similarities),
            "similarity_matrix": similarity_matrix,
            "chosen_segment_matches": chosen_matches,
            "worst_matched_segment": worst_match,
            "worst_segment_index": int(worst_match["expert_segment"]),
            "segment_similarities_raw": [float(x) for x in raw_segment_similarities],
            "segment_similarities_adjusted": [float(x) for x in adj_segment_similarities],
        },
        "per_segment_metrics": per_segment_metrics,
        "video_debug": {
            "expert": expert_video_debug,
            "learner": learner_video_debug,
        },
    }
    _write_json(derived_json_path, derived_payload)

    if vjepa.config.debug_export_raw_embeddings:
        raw_payload = {
            "run": result_payload["run"],
            "debug_mode": True,
            "raw_vectors_included": False,
            "note": "Raw embedding previews only; full vectors are not dumped by default.",
            "segments": [
                {
                    "segment_index": idx,
                    "expert_sampled_frame_indices": expert_segment_debug[idx].get("sampled_frame_indices"),
                    "learner_sampled_frame_indices": learner_segment_debug[learner_idx].get("sampled_frame_indices"),
                    "expert_embedding_norm": float(np.linalg.norm(expert_segment_embs[idx])),
                    "learner_embedding_norm": float(np.linalg.norm(learner_segment_embs[learner_idx])),
                    "expert_embedding_preview": _embedding_preview(expert_segment_embs[idx]),
                    "learner_embedding_preview": _embedding_preview(learner_segment_embs[learner_idx]),
                    "cosine_similarity": float(match["similarity"]),
                }
                for idx, match in enumerate(chosen_matches)
                for learner_idx in [int(match["learner_segment"])]
            ],
        }
        _write_json(raw_debug_json_path, raw_payload)

    return result
