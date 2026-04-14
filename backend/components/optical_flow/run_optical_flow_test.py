from __future__ import annotations

import argparse
from pathlib import Path

from .comparison_service import run_optical_flow_comparison
from .evaluation_service import (
    OpticalFlowEvaluationConfig,
    evaluate_optical_flow_summary,
)
from .farneback_service import FarnebackConfig
from .io_utils import save_optical_flow_results
from .schemas import (
    OpticalFlowEvaluationConfigUsed,
    OpticalFlowEvaluationResult,
    OpticalFlowScore,
    OpticalFlowSimilarities,
)
from .visualizer import create_comparison_visualizations


DEFAULT_OUTPUT_DIR = Path("backend/outputs/optical_flow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Farneback optical flow comparison on expert and learner videos."
    )

    parser.add_argument(
        "expert_video",
        type=str,
        help="Path to the expert video file",
    )
    parser.add_argument(
        "learner_video",
        type=str,
        help="Path to the learner video file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where raw JSON, summary JSON, and visualization videos will be saved",
    )

    parser.add_argument("--pyr-scale", type=float, default=0.5)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--winsize", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--poly-n", type=int, default=5)
    parser.add_argument("--poly-sigma", type=float, default=1.2)
    parser.add_argument("--flags", type=int, default=0)
    parser.add_argument("--motion-threshold", type=float, default=2.0)

    parser.add_argument(
        "--resize-width",
        type=int,
        default=None,
        help="Optional resize width before optical flow",
    )
    parser.add_argument(
        "--resize-height",
        type=int,
        default=None,
        help="Optional resize height before optical flow",
    )

    parser.add_argument(
        "--gaussian-blur-kernel",
        type=int,
        default=5,
        help="Odd Gaussian blur kernel size before flow. Use 0 to disable blur.",
    )

    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Also save HSV optical flow visualization videos for expert and learner",
    )
    parser.add_argument(
        "--magnitude-clip-percentile",
        type=float,
        default=99.0,
        help="Percentile used to clip magnitude before HSV normalization",
    )

    parser.add_argument("--magnitude-ref", type=float, default=1.0)
    parser.add_argument("--motion-area-ref", type=float, default=0.2)
    parser.add_argument("--angle-ref-deg", type=float, default=90.0)

    parser.add_argument("--magnitude-weight", type=float, default=0.4)
    parser.add_argument("--motion-area-weight", type=float, default=0.2)
    parser.add_argument("--angle-weight", type=float, default=0.4)

    return parser.parse_args()


def build_flow_config(args: argparse.Namespace) -> FarnebackConfig:
    return FarnebackConfig(
        pyr_scale=args.pyr_scale,
        levels=args.levels,
        winsize=args.winsize,
        iterations=args.iterations,
        poly_n=args.poly_n,
        poly_sigma=args.poly_sigma,
        flags=args.flags,
        motion_threshold=args.motion_threshold,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
        gaussian_blur_kernel=args.gaussian_blur_kernel,
    )


def build_evaluation_config(args: argparse.Namespace) -> OpticalFlowEvaluationConfig:
    return OpticalFlowEvaluationConfig(
        magnitude_ref=args.magnitude_ref,
        motion_area_ref=args.motion_area_ref,
        angle_ref_deg=args.angle_ref_deg,
        magnitude_weight=args.magnitude_weight,
        motion_area_weight=args.motion_area_weight,
        angle_weight=args.angle_weight,
    )


def build_schema_evaluation_result(evaluation_result: dict) -> OpticalFlowEvaluationResult:
    return OpticalFlowEvaluationResult(
        similarities=OpticalFlowSimilarities(**evaluation_result["similarities"]),
        score=OpticalFlowScore(**evaluation_result["score"]),
        config_used=OpticalFlowEvaluationConfigUsed(**evaluation_result["config_used"]),
    )


def main() -> None:
    args = parse_args()
    flow_config = build_flow_config(args)
    evaluation_config = build_evaluation_config(args)

    print("Starting optical flow comparison...")
    print(f"Expert video: {args.expert_video}")
    print(f"Learner video: {args.learner_video}")

    raw_result, summary_result = run_optical_flow_comparison(
        expert_video_path=args.expert_video,
        learner_video_path=args.learner_video,
        config=flow_config,
    )

    evaluation_result = evaluate_optical_flow_summary(
        summary_result=summary_result,
        config=evaluation_config,
    )
    summary_result.optical_flow_evaluation = build_schema_evaluation_result(evaluation_result)

    raw_path, summary_path = save_optical_flow_results(
        raw_result=raw_result,
        summary_result=summary_result,
        output_dir=args.output_dir,
    )

    similarities = evaluation_result["similarities"]
    optical_flow_score = evaluation_result["score"]["optical_flow_score"]

    print("\nOptical flow comparison completed successfully.")
    print(f"Run ID: {raw_result.run.run_id}")
    print(f"Raw JSON saved to: {raw_path}")
    print(f"Summary JSON saved to: {summary_path}")

    print("\nOptical flow evaluation:")
    print(f"  Magnitude similarity:   {similarities['magnitude_similarity']:.6f}")
    print(f"  Motion area similarity: {similarities['motion_area_similarity']:.6f}")
    print(f"  Angle similarity:       {similarities['angle_similarity']:.6f}")
    print(f"  Optical flow score:     {optical_flow_score:.6f}/100")

    if args.save_visualizations:
        expert_vis_path, learner_vis_path = create_comparison_visualizations(
            expert_video_path=args.expert_video,
            learner_video_path=args.learner_video,
            output_dir=args.output_dir,
            run_id=raw_result.run.run_id,
            config=flow_config,
            magnitude_clip_percentile=args.magnitude_clip_percentile,
        )

        print(f"Expert HSV video saved to: {expert_vis_path}")
        print(f"Learner HSV video saved to: {learner_vis_path}")


if __name__ == "__main__":
    main()