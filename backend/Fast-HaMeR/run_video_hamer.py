"""
Video wrapper for the Fast-HaMeR hand mesh recovery pipeline.

Extracts frames from an input video, runs the existing demo_image.py on
them, and reassembles the rendered output into a final mp4.  A structured
JSON result file is saved alongside the output video for later comparison
with other pipelines (MediaPipe, Optical Flow, V-JEPA, etc.).

demo_image.py uses RTMLib for keypoint detection and PyTorch3D for
rendering — no Detectron2, mmpose, or ViTPose dependencies required.

Usage:
    python run_video_hamer.py \
        --input_video path/to/input.mp4 \
        --output_video path/to/output.mp4 \
        --fps 10
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_video_hamer")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process a video through the Fast-HaMeR hand mesh recovery pipeline.",
    )
    parser.add_argument(
        "--input_video", type=str, required=True,
        help="Path to the input .mp4 video",
    )
    parser.add_argument(
        "--output_video", type=str, required=True,
        help="Path for the output .mp4 video",
    )
    parser.add_argument(
        "--fps", type=int, default=10,
        help="Frames per second to extract and encode (default: 10)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="HaMeR checkpoint path (passed through to demo_image.py)",
    )
    parser.add_argument(
        "--efficient_hamer", action="store_true", default=False,
        help="Use the efficient (student/KD) HaMeR model (passed through to demo_image.py)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Batch size for HaMeR inference (passed through to demo_image.py)",
    )
    parser.add_argument(
        "--json_dir", type=str, default="outputs/json",
        help="Directory for the structured JSON result file (default: outputs/json)",
    )
    parser.add_argument(
        "--export_vertices", action="store_true", default=False,
        help="Include full 778-vertex mesh in raw hand-data JSON (large output)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Video metadata probe
# ---------------------------------------------------------------------------

def probe_video(video_path: Path) -> dict:
    """Use ffprobe to extract duration, fps, width, height from a video file.

    Returns a dict with keys: filename, path, duration_sec, fps, width, height.
    All values are None when the probe fails for a given field.
    """
    meta = {
        "filename": video_path.name,
        "path": str(video_path),
        "duration_sec": None,
        "fps": None,
        "width": None,
        "height": None,
    }
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)

        if "format" in info and "duration" in info["format"]:
            meta["duration_sec"] = round(float(info["format"]["duration"]), 3)

        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                meta["width"] = int(stream.get("width", 0)) or None
                meta["height"] = int(stream.get("height", 0)) or None
                r_fps = stream.get("r_frame_rate", "")
                if "/" in r_fps:
                    num, den = r_fps.split("/")
                    if int(den) > 0:
                        meta["fps"] = round(int(num) / int(den), 3)
                break
    except Exception as exc:
        log.warning("  ffprobe failed for %s: %s", video_path, exc)

    return meta


# ---------------------------------------------------------------------------
# JSON result builder
# ---------------------------------------------------------------------------

def build_result_json(
    *,
    run_id: str,
    input_video: Path,
    output_video: Path,
    extraction_fps: int,
    frame_count: int,
    rendered_count: int,
    processing_time_sec: float,
    efficient_hamer: bool,
    cleanup_ok: bool,
    warnings: list[str],
) -> dict:
    """Build a pipeline-comparable JSON result dict.

    Fields that Fast-HaMeR cannot compute (semantic scores, VLM output, etc.)
    are explicitly set to None so downstream comparison tools see a stable
    schema that matches the parent project's PipelineResult structure.
    """
    fallback_count = frame_count - rendered_count
    input_meta = probe_video(input_video)
    output_meta = probe_video(output_video) if output_video.exists() else {
        "filename": output_video.name,
        "path": str(output_video),
        "duration_sec": None, "fps": None, "width": None, "height": None,
    }

    strengths = [
        "Produces dense 778-vertex 3D hand mesh, not just 2D keypoints",
        "Full-frame overlay preserves original scene context",
    ]
    if rendered_count == frame_count:
        strengths.append("All frames received mesh overlay (no fallback needed)")

    weaknesses = []
    if fallback_count > 0:
        weaknesses.append(
            f"{fallback_count}/{frame_count} frames had no hand detection "
            "and used original frame as fallback"
        )
    weaknesses.append("No semantic or task-level scoring is computed by this pipeline")
    weaknesses.append("Monocular depth is ambiguous without reference scale")

    return {
        "run": {
            "run_id": run_id,
            "pipeline_name": "fast-hamer",
            "processing_time_sec": round(processing_time_sec, 2),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "component_notes": {
                "extraction_fps": str(extraction_fps),
                "frames_extracted": str(frame_count),
                "frames_sent_to_hamer": str(frame_count),
                "frames_with_mesh_overlay": str(rendered_count),
                "frames_without_overlay": str(fallback_count),
                "rebuild_image_type": "{index}_all.jpg (full-frame overlay)",
                "demo_script": "demo_image.py",
                "pose_detector": "RTMLib (Wholebody)",
                "mesh_renderer": "PyTorch3D",
                "efficient_hamer": str(efficient_hamer),
                "temp_cleanup_ok": str(cleanup_ok),
            },
        },
        "input_video": input_meta,
        "output_video": output_meta,
        "overall_score": None,
        "metrics": {
            "joint_angle_deviation": None,
            "trajectory_deviation": None,
            "velocity_difference": None,
            "tool_alignment_deviation": None,
            "dtw_cost": None,
            "semantic_similarity": None,
            "optical_flow_similarity": None,
            "extra": None,
        },
        "confidences": {
            "overall": None,
            "same_task": None,
            "score": None,
            "explanation": None,
        },
        "explanation": {
            "text": (
                f"Fast-HaMeR processed {frame_count} frames at {extraction_fps} FPS. "
                f"{rendered_count} frames produced 3D hand mesh overlays via "
                f"demo_image.py (RTMLib + PyTorch3D). "
                f"{fallback_count} frames had no detected hands and used the "
                f"original frame as fallback."
            ),
            "strengths": strengths,
            "weaknesses": weaknesses,
            "raw_vlm_output": None,
            "structured_notes": {
                "total_detected_hands": "not reliably countable from output images",
                "mesh_vertices_per_hand": 778,
                "mesh_faces_per_hand": 1538,
                "model_type": "efficient_hamer" if efficient_hamer else "hamer (ViT teacher)",
            },
        },
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def check_ffmpeg():
    """Verify ffmpeg is reachable on PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg is not installed or not on PATH.\n"
            "  Windows:  winget install ffmpeg   OR   https://ffmpeg.org/download.html\n"
            "  Linux:    sudo apt install ffmpeg"
        )


def extract_frames(video_path: Path, frames_dir: Path, fps: int) -> int:
    """Extract video frames at *fps* into *frames_dir* as frame_000001.jpg ...

    Returns the total number of extracted frames.
    """
    log.info("Extracting frames at %d FPS from: %s", fps, video_path)
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-qscale:v", "2",
        "-y",
        "-loglevel", "warning",
        str(frames_dir / "frame_%06d.jpg"),
    ]
    log.info("  cmd: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed:\n{result.stderr}")

    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    count = len(frame_files)
    if count == 0:
        raise RuntimeError(
            "ffmpeg produced zero frames — check that the input video is valid."
        )
    log.info("  Extracted %d frames", count)
    return count


def run_hamer_demo(
    frames_dir: Path,
    rendered_dir: Path,
    checkpoint: str | None,
    efficient_hamer: bool,
    batch_size: int | None,
    raw_json_out: Path | None = None,
    export_vertices: bool = False,
) -> None:
    """Invoke demo_image.py as a subprocess on the folder of extracted frames.

    demo_image.py uses RTMLib for pose detection and PyTorch3D for mesh
    rendering.  It writes one ``{index}_all.jpg`` per input image (0-based
    index from ``enumerate``), including a raw-frame fallback when no hands
    are detected, so the output count always matches the input count.
    """
    demo_script = Path(__file__).resolve().parent / "demo_image.py"
    if not demo_script.exists():
        raise RuntimeError(
            f"demo_image.py not found at expected location: {demo_script}"
        )

    cmd = [
        sys.executable, str(demo_script),
        "--img_folder", str(frames_dir),
        "--out_folder", str(rendered_dir),
        "--file_type", "*.jpg",
    ]
    if checkpoint:
        cmd.extend(["--checkpoint", checkpoint])
    if efficient_hamer:
        cmd.append("--efficient_hamer")
    if batch_size is not None:
        cmd.extend(["--batch_size", str(batch_size)])
    if raw_json_out is not None:
        cmd.extend(["--raw_json_out", str(raw_json_out)])
    if export_vertices:
        cmd.append("--export_vertices")

    log.info("Running Fast-HaMeR demo_image.py ...")
    log.info("  cmd: %s", " ".join(cmd))

    env = os.environ.copy()
    if sys.platform == "win32":
        env.setdefault("PYOPENGL_PLATFORM", "win32")

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"demo_image.py exited with return code {result.returncode}. "
            "Check the output above for details."
        )


def assemble_rebuild_frames(
    frames_dir: Path,
    rendered_dir: Path,
    rebuild_dir: Path,
    frame_count: int,
) -> int:
    """Collect rendered frames into a sequential rebuild directory.

    demo_image.py writes ``{index}_all.jpg`` using a 0-based integer index
    from ``enumerate(img_paths)``.  It always writes every frame (using the
    raw input as fallback when no hands are detected), so normally every
    index from 0 to frame_count-1 should exist.  A safety fallback to the
    original extracted frame is still included in case of unexpected gaps.

    The rebuild directory receives files named ``out_000001.jpg`` ...
    (1-based, for ffmpeg ``-start_number 1``).

    Returns the number of frames that came from the rendered output.
    """
    log.info("Assembling rebuild sequence (%d frames) ...", frame_count)
    rendered_count = 0

    for i in range(frame_count):
        rendered = rendered_dir / f"{i}_all.jpg"
        original = frames_dir / f"frame_{i + 1:06d}.jpg"
        dest = rebuild_dir / f"out_{i + 1:06d}.jpg"

        if rendered.exists():
            shutil.copy2(rendered, dest)
            rendered_count += 1
        elif original.exists():
            log.warning("  Rendered frame %d missing, using original fallback", i)
            shutil.copy2(original, dest)
        else:
            raise RuntimeError(
                f"Neither rendered nor original frame found for index {i} "
                f"(expected {rendered} or {original})"
            )

    skipped = frame_count - rendered_count
    log.info(
        "  %d / %d frames from HaMeR output, %d from original fallback",
        rendered_count, frame_count, skipped,
    )
    return rendered_count


def build_video(rebuild_dir: Path, output_path: Path, fps: int) -> None:
    """Encode the sequentially-named rebuild frames into an mp4."""
    log.info("Encoding output video at %d FPS -> %s", fps, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-r", str(fps),
        "-start_number", "1",
        "-i", str(rebuild_dir / "out_%06d.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-y",
        "-loglevel", "warning",
        str(output_path),
    ]
    log.info("  cmd: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg video encoding failed:\n{result.stderr}")

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(
            "ffmpeg finished but the output file is missing or empty."
        )
    size_mb = output_path.stat().st_size / (1024 * 1024)
    log.info("  Output video: %.2f MB", size_mb)


def run_derived_metrics(
    raw_json_path: Path,
    derived_json_path: Path,
    summary_json_path: Path | None = None,
    output_video_path: Path | None = None,
) -> None:
    """Run compute_derived_metrics.py on the raw hand-data JSON."""
    script = Path(__file__).resolve().parent / "compute_derived_metrics.py"
    if not script.exists():
        log.warning(
            "compute_derived_metrics.py not found — skipping derived metrics"
        )
        return

    cmd = [
        sys.executable, str(script),
        "--raw_json", str(raw_json_path),
        "--output", str(derived_json_path),
    ]
    if summary_json_path:
        cmd.extend(["--summary_json", str(summary_json_path)])
    if output_video_path:
        cmd.extend(["--output_video", str(output_video_path)])

    log.info("Computing derived metrics ...")
    log.info("  cmd: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.warning("Derived metrics computation failed:\n%s", result.stderr)
    else:
        log.info("Derived metrics saved: %s", derived_json_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    input_video = Path(args.input_video).resolve()
    output_video = Path(args.output_video).resolve()
    json_dir = Path(args.json_dir).resolve()

    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    check_ffmpeg()

    run_id = uuid4().hex
    warnings: list[str] = []

    log.info("=" * 60)
    log.info("Fast-HaMeR Video Pipeline")
    log.info("  Run ID:  %s", run_id)
    log.info("  Input :  %s", input_video)
    log.info("  Output:  %s", output_video)
    log.info("  FPS   :  %d", args.fps)
    log.info("  Script:  demo_image.py (RTMLib + PyTorch3D)")
    log.info("=" * 60)

    tmpdir = tempfile.mkdtemp(prefix="hamer_video_")
    tmp = Path(tmpdir)
    frames_dir = tmp / "frames"
    rendered_dir = tmp / "rendered"
    rebuild_dir = tmp / "rebuild"
    frames_dir.mkdir()
    rendered_dir.mkdir()
    rebuild_dir.mkdir()
    raw_json_tmp = tmp / "raw_hand_data.json"

    t_start = time.time()
    frame_count = 0
    rendered_count = 0
    cleanup_ok = False
    raw_frames = []

    try:
        # 1. Extract frames
        frame_count = extract_frames(input_video, frames_dir, args.fps)

        # 2. Run HaMeR on extracted frames
        run_hamer_demo(
            frames_dir,
            rendered_dir,
            checkpoint=args.checkpoint,
            efficient_hamer=args.efficient_hamer,
            batch_size=args.batch_size,
            raw_json_out=raw_json_tmp,
            export_vertices=args.export_vertices,
        )

        if raw_json_tmp.exists():
            with open(raw_json_tmp, "r", encoding="utf-8") as f:
                raw_frames = json.load(f)

        # 3. Pick best output per frame (rendered overlay or original fallback)
        rendered_count = assemble_rebuild_frames(
            frames_dir, rendered_dir, rebuild_dir, frame_count,
        )
        fallback_count = frame_count - rendered_count
        if fallback_count > 0:
            warnings.append(
                f"{fallback_count}/{frame_count} frames used original fallback "
                "(no hand detected by HaMeR)"
            )
        if rendered_count == 0:
            warnings.append(
                "No hands were detected in any frame — output is original video only"
            )

        # 4. Encode final video
        build_video(rebuild_dir, output_video, args.fps)

        log.info("=" * 60)
        log.info("Done. Output video: %s", output_video)
        log.info("=" * 60)

    finally:
        log.info("Cleaning up temporary directory: %s", tmpdir)
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
            cleanup_ok = True
        except Exception:
            warnings.append("Temp directory cleanup may have failed")

    processing_time = time.time() - t_start

    # --- Honesty warnings for downstream comparison ---
    warnings.append("overall_score is null — Fast-HaMeR does not compute a comparison score")
    warnings.append("All metrics fields are null — no quantitative evaluation was performed")
    warnings.append("No semantic or VLM-based analysis was performed")

    # 5. Build and save JSON result
    result = build_result_json(
        run_id=run_id,
        input_video=input_video,
        output_video=output_video,
        extraction_fps=args.fps,
        frame_count=frame_count,
        rendered_count=rendered_count,
        processing_time_sec=processing_time,
        efficient_hamer=args.efficient_hamer,
        cleanup_ok=cleanup_ok,
        warnings=warnings,
    )

    json_dir.mkdir(parents=True, exist_ok=True)
    json_filename = f"fast_hamer_{run_id[:12]}.json"
    json_path = json_dir / json_filename
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.info("Summary JSON saved: %s", json_path)

    # 6. Build and save raw hand-data JSON
    for fr in raw_frames:
        fr["timestamp_sec"] = round(fr["frame_index"] / args.fps, 6)

    input_meta = probe_video(input_video)
    raw_result = {
        "run": {
            "run_id": run_id,
            "pipeline_name": "fast-hamer",
            "model_name": "efficient_hamer" if args.efficient_hamer else "hamer",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fps_used_for_processing": args.fps,
            "total_frames": frame_count,
            "raw_schema_version": "1.0",
        },
        "video_metadata": {
            "filename": input_meta["filename"],
            "path": input_meta["path"],
            "width": input_meta["width"],
            "height": input_meta["height"],
            "fps": input_meta["fps"],
            "duration_sec": input_meta["duration_sec"],
        },
        "frames": raw_frames,
    }

    raw_json_filename = f"fast_hamer_{run_id[:12]}_raw_hand_data.json"
    raw_json_path = json_dir / raw_json_filename
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(raw_result, f, indent=2, ensure_ascii=False)
    log.info("Raw hand-data JSON saved: %s", raw_json_path)

    # 7. Compute derived metrics from raw hand data
    derived_json_filename = f"fast_hamer_{run_id[:12]}_derived_metrics.json"
    derived_json_path = json_dir / derived_json_filename
    run_derived_metrics(raw_json_path, derived_json_path, json_path, output_video)


if __name__ == "__main__":
    main()
