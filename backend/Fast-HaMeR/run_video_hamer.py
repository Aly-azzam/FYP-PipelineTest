"""
Video wrapper for the Fast-HaMeR hand mesh recovery pipeline.

Extracts frames from an input video, runs the existing demo_image.py on
them, and reassembles the rendered output into a final mp4.

demo_image.py uses RTMLib for keypoint detection and PyTorch3D for
rendering — no Detectron2, mmpose, or ViTPose dependencies required.

Usage:
    python run_video_hamer.py \
        --input_video path/to/input.mp4 \
        --output_video path/to/output.mp4 \
        --fps 10
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

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
    return parser.parse_args()


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    input_video = Path(args.input_video).resolve()
    output_video = Path(args.output_video).resolve()

    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    check_ffmpeg()

    log.info("=" * 60)
    log.info("Fast-HaMeR Video Pipeline")
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
        )

        # 3. Pick best output per frame (rendered overlay or original fallback)
        assemble_rebuild_frames(frames_dir, rendered_dir, rebuild_dir, frame_count)

        # 4. Encode final video
        build_video(rebuild_dir, output_video, args.fps)

        log.info("=" * 60)
        log.info("Done. Output video: %s", output_video)
        log.info("=" * 60)

    finally:
        log.info("Cleaning up temporary directory: %s", tmpdir)
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
