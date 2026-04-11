"""
Compute derived motion/posture metrics from Fast-HaMeR raw hand-data JSON.

Reads the per-frame raw hand data exported by the pipeline and produces
a separate JSON with coverage, velocity, acceleration, angle, openness,
trajectory, and smoothness metrics.  No GPU or model dependencies needed.

Can be run standalone or called automatically from run_video_hamer.py.

Usage (standalone):
    python compute_derived_metrics.py \
        --raw_json  outputs/json/fast_hamer_abc123_raw_hand_data.json \
        --output    outputs/json/fast_hamer_abc123_derived_metrics.json
"""

import argparse
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("derived_metrics")

# ---------------------------------------------------------------------------
# OpenPose hand joint indices (21 joints, as output by MANO wrapper)
#
#  0  Wrist
#  1–4   Thumb  (CMC, MCP, IP, Tip)
#  5–8   Index  (MCP, PIP, DIP, Tip)
#  9–12  Middle (MCP, PIP, DIP, Tip)
# 13–16  Ring   (MCP, PIP, DIP, Tip)
# 17–20  Pinky  (MCP, PIP, DIP, Tip)
# ---------------------------------------------------------------------------
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_TIP = 16
PINKY_TIP = 20

ALL_FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

MAX_GAP_FRAMES = 3

# Wrist coordinates beyond this absolute magnitude (in any axis) are treated
# as degenerate model outputs and excluded from velocity / trajectory metrics.
# Normal MANO-space camera-Z values are ~5–50; outliers can be ~1e13.
OUTLIER_COORD_THRESHOLD = 500.0


# ---------------------------------------------------------------------------
# Vector maths (pure Python — no numpy dependency)
# ---------------------------------------------------------------------------

def _vsub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _vnorm(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _vdot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _angle_deg(a, vertex, c):
    """Angle (degrees) at *vertex* between rays vertex->a and vertex->c."""
    va = _vsub(a, vertex)
    vc = _vsub(c, vertex)
    na, nc = _vnorm(va), _vnorm(vc)
    if na < 1e-9 or nc < 1e-9:
        return None
    cos_a = max(-1.0, min(1.0, _vdot(va, vc) / (na * nc)))
    return math.degrees(math.acos(cos_a))


def _r(val, n=6):
    return round(val, n) if val is not None else None


def _rl(lst, n=6):
    return [round(v, n) for v in lst] if lst is not None else None


def _is_outlier(wrist):
    """True when any wrist coordinate exceeds the plausibility threshold."""
    return any(abs(v) > OUTLIER_COORD_THRESHOLD for v in wrist)


# ---------------------------------------------------------------------------
# A. Coverage / robustness
# ---------------------------------------------------------------------------

def _coverage(frames):
    total = len(frames)
    detected = sum(1 for f in frames if f["hands_detected"] > 0)
    longest = streak = 0
    first_det = last_det = None
    for f in frames:
        if f["hands_detected"] > 0:
            streak += 1
            longest = max(longest, streak)
            if first_det is None:
                first_det = f["frame_index"]
            last_det = f["frame_index"]
        else:
            streak = 0
    return {
        "total_frames": total,
        "frames_with_detection": detected,
        "frames_without_detection": total - detected,
        "detection_rate": round(detected / total, 4) if total else 0.0,
        "longest_detection_streak": longest,
        "first_detected_frame": first_det,
        "last_detected_frame": last_det,
    }


# ---------------------------------------------------------------------------
# Organise raw frames by hand side
# ---------------------------------------------------------------------------

def _by_side(frames):
    """Return {side: [entry, ...]} sorted by frame_index."""
    sides = {}
    for f in frames:
        for h in f.get("hands", []):
            s = h["hand_side"]
            sides.setdefault(s, []).append({
                "fi": f["frame_index"],
                "ts": f["timestamp_sec"],
                "j3d": h["joints_3d_cam"],
            })
    return sides


# ---------------------------------------------------------------------------
# B–G. Per-frame + summary metrics for one hand side
# ---------------------------------------------------------------------------

def _per_frame_hand(entries, fps):
    """Return (per_frame_dict, vel_cache, accel_cache) for one hand side.

    Three passes over the sorted *entries* list:
      1. Static posture (angles, openness) — per-frame, no temporal need.
      2. Velocities — consecutive pairs within gap threshold.
      3. Accelerations — consecutive pairs that both have velocity.
    """
    max_dt = MAX_GAP_FRAMES / fps
    n = len(entries)
    pf = {}

    # ---- pass 1: static posture ----
    for e in entries:
        j = e["j3d"]
        pf[e["fi"]] = {
            "wrist_position": _rl(j[WRIST]),
            "wrist_velocity": None,
            "wrist_speed": None,
            "wrist_acceleration_mag": None,
            "thumb_tip_speed": None,
            "index_tip_speed": None,
            "middle_tip_speed": None,
            "index_pip_flexion_deg": _r(
                _angle_deg(j[INDEX_MCP], j[INDEX_PIP], j[INDEX_DIP]), 2
            ),
            "middle_pip_flexion_deg": _r(
                _angle_deg(j[MIDDLE_MCP], j[MIDDLE_PIP], j[MIDDLE_DIP]), 2
            ),
            "thumb_index_spread_deg": _r(
                _angle_deg(j[THUMB_TIP], j[WRIST], j[INDEX_TIP]), 2
            ),
            "hand_openness": _r(
                sum(_vnorm(_vsub(j[ft], j[WRIST])) for ft in ALL_FINGERTIPS)
                / len(ALL_FINGERTIPS)
            ),
        }

    # ---- pass 2: velocities (skip outlier wrist positions) ----
    vc = {}
    outlier_count = 0
    for i in range(1, n):
        p, c = entries[i - 1], entries[i]
        dt = c["ts"] - p["ts"]
        if dt <= 0 or dt > max_dt:
            continue
        if _is_outlier(p["j3d"][WRIST]) or _is_outlier(c["j3d"][WRIST]):
            outlier_count += 1
            continue
        fi = c["fi"]
        jp, jc = p["j3d"], c["j3d"]

        wv = [(jc[WRIST][k] - jp[WRIST][k]) / dt for k in range(3)]
        pf[fi]["wrist_velocity"] = _rl(wv)
        pf[fi]["wrist_speed"] = _r(_vnorm(wv))
        vc[fi] = wv

        for key, idx in [("thumb_tip_speed", THUMB_TIP),
                         ("index_tip_speed", INDEX_TIP),
                         ("middle_tip_speed", MIDDLE_TIP)]:
            v = [(jc[idx][k] - jp[idx][k]) / dt for k in range(3)]
            pf[fi][key] = _r(_vnorm(v))

    # ---- pass 3: accelerations ----
    ac = {}
    for i in range(2, n):
        p, c = entries[i - 1], entries[i]
        dt = c["ts"] - p["ts"]
        fi, pfi = c["fi"], p["fi"]
        if dt <= 0 or dt > max_dt or fi not in vc or pfi not in vc:
            continue
        av = [(vc[fi][k] - vc[pfi][k]) / dt for k in range(3)]
        pf[fi]["wrist_acceleration_mag"] = _r(_vnorm(av))
        ac[fi] = av

    return pf, vc, ac, outlier_count


def _summarise_hand(pf, entries, vc, ac, fps):
    """Aggregate per-frame metrics into summary statistics for one hand side."""
    max_dt = MAX_GAP_FRAMES / fps
    ts_of = {e["fi"]: e["ts"] for e in entries}

    sw, sth, six, smd, aw = [], [], [], [], []
    a_idx, a_mid, a_sp, op_vals = [], [], [], []

    for fi in sorted(pf):
        m = pf[fi]
        if m["wrist_speed"] is not None:
            sw.append(m["wrist_speed"])
        if m["thumb_tip_speed"] is not None:
            sth.append(m["thumb_tip_speed"])
        if m["index_tip_speed"] is not None:
            six.append(m["index_tip_speed"])
        if m["middle_tip_speed"] is not None:
            smd.append(m["middle_tip_speed"])
        if m["wrist_acceleration_mag"] is not None:
            aw.append(m["wrist_acceleration_mag"])
        if m["index_pip_flexion_deg"] is not None:
            a_idx.append(m["index_pip_flexion_deg"])
        if m["middle_pip_flexion_deg"] is not None:
            a_mid.append(m["middle_pip_flexion_deg"])
        if m["thumb_index_spread_deg"] is not None:
            a_sp.append(m["thumb_index_spread_deg"])
        if m["hand_openness"] is not None:
            op_vals.append(m["hand_openness"])

    # F. Trajectory lengths (skip outlier wrist positions)
    traj_w = traj_ix = 0.0
    for i in range(1, len(entries)):
        dt = entries[i]["ts"] - entries[i - 1]["ts"]
        if dt <= 0 or dt > max_dt:
            continue
        jp, jc = entries[i - 1]["j3d"], entries[i]["j3d"]
        if _is_outlier(jp[WRIST]) or _is_outlier(jc[WRIST]):
            continue
        traj_w += _vnorm(_vsub(jc[WRIST], jp[WRIST]))
        traj_ix += _vnorm(_vsub(jc[INDEX_TIP], jp[INDEX_TIP]))

    # G. Jerk (3rd derivative of position) from consecutive acceleration vectors
    jerks = []
    akeys = sorted(ac)
    for i in range(1, len(akeys)):
        fi, pfi = akeys[i], akeys[i - 1]
        dt = ts_of[fi] - ts_of[pfi]
        if 0 < dt <= max_dt:
            jv = [(ac[fi][k] - ac[pfi][k]) / dt for k in range(3)]
            jerks.append(_vnorm(jv))

    def _stats(vals, d=6):
        if not vals:
            return None
        return {
            "mean": round(sum(vals) / len(vals), d),
            "min": round(min(vals), d),
            "max": round(max(vals), d),
            "count": len(vals),
        }

    def _spd(vals):
        if not vals:
            return {"mean_speed": None, "max_speed": None}
        return {
            "mean_speed": _r(sum(vals) / len(vals)),
            "max_speed": _r(max(vals)),
        }

    return {
        "frames_present": len(entries),
        "wrist": {
            "mean_speed": _r(sum(sw) / len(sw)) if sw else None,
            "max_speed": _r(max(sw)) if sw else None,
            "mean_acceleration": _r(sum(aw) / len(aw)) if aw else None,
            "max_acceleration": _r(max(aw)) if aw else None,
            "trajectory_length": _r(traj_w),
        },
        "fingertips": {
            "thumb_tip": _spd(sth),
            "index_tip": {**_spd(six), "trajectory_length": _r(traj_ix)},
            "middle_tip": _spd(smd),
        },
        "angles": {
            "index_pip_flexion_deg": _stats(a_idx, 2),
            "middle_pip_flexion_deg": _stats(a_mid, 2),
            "thumb_index_spread_deg": _stats(a_sp, 2),
        },
        "hand_openness": _stats(op_vals),
        "smoothness": {
            "mean_jerk_magnitude": _r(sum(jerks) / len(jerks)) if jerks else None,
            "max_jerk_magnitude": _r(max(jerks)) if jerks else None,
            "jerk_samples": len(jerks),
            "note": (
                "Jerk = 3rd derivative of wrist position (model-units/s^3). "
                "Lower values indicate smoother motion."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Main computation entry point
# ---------------------------------------------------------------------------

def compute_metrics(raw_json_path, summary_json_path=None, output_video_path=None):
    """Read raw hand-data JSON, compute all metrics, return result dict."""
    raw_json_path = Path(raw_json_path)
    with open(raw_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    run_info = raw["run"]
    frames = raw["frames"]
    fps = run_info["fps_used_for_processing"]

    log.info("Deriving metrics from %d frames (%d FPS) ...", len(frames), fps)

    cov = _coverage(frames)
    sides = _by_side(frames)
    warns = []

    hand_summaries = {}
    pf_by_side = {}

    total_outlier_pairs = 0
    for side, entries in sides.items():
        pf, vc, ac, outliers = _per_frame_hand(entries, fps)
        total_outlier_pairs += outliers
        pf_by_side[side] = pf
        hand_summaries[side] = _summarise_hand(pf, entries, vc, ac, fps)

        js = hand_summaries[side]["smoothness"]["jerk_samples"]
        if js < 3:
            warns.append(
                f"{side} hand: {js} jerk samples — smoothness estimate "
                "may be unreliable"
            )
        if outliers > 0:
            warns.append(
                f"{side} hand: {outliers} frame pairs skipped for motion "
                f"metrics due to degenerate depth estimates "
                f"(wrist coordinate > {OUTLIER_COORD_THRESHOLD})"
            )

    if cov["frames_without_detection"] > 0:
        warns.append(
            f"{cov['frames_without_detection']}/{cov['total_frames']} "
            "frames had no hand detection"
        )
    if not sides:
        warns.append(
            "No hands detected in any frame — all per-hand metrics are empty"
        )

    # Assemble per-frame list (one entry for every processed frame)
    pfm = []
    for f in frames:
        rec = {
            "frame_index": f["frame_index"],
            "timestamp_sec": f["timestamp_sec"],
            "hands_detected": f["hands_detected"],
            "hands": {},
        }
        for side, pf_dict in pf_by_side.items():
            if f["frame_index"] in pf_dict:
                rec["hands"][side] = pf_dict[f["frame_index"]]
        pfm.append(rec)

    src = {"raw_hand_data_json": str(raw_json_path)}
    if summary_json_path:
        src["summary_json"] = str(summary_json_path)
    if output_video_path:
        src["output_video"] = str(output_video_path)

    warns.extend([
        "Speeds/accelerations are in MANO model-units/s "
        "(approximate hand-scale metres)",
        "Angles are in degrees",
        "Hand openness = mean fingertip-to-wrist distance (model units)",
        "Trajectory lengths = cumulative Euclidean displacement (model units)",
        f"Velocity gap threshold: {MAX_GAP_FRAMES} frame intervals "
        f"({MAX_GAP_FRAMES / fps:.4f} s)",
    ])

    return {
        "run": {
            "run_id": run_info["run_id"],
            "pipeline_name": run_info["pipeline_name"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metrics_schema_version": "1.0",
        },
        "source_files": src,
        "summary_metrics": {
            "coverage": cov,
            "per_hand": hand_summaries,
        },
        "per_frame_metrics": pfm,
        "warnings": warns,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Derive motion/posture metrics from Fast-HaMeR raw hand-data JSON."
        ),
    )
    ap.add_argument(
        "--raw_json", required=True,
        help="Path to the raw hand-data JSON produced by the pipeline",
    )
    ap.add_argument(
        "--output", required=True,
        help="Path for the derived-metrics JSON output",
    )
    ap.add_argument(
        "--summary_json", default=None,
        help="Path to the summary JSON (recorded in source_files)",
    )
    ap.add_argument(
        "--output_video", default=None,
        help="Path to the output video (recorded in source_files)",
    )
    args = ap.parse_args()

    result = compute_metrics(
        args.raw_json, args.summary_json, args.output_video,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.info("Derived metrics saved -> %s", out)


if __name__ == "__main__":
    main()
