import os
import json
import math
from statistics import mean, pstdev


def load_json_file(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(data, output_json_path):
    output_dir = os.path.dirname(output_json_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return output_json_path


def get_bbox_iou(box_a, box_b):
    if box_a is None or box_b is None:
        return None

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return None

    return inter_area / union_area


def euclidean_distance(point_a, point_b):
    if point_a is None or point_b is None:
        return None

    dx = float(point_b[0]) - float(point_a[0])
    dy = float(point_b[1]) - float(point_a[1])
    return math.sqrt(dx * dx + dy * dy)


def get_active_objects(frame_item):
    return frame_item.get("objects", [])


def build_track_index(frames):
    tracks = {}

    for frame_item in frames:
        frame_index = frame_item["frame_index"]
        timestamp_sec = frame_item.get("timestamp_sec")
        objects = get_active_objects(frame_item)

        for obj in objects:
            track_id = obj["track_id"]

            if track_id not in tracks:
                tracks[track_id] = []

            tracks[track_id].append(
                {
                    "frame_index": frame_index,
                    "timestamp_sec": timestamp_sec,
                    "class_name": obj.get("class_name"),
                    "bbox_xyxy": obj.get("bbox_xyxy"),
                    "mask_area_px": obj.get("mask_area_px"),
                    "mask_centroid_xy": obj.get("mask_centroid_xy"),
                    "is_new_track": obj.get("is_new_track", False),
                    "is_lost_track": obj.get("is_lost_track", False),
                }
            )

    for track_id in tracks:
        tracks[track_id].sort(key=lambda x: x["frame_index"])

    return tracks


def compute_track_fragmentation(track_frames):
    if len(track_frames) <= 1:
        return 0

    gaps = 0

    for i in range(1, len(track_frames)):
        previous_frame = track_frames[i - 1]["frame_index"]
        current_frame = track_frames[i]["frame_index"]

        if current_frame > previous_frame + 1:
            gaps += 1

    return gaps


def compute_track_metrics(track_id, track_frames, total_frames):
    frame_indices = [item["frame_index"] for item in track_frames]
    areas = [item["mask_area_px"] for item in track_frames if item.get("mask_area_px") is not None]
    centroids = [item["mask_centroid_xy"] for item in track_frames if item.get("mask_centroid_xy") is not None]
    bboxes = [item["bbox_xyxy"] for item in track_frames if item.get("bbox_xyxy") is not None]

    start_frame = frame_indices[0]
    end_frame = frame_indices[-1]
    track_length_frames = len(track_frames)
    visibility_ratio = track_length_frames / total_frames if total_frames > 0 else None

    mean_mask_area_px = mean(areas) if len(areas) > 0 else None
    mask_area_std_px = pstdev(areas) if len(areas) > 1 else 0.0 if len(areas) == 1 else None

    centroid_speeds = []
    trajectory_length_px = 0.0

    for i in range(1, len(track_frames)):
        prev_centroid = track_frames[i - 1].get("mask_centroid_xy")
        curr_centroid = track_frames[i].get("mask_centroid_xy")
        distance = euclidean_distance(prev_centroid, curr_centroid)

        if distance is not None:
            centroid_speeds.append(distance)
            trajectory_length_px += distance

    mean_centroid_speed_px_per_frame = mean(centroid_speeds) if len(centroid_speeds) > 0 else None
    max_centroid_speed_px_per_frame = max(centroid_speeds) if len(centroid_speeds) > 0 else None

    bbox_ious = []

    for i in range(1, len(track_frames)):
        prev_bbox = track_frames[i - 1].get("bbox_xyxy")
        curr_bbox = track_frames[i].get("bbox_xyxy")
        iou_value = get_bbox_iou(prev_bbox, curr_bbox)

        if iou_value is not None:
            bbox_ious.append(iou_value)

    mean_bbox_iou_temporal = mean(bbox_ious) if len(bbox_ious) > 0 else None

    fragmentation_count = compute_track_fragmentation(track_frames)
    fragmented = fragmentation_count > 0

    temporal_stability_score = None
    if mean_bbox_iou_temporal is not None:
        temporal_stability_score = mean_bbox_iou_temporal

    class_name = track_frames[0].get("class_name")

    return {
        "track_id": track_id,
        "class_name": class_name,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "track_length_frames": track_length_frames,
        "visibility_ratio": visibility_ratio,
        "mean_mask_area_px": mean_mask_area_px,
        "mask_area_std_px": mask_area_std_px,
        "mean_centroid_speed_px_per_frame": mean_centroid_speed_px_per_frame,
        "max_centroid_speed_px_per_frame": max_centroid_speed_px_per_frame,
        "trajectory_length_px": trajectory_length_px,
        "temporal_stability_score": temporal_stability_score,
        "fragmented": fragmented,
        "fragmentation_count": fragmentation_count,
        "mean_bbox_iou_temporal": mean_bbox_iou_temporal,
    }


def compute_occlusion_recovery_rate(frames):
    visibility_flags = []

    for frame_item in frames:
        visibility_flags.append(frame_item.get("objects_detected", 0) > 0)

    loss_count = 0
    recovery_count = 0

    for i in range(1, len(visibility_flags)):
        if visibility_flags[i - 1] and not visibility_flags[i]:
            loss_count += 1

            recovered = False
            for j in range(i + 1, len(visibility_flags)):
                if visibility_flags[j]:
                    recovered = True
                    break

            if recovered:
                recovery_count += 1

    if loss_count == 0:
        return None

    return recovery_count / loss_count


def build_per_frame_metrics(frames):
    per_frame_metrics = []

    previous_centroids_by_track = {}

    for frame_item in frames:
        active_objects = get_active_objects(frame_item)
        active_track_ids = []
        total_mask_area_px = 0
        speeds_this_frame = []

        current_centroids_by_track = {}

        for obj in active_objects:
            track_id = obj["track_id"]
            active_track_ids.append(track_id)

            area_value = obj.get("mask_area_px")
            if area_value is not None:
                total_mask_area_px += area_value

            centroid = obj.get("mask_centroid_xy")
            current_centroids_by_track[track_id] = centroid

            if track_id in previous_centroids_by_track:
                speed = euclidean_distance(previous_centroids_by_track[track_id], centroid)
                if speed is not None:
                    speeds_this_frame.append(speed)

        mean_speed = mean(speeds_this_frame) if len(speeds_this_frame) > 0 else None

        per_frame_metrics.append(
            {
                "frame_index": frame_item["frame_index"],
                "objects_detected": frame_item.get("objects_detected", 0),
                "active_track_ids": active_track_ids,
                "total_mask_area_px": total_mask_area_px,
                "mean_centroid_speed_px_per_frame": mean_speed,
            }
        )

        previous_centroids_by_track = current_centroids_by_track

    return per_frame_metrics


def build_summary_metrics(frames, track_metrics):
    total_frames = len(frames)
    frames_with_objects = sum(1 for frame_item in frames if frame_item.get("objects_detected", 0) > 0)
    frames_without_objects = total_frames - frames_with_objects
    detection_rate = frames_with_objects / total_frames if total_frames > 0 else None

    total_tracks = len(track_metrics)

    track_lengths = [item["track_length_frames"] for item in track_metrics]
    mean_track_length_frames = mean(track_lengths) if len(track_lengths) > 0 else None
    max_track_length_frames = max(track_lengths) if len(track_lengths) > 0 else None

    track_fragmentation_count = sum(item["fragmentation_count"] for item in track_metrics)

    mask_means = [item["mean_mask_area_px"] for item in track_metrics if item["mean_mask_area_px"] is not None]
    mean_mask_area_px = mean(mask_means) if len(mask_means) > 0 else None

    all_mask_areas = []
    for frame_item in frames:
        for obj in get_active_objects(frame_item):
            area_value = obj.get("mask_area_px")
            if area_value is not None:
                all_mask_areas.append(area_value)

    mask_area_std_px = pstdev(all_mask_areas) if len(all_mask_areas) > 1 else 0.0 if len(all_mask_areas) == 1 else None

    all_mean_speeds = [
        item["mean_centroid_speed_px_per_frame"]
        for item in track_metrics
        if item["mean_centroid_speed_px_per_frame"] is not None
    ]
    mean_centroid_speed_px_per_frame = mean(all_mean_speeds) if len(all_mean_speeds) > 0 else None

    all_max_speeds = [
        item["max_centroid_speed_px_per_frame"]
        for item in track_metrics
        if item["max_centroid_speed_px_per_frame"] is not None
    ]
    max_centroid_speed_px_per_frame = max(all_max_speeds) if len(all_max_speeds) > 0 else None

    all_bbox_ious = [
        item["mean_bbox_iou_temporal"]
        for item in track_metrics
        if item["mean_bbox_iou_temporal"] is not None
    ]
    mean_bbox_iou_temporal = mean(all_bbox_ious) if len(all_bbox_ious) > 0 else None

    occlusion_recovery_rate = compute_occlusion_recovery_rate(frames)

    return {
        "total_frames": total_frames,
        "frames_with_objects": frames_with_objects,
        "frames_without_objects": frames_without_objects,
        "detection_rate": detection_rate,
        "total_tracks": total_tracks,
        "mean_track_length_frames": mean_track_length_frames,
        "max_track_length_frames": max_track_length_frames,
        "track_fragmentation_count": track_fragmentation_count,
        "mean_mask_area_px": mean_mask_area_px,
        "mask_area_std_px": mask_area_std_px,
        "mean_centroid_speed_px_per_frame": mean_centroid_speed_px_per_frame,
        "max_centroid_speed_px_per_frame": max_centroid_speed_px_per_frame,
        "mean_bbox_iou_temporal": mean_bbox_iou_temporal,
        "occlusion_recovery_rate": occlusion_recovery_rate,
    }


def build_derived_metrics_json(raw_json, source_raw_json_path):
    run_info = raw_json["run"]
    frames = raw_json["frames"]

    track_index = build_track_index(frames)

    track_metrics = []
    for track_id, track_frames in track_index.items():
        track_metrics.append(
            compute_track_metrics(
                track_id=track_id,
                track_frames=track_frames,
                total_frames=len(frames),
            )
        )

    track_metrics.sort(key=lambda x: x["track_id"])

    per_frame_metrics = build_per_frame_metrics(frames)
    summary_metrics = build_summary_metrics(frames, track_metrics)

    return {
        "run": {
            "run_id": run_info["run_id"],
            "source_raw_json": source_raw_json_path,
            "pipeline_name": "sam2_video_segmentation_metrics",
            "created_at": run_info["created_at"],
            "metrics_schema_version": "1.0",
        },
        "summary_metrics": summary_metrics,
        "track_metrics": track_metrics,
        "per_frame_metrics": per_frame_metrics,
        "warnings": [
            "No semantic class verification beyond provided prompts",
            "Mask quality depends on initialization/prompt quality",
        ],
    }


def export_derived_metrics_json(raw_json_path, output_json_path=None):
    raw_json = load_json_file(raw_json_path)
    derived_json = build_derived_metrics_json(raw_json, raw_json_path)

    if output_json_path is None:
        base_dir = os.path.dirname(raw_json_path)
        base_name = os.path.basename(raw_json_path)
        name_without_ext = os.path.splitext(base_name)[0]
        output_json_path = os.path.join(base_dir, f"{name_without_ext}_metrics.json")

    save_json_file(derived_json, output_json_path)
    return output_json_path