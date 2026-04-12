# import os
# import cv2
# import json
# import uuid
# import time
# import shutil
# import datetime
# import subprocess
# import numpy as np
# import torch
# import imageio_ffmpeg
# from contextlib import nullcontext

# from sam2.build_sam import build_sam2_video_predictor
# from video_utils import get_video_info, get_analysis_window, extract_frame_range_to_folder


# VIDEO_CHECKPOINT = "checkpoints/sam2_hiera_tiny.pt"
# VIDEO_MODEL_CFG = "configs/sam2/sam2_hiera_t.yaml"
# VIDEO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# _VIDEO_PREDICTOR = None


# def get_video_predictor():
#     global _VIDEO_PREDICTOR

#     if _VIDEO_PREDICTOR is None:
#         print("SAM video running on:", VIDEO_DEVICE)  # ✅ ADD THIS LINE

#         _VIDEO_PREDICTOR = build_sam2_video_predictor(
#             VIDEO_MODEL_CFG,
#             VIDEO_CHECKPOINT,
#             device=VIDEO_DEVICE,
#         )

#     return _VIDEO_PREDICTOR


# def get_autocast_context():
#     if VIDEO_DEVICE == "cuda":
#         return torch.autocast("cuda", dtype=torch.bfloat16)
#     return nullcontext()


# def ensure_clean_dir(folder_path):
#     if os.path.exists(folder_path):
#         shutil.rmtree(folder_path)

#     os.makedirs(folder_path, exist_ok=True)


# def mask_to_overlay(image_bgr, mask):
#     overlay = image_bgr.copy()
#     overlay[mask.astype(bool)] = [0, 255, 0]
#     return overlay


# def mask_area(mask):
#     return int(mask.astype(bool).sum())


# def mask_centroid(mask):
#     ys, xs = mask.astype(bool).nonzero()

#     if len(xs) == 0 or len(ys) == 0:
#         return None

#     return int(xs.mean()), int(ys.mean())


# def mask_bbox(mask):
#     ys, xs = mask.astype(bool).nonzero()

#     if len(xs) == 0 or len(ys) == 0:
#         return None

#     x_min = int(xs.min())
#     y_min = int(ys.min())
#     x_max = int(xs.max())
#     y_max = int(ys.max())

#     return x_min, y_min, x_max, y_max


# def save_binary_mask_image(mask, output_path):
#     if mask is None:
#         return None

#     output_dir = os.path.dirname(output_path)
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)

#     mask_image = (mask.astype(np.uint8) * 255)
#     cv2.imwrite(output_path, mask_image)
#     return os.path.abspath(output_path)


# def write_temp_video_from_frames(frame_paths, output_video_path, fps):
#     if len(frame_paths) == 0:
#         return False

#     first_frame = cv2.imread(frame_paths[0])

#     if first_frame is None:
#         return False

#     height, width = first_frame.shape[:2]

#     output_dir = os.path.dirname(output_video_path)
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)

#     writer = cv2.VideoWriter(
#         output_video_path,
#         cv2.VideoWriter_fourcc(*"mp4v"),
#         fps,
#         (width, height),
#     )

#     for frame_path in frame_paths:
#         frame = cv2.imread(frame_path)

#         if frame is None:
#             continue

#         writer.write(frame)

#     writer.release()
#     return True


# def convert_video_to_web_mp4(input_video_path, output_video_path):
#     ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

#     command = [
#         ffmpeg_exe,
#         "-y",
#         "-i", input_video_path,
#         "-c:v", "libx264",
#         "-pix_fmt", "yuv420p",
#         "-movflags", "+faststart",
#         "-an",
#         output_video_path,
#     ]

#     result = subprocess.run(
#         command,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         text=True,
#     )

#     return result.returncode == 0


# def write_video_from_frames(frame_paths, output_video_path, fps):
#     temp_output_video_path = output_video_path.replace(".mp4", "_temp.mp4")
#     web_output_video_path = output_video_path.replace(".mp4", "_web.mp4")

#     temp_written = write_temp_video_from_frames(
#         frame_paths=frame_paths,
#         output_video_path=temp_output_video_path,
#         fps=fps,
#     )

#     if not temp_written:
#         return None

#     converted = convert_video_to_web_mp4(
#         input_video_path=temp_output_video_path,
#         output_video_path=web_output_video_path,
#     )

#     if converted:
#         if os.path.exists(temp_output_video_path):
#             os.remove(temp_output_video_path)
#         return os.path.abspath(web_output_video_path)

#     return os.path.abspath(temp_output_video_path)


# def get_binary_mask_for_object(out_obj_ids, out_mask_logits, target_obj_id=1):
#     if out_mask_logits is None:
#         return None

#     target_index = None

#     for i, current_obj_id in enumerate(out_obj_ids):
#         if int(current_obj_id) == int(target_obj_id):
#             target_index = i
#             break

#     if target_index is None:
#         if len(out_obj_ids) == 0:
#             return None
#         target_index = 0

#     mask_tensor = out_mask_logits[target_index]

#     if hasattr(mask_tensor, "detach"):
#         mask_np = mask_tensor.detach().cpu().numpy()
#     else:
#         mask_np = np.array(mask_tensor)

#     mask_np = np.squeeze(mask_np)
#     return (mask_np > 0.0).astype(np.uint8)


# def store_prompt_frame_mask(masks_by_local_frame, frame_idx, out_obj_ids, out_mask_logits, target_obj_id=1):
#     binary_mask = get_binary_mask_for_object(
#         out_obj_ids,
#         out_mask_logits,
#         target_obj_id=target_obj_id,
#     )

#     if binary_mask is None:
#         return

#     masks_by_local_frame[int(frame_idx)] = binary_mask


# def collect_propagation_masks(predictor, inference_state, selected_local_frame_index, target_obj_id=1, reverse=False):
#     collected = {}

#     try:
#         iterator = predictor.propagate_in_video(
#             inference_state,
#             start_frame_idx=selected_local_frame_index,
#             reverse=reverse,
#         )
#     except TypeError:
#         iterator = predictor.propagate_in_video(inference_state)

#     for out_frame_idx, out_obj_ids, out_mask_logits in iterator:
#         binary_mask = get_binary_mask_for_object(
#             out_obj_ids,
#             out_mask_logits,
#             target_obj_id=target_obj_id,
#         )

#         if binary_mask is None:
#             continue

#         collected[int(out_frame_idx)] = binary_mask

#     return collected


# def prepare_single_video_for_tracking(
#     video_path,
#     video_role,
#     run_root_folder,
#     analysis_mode="full",
#     max_seconds=15,
#     frame_stride=1,
# ):
#     video_info = get_video_info(video_path)

#     if "error" in video_info:
#         return {"error": video_info["error"]}

#     analysis_window = get_analysis_window(
#         video_path,
#         analysis_mode=analysis_mode,
#         max_seconds=max_seconds,
#     )

#     if "error" in analysis_window:
#         return {"error": analysis_window["error"]}

#     output_folder = os.path.join(run_root_folder, f"{video_role}_frames")

#     extraction = extract_frame_range_to_folder(
#         video_path=video_path,
#         output_folder=output_folder,
#         start_frame_index=analysis_window["start_frame_index"],
#         end_frame_index=analysis_window["end_frame_index"],
#         prefix="frame",
#         frame_stride=frame_stride,
#     )

#     if "error" in extraction:
#         return {"error": extraction["error"]}

#     return {
#         "video_role": video_role,
#         "source_path": video_path,
#         "frame_count": video_info["frame_count"],
#         "fps": video_info["fps"],
#         "width": video_info["width"],
#         "height": video_info["height"],
#         "duration_seconds": video_info["duration_seconds"],
#         "analysis_start_frame_index": analysis_window["start_frame_index"],
#         "analysis_end_frame_index": analysis_window["end_frame_index"],
#         "analysis_frame_count": extraction["saved_frame_count"],
#         "analysis_start_time_seconds": analysis_window["start_time_seconds"],
#         "analysis_end_time_seconds": analysis_window["end_time_seconds"],
#         "frames_folder": extraction["output_folder"],
#         "saved_frame_count": extraction["saved_frame_count"],
#         "saved_paths": extraction["saved_paths"],
#         "saved_absolute_frame_indices": extraction["saved_absolute_frame_indices"],
#         "frame_stride": extraction["frame_stride"],
#     }


# def sanitize_prompt_time(
#     video_path,
#     selected_time_seconds,
#     analysis_mode="full",
#     max_seconds=15,
#     frame_stride=1,
# ):
#     info = get_video_info(video_path)

#     if "error" in info:
#         return {"error": info["error"]}

#     window = get_analysis_window(
#         video_path,
#         analysis_mode=analysis_mode,
#         max_seconds=max_seconds,
#     )

#     if "error" in window:
#         return {"error": window["error"]}

#     fps = info["fps"]
#     frame_count = info["frame_count"]

#     if selected_time_seconds is None:
#         selected_time_seconds = 0.0

#     requested_time_seconds = float(selected_time_seconds)

#     if requested_time_seconds < 0:
#         requested_time_seconds = 0.0

#     if requested_time_seconds > window["end_time_seconds"]:
#         requested_time_seconds = window["end_time_seconds"]

#     requested_absolute_frame_index = 0
#     if fps > 0:
#         requested_absolute_frame_index = int(requested_time_seconds * fps)

#     if requested_absolute_frame_index >= frame_count:
#         requested_absolute_frame_index = frame_count - 1

#     if frame_stride is None:
#         frame_stride = 1

#     frame_stride = int(frame_stride)

#     if frame_stride <= 0:
#         return {"error": "frame_stride doit être >= 1"}

#     sampled_absolute_frame_indices = list(
#         range(
#             window["start_frame_index"],
#             window["end_frame_index"] + 1,
#             frame_stride,
#         )
#     )

#     if len(sampled_absolute_frame_indices) == 0:
#         return {"error": "Aucune frame échantillonnée disponible"}

#     snapped_absolute_frame_index = min(
#         sampled_absolute_frame_indices,
#         key=lambda x: abs(x - requested_absolute_frame_index),
#     )

#     local_frame_index = sampled_absolute_frame_indices.index(snapped_absolute_frame_index)

#     snapped_time_seconds = requested_time_seconds
#     if fps > 0:
#         snapped_time_seconds = snapped_absolute_frame_index / fps

#     return {
#         "requested_time_seconds": requested_time_seconds,
#         "requested_absolute_frame_index": requested_absolute_frame_index,
#         "selected_time_seconds": snapped_time_seconds,
#         "absolute_frame_index": snapped_absolute_frame_index,
#         "local_frame_index": local_frame_index,
#         "analysis_start_frame_index": window["start_frame_index"],
#         "analysis_end_frame_index": window["end_frame_index"],
#         "frame_stride": frame_stride,
#     }


# def run_tracking_on_prepared_video(
#     prepared_video,
#     selected_local_frame_index,
#     point_xy,
#     run_root_folder,
#     target_obj_id=1,
# ):
#     predictor = get_video_predictor()

#     frames_folder = prepared_video["frames_folder"]
#     saved_paths = prepared_video["saved_paths"]
#     saved_absolute_frame_indices = prepared_video["saved_absolute_frame_indices"]
#     fps = prepared_video["fps"]
#     video_role = prepared_video["video_role"]

#     overlay_folder = os.path.join(run_root_folder, f"{video_role}_overlay_frames")
#     mask_folder = os.path.join(run_root_folder, f"{video_role}_mask_frames")
#     ensure_clean_dir(overlay_folder)
#     ensure_clean_dir(mask_folder)

#     warnings = []
#     masks_by_local_frame = {}

#     points = np.array([[point_xy[0], point_xy[1]]], dtype=np.float32)
#     labels = np.array([1], np.int32)

#     with torch.inference_mode():
#         with get_autocast_context():
#             forward_state = predictor.init_state(video_path=frames_folder)

#             _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
#                 inference_state=forward_state,
#                 frame_idx=int(selected_local_frame_index),
#                 obj_id=int(target_obj_id),
#                 points=points,
#                 labels=labels,
#             )

#             store_prompt_frame_mask(
#                 masks_by_local_frame,
#                 selected_local_frame_index,
#                 out_obj_ids,
#                 out_mask_logits,
#                 target_obj_id=target_obj_id,
#             )

#             forward_masks = collect_propagation_masks(
#                 predictor,
#                 forward_state,
#                 selected_local_frame_index,
#                 target_obj_id=target_obj_id,
#                 reverse=False,
#             )

#             masks_by_local_frame.update(forward_masks)

#     try:
#         with torch.inference_mode():
#             with get_autocast_context():
#                 backward_state = predictor.init_state(video_path=frames_folder)

#                 _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
#                     inference_state=backward_state,
#                     frame_idx=int(selected_local_frame_index),
#                     obj_id=int(target_obj_id),
#                     points=points,
#                     labels=labels,
#                 )

#                 store_prompt_frame_mask(
#                     masks_by_local_frame,
#                     selected_local_frame_index,
#                     out_obj_ids,
#                     out_mask_logits,
#                     target_obj_id=target_obj_id,
#                 )

#                 backward_masks = collect_propagation_masks(
#                     predictor,
#                     backward_state,
#                     selected_local_frame_index,
#                     target_obj_id=target_obj_id,
#                     reverse=True,
#                 )

#                 for k, v in backward_masks.items():
#                     if k not in masks_by_local_frame:
#                         masks_by_local_frame[k] = v
#     except Exception as e:
#         warnings.append(f"reverse_tracking_failed: {str(e)}")

#     overlay_paths = []
#     frame_results = []

#     for local_idx, frame_path in enumerate(saved_paths):
#         frame = cv2.imread(frame_path)

#         if frame is None:
#             continue

#         absolute_frame_index = saved_absolute_frame_indices[local_idx]
#         time_seconds = None
#         if fps > 0:
#             time_seconds = absolute_frame_index / fps

#         if local_idx in masks_by_local_frame:
#             mask = masks_by_local_frame[local_idx]
#             object_present = bool(mask.astype(bool).any())
#         else:
#             mask = None
#             object_present = False

#         saved_mask_path = None

#         if mask is not None and object_present:
#             overlay = mask_to_overlay(frame, mask)
#             area_value = mask_area(mask)
#             centroid_value = mask_centroid(mask)
#             bbox_value = mask_bbox(mask)

#             mask_filename = f"frame_{absolute_frame_index:06d}_track_{target_obj_id:03d}.png"
#             mask_path = os.path.join(mask_folder, mask_filename)
#             saved_mask_path = save_binary_mask_image(mask, mask_path)
#         else:
#             overlay = frame
#             area_value = None
#             centroid_value = None
#             bbox_value = None

#         overlay_path = os.path.join(overlay_folder, f"overlay_{local_idx:06d}.jpg")
#         cv2.imwrite(overlay_path, overlay)
#         overlay_paths.append(overlay_path)

#         frame_results.append(
#             {
#                 "local_frame_index": local_idx,
#                 "absolute_frame_index": absolute_frame_index,
#                 "time_seconds": time_seconds,
#                 "object_present": object_present,
#                 "mask_area": area_value,
#                 "centroid": centroid_value,
#                 "bbox": bbox_value,
#                 "mask_path": saved_mask_path,
#             }
#         )

#     output_video_path = os.path.join(run_root_folder, f"{video_role}_annotated.mp4")
#     final_video_path = write_video_from_frames(overlay_paths, output_video_path, fps)

#     if final_video_path is None:
#         warnings.append("annotated_video_not_written")

#     detected_frame_count = 0
#     for item in frame_results:
#         if item["object_present"]:
#             detected_frame_count += 1

#     return {
#         "video_role": video_role,
#         "annotated_video_path": final_video_path,
#         "overlay_folder": overlay_folder,
#         "mask_folder": mask_folder,
#         "overlay_frame_count": len(overlay_paths),
#         "detected_frame_count": detected_frame_count,
#         "frame_results": frame_results,
#         "warnings": warnings,
#     }


# def save_json_file(data, output_json_path):
#     output_dir = os.path.dirname(output_json_path)
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)

#     with open(output_json_path, "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)

#     return output_json_path


# def build_raw_json_for_video(
#     run_id,
#     created_at,
#     prepared_video,
#     tracking_result,
#     prompt_info,
#     model_name="sam2_hiera_tiny",
# ):
#     source_path = prepared_video["source_path"]
#     filename = os.path.basename(source_path)
#     fps = prepared_video["fps"]

#     detected_absolute_frames = []
#     for item in tracking_result["frame_results"]:
#         if item["object_present"]:
#             detected_absolute_frames.append(item["absolute_frame_index"])

#     first_detected_frame = None
#     last_detected_frame = None

#     if len(detected_absolute_frames) > 0:
#         first_detected_frame = min(detected_absolute_frames)
#         last_detected_frame = max(detected_absolute_frames)

#     frames = []

#     for item in tracking_result["frame_results"]:
#         objects = []

#         if item["object_present"]:
#             temporal_source = "propagated"
#             if int(item["absolute_frame_index"]) == int(prompt_info["absolute_frame_index"]):
#                 temporal_source = "prompted"

#             centroid_xy = None
#             if item["centroid"] is not None:
#                 centroid_xy = [float(item["centroid"][0]), float(item["centroid"][1])]

#             bbox_xyxy = None
#             if item["bbox"] is not None:
#                 bbox_xyxy = [int(v) for v in item["bbox"]]

#             objects.append(
#                 {
#                     "track_id": 1,
#                     "class_name": "prompted_object",
#                     "class_confidence": None,
#                     "bbox_xyxy": bbox_xyxy,
#                     "bbox_confidence": None,
#                     "mask_area_px": item["mask_area"],
#                     "mask_bbox_xyxy": bbox_xyxy,
#                     "mask_centroid_xy": centroid_xy,
#                     "mask_iou_confidence": None,
#                     "mask_path": item.get("mask_path"),
#                     "temporal_source": temporal_source,
#                     "is_new_track": bool(item["absolute_frame_index"] == first_detected_frame),
#                     "is_lost_track": bool(item["absolute_frame_index"] == last_detected_frame),
#                 }
#             )

#         frames.append(
#             {
#                 "frame_index": int(item["absolute_frame_index"]),
#                 "timestamp_sec": float(item["time_seconds"]) if item["time_seconds"] is not None else None,
#                 "objects_detected": len(objects),
#                 "objects": objects,
#             }
#         )

#     raw_json = {
#         "run": {
#             "run_id": run_id,
#             "pipeline_name": "sam2_video_segmentation",
#             "model_name": model_name,
#             "created_at": created_at,
#             "input_video": filename,
#             "fps_used_for_processing": fps,
#             "total_frames": prepared_video["analysis_frame_count"],
#             "raw_schema_version": "1.0",
#         },
#         "video_metadata": {
#             "filename": filename,
#             "path": source_path,
#             "width": prepared_video["width"],
#             "height": prepared_video["height"],
#             "fps": fps,
#             "duration_sec": prepared_video["duration_seconds"],
#         },
#         "frames": frames,
#     }

#     return raw_json


# def export_raw_json_for_video(run_root_folder, video_role, raw_json):
#     json_folder = os.path.join(run_root_folder, "json")
#     output_json_path = os.path.join(json_folder, f"{video_role}_sam2_raw.json")
#     save_json_file(raw_json, output_json_path)
#     return output_json_path


# def track_two_videos_from_selected_points(
#     expert_video_path,
#     learner_video_path,
#     expert_time_seconds,
#     learner_time_seconds,
#     expert_point_xy,
#     learner_point_xy,
#     analysis_mode="full",
#     max_seconds=15,
#     frame_stride=1,
#     output_root="outputs/tmp_tracking_runs",
# ):
#     started_at = time.time()
#     created_at = datetime.datetime.now().isoformat()

#     run_id = uuid.uuid4().hex
#     run_root_folder = os.path.join(output_root, run_id)
#     os.makedirs(run_root_folder, exist_ok=True)

#     expert_prompt = sanitize_prompt_time(
#         expert_video_path,
#         expert_time_seconds,
#         analysis_mode=analysis_mode,
#         max_seconds=max_seconds,
#         frame_stride=frame_stride,
#     )
#     if "error" in expert_prompt:
#         return {"error": f"Expert: {expert_prompt['error']}"}

#     learner_prompt = sanitize_prompt_time(
#         learner_video_path,
#         learner_time_seconds,
#         analysis_mode=analysis_mode,
#         max_seconds=max_seconds,
#         frame_stride=frame_stride,
#     )
#     if "error" in learner_prompt:
#         return {"error": f"Learner: {learner_prompt['error']}"}

#     expert_prepared = prepare_single_video_for_tracking(
#         video_path=expert_video_path,
#         video_role="expert",
#         run_root_folder=run_root_folder,
#         analysis_mode=analysis_mode,
#         max_seconds=max_seconds,
#         frame_stride=frame_stride,
#     )
#     if "error" in expert_prepared:
#         return {"error": f"Expert: {expert_prepared['error']}"}

#     learner_prepared = prepare_single_video_for_tracking(
#         video_path=learner_video_path,
#         video_role="learner",
#         run_root_folder=run_root_folder,
#         analysis_mode=analysis_mode,
#         max_seconds=max_seconds,
#         frame_stride=frame_stride,
#     )
#     if "error" in learner_prepared:
#         return {"error": f"Learner: {learner_prepared['error']}"}

#     expert_tracking = run_tracking_on_prepared_video(
#         prepared_video=expert_prepared,
#         selected_local_frame_index=expert_prompt["local_frame_index"],
#         point_xy=expert_point_xy,
#         run_root_folder=run_root_folder,
#         target_obj_id=1,
#     )

#     learner_tracking = run_tracking_on_prepared_video(
#         prepared_video=learner_prepared,
#         selected_local_frame_index=learner_prompt["local_frame_index"],
#         point_xy=learner_point_xy,
#         run_root_folder=run_root_folder,
#         target_obj_id=1,
#     )

#     expert_raw_json = build_raw_json_for_video(
#         run_id=run_id,
#         created_at=created_at,
#         prepared_video=expert_prepared,
#         tracking_result=expert_tracking,
#         prompt_info=expert_prompt,
#         model_name="sam2_hiera_tiny",
#     )

#     learner_raw_json = build_raw_json_for_video(
#         run_id=run_id,
#         created_at=created_at,
#         prepared_video=learner_prepared,
#         tracking_result=learner_tracking,
#         prompt_info=learner_prompt,
#         model_name="sam2_hiera_tiny",
#     )

#     expert_raw_json_path = export_raw_json_for_video(
#         run_root_folder=run_root_folder,
#         video_role="expert",
#         raw_json=expert_raw_json,
#     )

#     learner_raw_json_path = export_raw_json_for_video(
#         run_root_folder=run_root_folder,
#         video_role="learner",
#         raw_json=learner_raw_json,
#     )

#     processing_time_sec = time.time() - started_at

#     warnings = ["score_not_computed"]

#     for warning in expert_tracking["warnings"]:
#         warnings.append(f"expert_{warning}")

#     for warning in learner_tracking["warnings"]:
#         warnings.append(f"learner_{warning}")

#     return {
#         "run": {
#             "run_id": run_id,
#             "pipeline_name": "sam2_video_point_tracking",
#             "processing_time_sec": processing_time_sec,
#             "created_at": created_at,
#             "component_notes": {
#                 "sam2_video": f"device={VIDEO_DEVICE}; analysis_mode={analysis_mode}; max_seconds={max_seconds}; frame_stride={frame_stride}",
#                 "prompting": "single_positive_point_per_video",
#                 "raw_json_export": "completed",
#             }
#         },
#         "expert_video": {
#             "path": expert_video_path,
#             "annotated_video_path": expert_tracking["annotated_video_path"],
#             "raw_json_path": expert_raw_json_path,
#             "duration_sec": expert_prepared["duration_seconds"],
#             "fps": expert_prepared["fps"],
#             "width": expert_prepared["width"],
#             "height": expert_prepared["height"],
#             "analysis_start_frame_index": expert_prepared["analysis_start_frame_index"],
#             "analysis_end_frame_index": expert_prepared["analysis_end_frame_index"],
#             "analysis_frame_count": expert_prepared["analysis_frame_count"],
#             "requested_time_seconds": expert_prompt["requested_time_seconds"],
#             "selected_time_seconds": expert_prompt["selected_time_seconds"],
#             "requested_absolute_frame_index": expert_prompt["requested_absolute_frame_index"],
#             "selected_absolute_frame_index": expert_prompt["absolute_frame_index"],
#             "selected_local_frame_index": expert_prompt["local_frame_index"],
#             "frame_stride": frame_stride,
#             "selected_point_xy": expert_point_xy,
#             "detected_frame_count": expert_tracking["detected_frame_count"],
#         },
#         "learner_video": {
#             "path": learner_video_path,
#             "annotated_video_path": learner_tracking["annotated_video_path"],
#             "raw_json_path": learner_raw_json_path,
#             "duration_sec": learner_prepared["duration_seconds"],
#             "fps": learner_prepared["fps"],
#             "width": learner_prepared["width"],
#             "height": learner_prepared["height"],
#             "analysis_start_frame_index": learner_prepared["analysis_start_frame_index"],
#             "analysis_end_frame_index": learner_prepared["analysis_end_frame_index"],
#             "analysis_frame_count": learner_prepared["analysis_frame_count"],
#             "requested_time_seconds": learner_prompt["requested_time_seconds"],
#             "selected_time_seconds": learner_prompt["selected_time_seconds"],
#             "requested_absolute_frame_index": learner_prompt["requested_absolute_frame_index"],
#             "selected_absolute_frame_index": learner_prompt["absolute_frame_index"],
#             "selected_local_frame_index": learner_prompt["local_frame_index"],
#             "frame_stride": frame_stride,
#             "selected_point_xy": learner_point_xy,
#             "detected_frame_count": learner_tracking["detected_frame_count"],
#         },
#         "overall_score": None,
#         "metrics": {
#             "joint_angle_deviation": None,
#             "trajectory_deviation": None,
#             "velocity_difference": None,
#             "tool_alignment_deviation": None,
#             "dtw_cost": None,
#             "semantic_similarity": None,
#             "optical_flow_similarity": None,
#             "extra": {
#                 "expert_detected_frame_count": expert_tracking["detected_frame_count"],
#                 "learner_detected_frame_count": learner_tracking["detected_frame_count"],
#             }
#         },
#         "confidences": {
#             "overall": None,
#             "same_task": None,
#             "score": None,
#             "explanation": None
#         },
#         "explanation": {
#             "text": "SAM 2 video tracking completed and raw JSON exported.",
#             "strengths": [],
#             "weaknesses": [],
#             "raw_vlm_output": None,
#             "structured_notes": {
#                 "expert_overlay_frame_count": expert_tracking["overlay_frame_count"],
#                 "learner_overlay_frame_count": learner_tracking["overlay_frame_count"],
#                 "expert_raw_json_path": expert_raw_json_path,
#                 "learner_raw_json_path": learner_raw_json_path,
#             }
#         },
#         "warnings": warnings
#     }


import os
import cv2
import json
import uuid
import time
import shutil
import datetime
import subprocess
import numpy as np
import torch
import imageio_ffmpeg
from contextlib import nullcontext

from sam2.build_sam import build_sam2_video_predictor
from video_utils import get_video_info, get_analysis_window, extract_frame_range_to_folder


VIDEO_CHECKPOINT = "checkpoints/sam2_hiera_tiny.pt"
VIDEO_MODEL_CFG = "configs/sam2/sam2_hiera_t.yaml"
VIDEO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CUDA performance knobs
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

_VIDEO_PREDICTOR = None
_VIDEO_PREDICTOR_COMPILED = False


def maybe_cleanup_cuda():
    if VIDEO_DEVICE == "cuda":
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def get_video_predictor():
    global _VIDEO_PREDICTOR
    global _VIDEO_PREDICTOR_COMPILED

    if _VIDEO_PREDICTOR is None:
        print("SAM video running on:", VIDEO_DEVICE)

        if VIDEO_DEVICE == "cuda":
            maybe_cleanup_cuda()

        predictor = build_sam2_video_predictor(
            VIDEO_MODEL_CFG,
            VIDEO_CHECKPOINT,
            device=VIDEO_DEVICE,
        )

        # Best effort compile for newer PyTorch. Safe fallback if it fails.
        if VIDEO_DEVICE == "cuda" and hasattr(torch, "compile"):
            try:
                predictor = torch.compile(predictor, mode="reduce-overhead", fullgraph=False)
                _VIDEO_PREDICTOR_COMPILED = True
                print("Predictor compiled with torch.compile")
            except Exception as e:
                print("torch.compile skipped:", str(e))
                _VIDEO_PREDICTOR_COMPILED = False

        _VIDEO_PREDICTOR = predictor

        # Warmup
        if VIDEO_DEVICE == "cuda":
            try:
                torch.cuda.synchronize()
                print("CUDA device:", torch.cuda.get_device_name(0))
            except Exception:
                pass

    return _VIDEO_PREDICTOR


def get_autocast_context():
    if VIDEO_DEVICE == "cuda":
        # float16 is often faster than bfloat16 on many consumer NVIDIA GPUs
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def ensure_clean_dir(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)


def mask_to_overlay(image_bgr, mask):
    overlay = image_bgr.copy()
    overlay[mask.astype(bool)] = [0, 255, 0]
    return overlay


def mask_area(mask):
    return int(mask.astype(bool).sum())


def mask_centroid(mask):
    ys, xs = mask.astype(bool).nonzero()
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.mean()), int(ys.mean())


def mask_bbox(mask):
    ys, xs = mask.astype(bool).nonzero()
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min = int(xs.min())
    y_min = int(ys.min())
    x_max = int(xs.max())
    y_max = int(ys.max())
    return x_min, y_min, x_max, y_max


def save_binary_mask_image(mask, output_path):
    if mask is None:
        return None

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    mask_image = (mask.astype(np.uint8) * 255)
    cv2.imwrite(output_path, mask_image)
    return os.path.abspath(output_path)


def write_temp_video_from_frames(frame_paths, output_video_path, fps):
    if len(frame_paths) == 0:
        return False

    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        return False

    height, width = first_frame.shape[:2]

    output_dir = os.path.dirname(output_video_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        writer.write(frame)

    writer.release()
    return True


def convert_video_to_web_mp4(input_video_path, output_video_path):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_exe,
        "-y",
        "-i", input_video_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        output_video_path,
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    return result.returncode == 0


def write_video_from_frames(frame_paths, output_video_path, fps):
    temp_output_video_path = output_video_path.replace(".mp4", "_temp.mp4")
    web_output_video_path = output_video_path.replace(".mp4", "_web.mp4")

    temp_written = write_temp_video_from_frames(
        frame_paths=frame_paths,
        output_video_path=temp_output_video_path,
        fps=fps,
    )

    if not temp_written:
        return None

    converted = convert_video_to_web_mp4(
        input_video_path=temp_output_video_path,
        output_video_path=web_output_video_path,
    )

    if converted:
        if os.path.exists(temp_output_video_path):
            os.remove(temp_output_video_path)
        return os.path.abspath(web_output_video_path)

    return os.path.abspath(temp_output_video_path)


def get_binary_mask_for_object(out_obj_ids, out_mask_logits, target_obj_id=1):
    if out_mask_logits is None:
        return None

    target_index = None
    for i, current_obj_id in enumerate(out_obj_ids):
        if int(current_obj_id) == int(target_obj_id):
            target_index = i
            break

    if target_index is None:
        if len(out_obj_ids) == 0:
            return None
        target_index = 0

    mask_tensor = out_mask_logits[target_index]

    if not torch.is_tensor(mask_tensor):
        mask_np = np.array(mask_tensor)
        mask_np = np.squeeze(mask_np)
        return (mask_np > 0.0).astype(np.uint8)

    # Keep tensor path efficient, single transfer to CPU only when needed
    mask_tensor = mask_tensor.detach()
    mask_tensor = torch.squeeze(mask_tensor)
    binary_mask = (mask_tensor > 0.0).to(torch.uint8)

    # We must eventually use NumPy for cv2/json side, but do it once only
    binary_mask_cpu = binary_mask.to("cpu", non_blocking=True).numpy()
    return binary_mask_cpu


def store_prompt_frame_mask(masks_by_local_frame, frame_idx, out_obj_ids, out_mask_logits, target_obj_id=1):
    binary_mask = get_binary_mask_for_object(
        out_obj_ids,
        out_mask_logits,
        target_obj_id=target_obj_id,
    )
    if binary_mask is None:
        return
    masks_by_local_frame[int(frame_idx)] = binary_mask


def collect_propagation_masks(predictor, inference_state, selected_local_frame_index, target_obj_id=1, reverse=False):
    collected = {}

    try:
        iterator = predictor.propagate_in_video(
            inference_state,
            start_frame_idx=selected_local_frame_index,
            reverse=reverse,
        )
    except TypeError:
        iterator = predictor.propagate_in_video(inference_state)

    for out_frame_idx, out_obj_ids, out_mask_logits in iterator:
        binary_mask = get_binary_mask_for_object(
            out_obj_ids,
            out_mask_logits,
            target_obj_id=target_obj_id,
        )
        if binary_mask is None:
            continue
        collected[int(out_frame_idx)] = binary_mask

    return collected


def prepare_single_video_for_tracking(
    video_path,
    video_role,
    run_root_folder,
    analysis_mode="full",
    max_seconds=15,
    frame_stride=1,
):
    video_info = get_video_info(video_path)
    if "error" in video_info:
        return {"error": video_info["error"]}

    analysis_window = get_analysis_window(
        video_path,
        analysis_mode=analysis_mode,
        max_seconds=max_seconds,
    )
    if "error" in analysis_window:
        return {"error": analysis_window["error"]}

    output_folder = os.path.join(run_root_folder, f"{video_role}_frames")

    extraction = extract_frame_range_to_folder(
        video_path=video_path,
        output_folder=output_folder,
        start_frame_index=analysis_window["start_frame_index"],
        end_frame_index=analysis_window["end_frame_index"],
        prefix="frame",
        frame_stride=frame_stride,
    )
    if "error" in extraction:
        return {"error": extraction["error"]}

    return {
        "video_role": video_role,
        "source_path": video_path,
        "frame_count": video_info["frame_count"],
        "fps": video_info["fps"],
        "width": video_info["width"],
        "height": video_info["height"],
        "duration_seconds": video_info["duration_seconds"],
        "analysis_start_frame_index": analysis_window["start_frame_index"],
        "analysis_end_frame_index": analysis_window["end_frame_index"],
        "analysis_frame_count": extraction["saved_frame_count"],
        "analysis_start_time_seconds": analysis_window["start_time_seconds"],
        "analysis_end_time_seconds": analysis_window["end_time_seconds"],
        "frames_folder": extraction["output_folder"],
        "saved_frame_count": extraction["saved_frame_count"],
        "saved_paths": extraction["saved_paths"],
        "saved_absolute_frame_indices": extraction["saved_absolute_frame_indices"],
        "frame_stride": extraction["frame_stride"],
    }


def sanitize_prompt_time(
    video_path,
    selected_time_seconds,
    analysis_mode="full",
    max_seconds=15,
    frame_stride=1,
):
    info = get_video_info(video_path)
    if "error" in info:
        return {"error": info["error"]}

    window = get_analysis_window(
        video_path,
        analysis_mode=analysis_mode,
        max_seconds=max_seconds,
    )
    if "error" in window:
        return {"error": window["error"]}

    fps = info["fps"]
    frame_count = info["frame_count"]

    if selected_time_seconds is None:
        selected_time_seconds = 0.0

    requested_time_seconds = float(selected_time_seconds)

    if requested_time_seconds < 0:
        requested_time_seconds = 0.0

    if requested_time_seconds > window["end_time_seconds"]:
        requested_time_seconds = window["end_time_seconds"]

    requested_absolute_frame_index = 0
    if fps > 0:
        requested_absolute_frame_index = int(requested_time_seconds * fps)

    if requested_absolute_frame_index >= frame_count:
        requested_absolute_frame_index = frame_count - 1

    if frame_stride is None:
        frame_stride = 1

    frame_stride = int(frame_stride)
    if frame_stride <= 0:
        return {"error": "frame_stride doit être >= 1"}

    sampled_absolute_frame_indices = list(
        range(
            window["start_frame_index"],
            window["end_frame_index"] + 1,
            frame_stride,
        )
    )

    if len(sampled_absolute_frame_indices) == 0:
        return {"error": "Aucune frame échantillonnée disponible"}

    snapped_absolute_frame_index = min(
        sampled_absolute_frame_indices,
        key=lambda x: abs(x - requested_absolute_frame_index),
    )

    local_frame_index = sampled_absolute_frame_indices.index(snapped_absolute_frame_index)

    snapped_time_seconds = requested_time_seconds
    if fps > 0:
        snapped_time_seconds = snapped_absolute_frame_index / fps

    return {
        "requested_time_seconds": requested_time_seconds,
        "requested_absolute_frame_index": requested_absolute_frame_index,
        "selected_time_seconds": snapped_time_seconds,
        "absolute_frame_index": snapped_absolute_frame_index,
        "local_frame_index": local_frame_index,
        "analysis_start_frame_index": window["start_frame_index"],
        "analysis_end_frame_index": window["end_frame_index"],
        "frame_stride": frame_stride,
    }


def run_tracking_on_prepared_video(
    prepared_video,
    selected_local_frame_index,
    point_xy,
    run_root_folder,
    target_obj_id=1,
):
    predictor = get_video_predictor()

    frames_folder = prepared_video["frames_folder"]
    saved_paths = prepared_video["saved_paths"]
    saved_absolute_frame_indices = prepared_video["saved_absolute_frame_indices"]
    fps = prepared_video["fps"]
    video_role = prepared_video["video_role"]

    overlay_folder = os.path.join(run_root_folder, f"{video_role}_overlay_frames")
    mask_folder = os.path.join(run_root_folder, f"{video_role}_mask_frames")
    ensure_clean_dir(overlay_folder)
    ensure_clean_dir(mask_folder)

    warnings = []
    masks_by_local_frame = {}

    # Keep prompt tensors on GPU when possible
    if VIDEO_DEVICE == "cuda":
        points = torch.tensor([[point_xy[0], point_xy[1]]], dtype=torch.float32, device=VIDEO_DEVICE)
        labels = torch.tensor([1], dtype=torch.int32, device=VIDEO_DEVICE)
    else:
        points = np.array([[point_xy[0], point_xy[1]]], dtype=np.float32)
        labels = np.array([1], np.int32)

    with torch.inference_mode():
        with get_autocast_context():
            inference_state = predictor.init_state(video_path=frames_folder)

            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=int(selected_local_frame_index),
                obj_id=int(target_obj_id),
                points=points,
                labels=labels,
            )

            store_prompt_frame_mask(
                masks_by_local_frame,
                selected_local_frame_index,
                out_obj_ids,
                out_mask_logits,
                target_obj_id=target_obj_id,
            )

            forward_masks = collect_propagation_masks(
                predictor,
                inference_state,
                selected_local_frame_index,
                target_obj_id=target_obj_id,
                reverse=False,
            )
            masks_by_local_frame.update(forward_masks)

    # Reverse pass
    try:
        with torch.inference_mode():
            with get_autocast_context():
                reverse_state = predictor.init_state(video_path=frames_folder)

                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=reverse_state,
                    frame_idx=int(selected_local_frame_index),
                    obj_id=int(target_obj_id),
                    points=points,
                    labels=labels,
                )

                store_prompt_frame_mask(
                    masks_by_local_frame,
                    selected_local_frame_index,
                    out_obj_ids,
                    out_mask_logits,
                    target_obj_id=target_obj_id,
                )

                backward_masks = collect_propagation_masks(
                    predictor,
                    reverse_state,
                    selected_local_frame_index,
                    target_obj_id=target_obj_id,
                    reverse=True,
                )

                for k, v in backward_masks.items():
                    if k not in masks_by_local_frame:
                        masks_by_local_frame[k] = v
    except Exception as e:
        warnings.append(f"reverse_tracking_failed: {str(e)}")

    if VIDEO_DEVICE == "cuda":
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

    overlay_paths = []
    frame_results = []

    # This phase is CPU-heavy by nature: cv2 read/draw/write
    for local_idx, frame_path in enumerate(saved_paths):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        absolute_frame_index = saved_absolute_frame_indices[local_idx]
        time_seconds = None
        if fps > 0:
            time_seconds = absolute_frame_index / fps

        if local_idx in masks_by_local_frame:
            mask = masks_by_local_frame[local_idx]
            object_present = bool(mask.astype(bool).any())
        else:
            mask = None
            object_present = False

        saved_mask_path = None

        if mask is not None and object_present:
            overlay = mask_to_overlay(frame, mask)
            area_value = mask_area(mask)
            centroid_value = mask_centroid(mask)
            bbox_value = mask_bbox(mask)

            mask_filename = f"frame_{absolute_frame_index:06d}_track_{target_obj_id:03d}.png"
            mask_path = os.path.join(mask_folder, mask_filename)
            saved_mask_path = save_binary_mask_image(mask, mask_path)
        else:
            overlay = frame
            area_value = None
            centroid_value = None
            bbox_value = None

        overlay_path = os.path.join(overlay_folder, f"overlay_{local_idx:06d}.jpg")
        cv2.imwrite(overlay_path, overlay)
        overlay_paths.append(overlay_path)

        frame_results.append(
            {
                "local_frame_index": local_idx,
                "absolute_frame_index": absolute_frame_index,
                "time_seconds": time_seconds,
                "object_present": object_present,
                "mask_area": area_value,
                "centroid": centroid_value,
                "bbox": bbox_value,
                "mask_path": saved_mask_path,
            }
        )

    output_video_path = os.path.join(run_root_folder, f"{video_role}_annotated.mp4")
    final_video_path = write_video_from_frames(overlay_paths, output_video_path, fps)

    if final_video_path is None:
        warnings.append("annotated_video_not_written")

    detected_frame_count = 0
    for item in frame_results:
        if item["object_present"]:
            detected_frame_count += 1

    if VIDEO_DEVICE == "cuda":
        maybe_cleanup_cuda()

    return {
        "video_role": video_role,
        "annotated_video_path": final_video_path,
        "overlay_folder": overlay_folder,
        "mask_folder": mask_folder,
        "overlay_frame_count": len(overlay_paths),
        "detected_frame_count": detected_frame_count,
        "frame_results": frame_results,
        "warnings": warnings,
    }


def save_json_file(data, output_json_path):
    output_dir = os.path.dirname(output_json_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return output_json_path


def build_raw_json_for_video(
    run_id,
    created_at,
    prepared_video,
    tracking_result,
    prompt_info,
    model_name="sam2_hiera_tiny",
):
    source_path = prepared_video["source_path"]
    filename = os.path.basename(source_path)
    fps = prepared_video["fps"]

    detected_absolute_frames = []
    for item in tracking_result["frame_results"]:
        if item["object_present"]:
            detected_absolute_frames.append(item["absolute_frame_index"])

    first_detected_frame = None
    last_detected_frame = None
    if len(detected_absolute_frames) > 0:
        first_detected_frame = min(detected_absolute_frames)
        last_detected_frame = max(detected_absolute_frames)

    frames = []

    for item in tracking_result["frame_results"]:
        objects = []

        if item["object_present"]:
            temporal_source = "propagated"
            if int(item["absolute_frame_index"]) == int(prompt_info["absolute_frame_index"]):
                temporal_source = "prompted"

            centroid_xy = None
            if item["centroid"] is not None:
                centroid_xy = [float(item["centroid"][0]), float(item["centroid"][1])]

            bbox_xyxy = None
            if item["bbox"] is not None:
                bbox_xyxy = [int(v) for v in item["bbox"]]

            objects.append(
                {
                    "track_id": 1,
                    "class_name": "prompted_object",
                    "class_confidence": None,
                    "bbox_xyxy": bbox_xyxy,
                    "bbox_confidence": None,
                    "mask_area_px": item["mask_area"],
                    "mask_bbox_xyxy": bbox_xyxy,
                    "mask_centroid_xy": centroid_xy,
                    "mask_iou_confidence": None,
                    "mask_path": item.get("mask_path"),
                    "temporal_source": temporal_source,
                    "is_new_track": bool(item["absolute_frame_index"] == first_detected_frame),
                    "is_lost_track": bool(item["absolute_frame_index"] == last_detected_frame),
                }
            )

        frames.append(
            {
                "frame_index": int(item["absolute_frame_index"]),
                "timestamp_sec": float(item["time_seconds"]) if item["time_seconds"] is not None else None,
                "objects_detected": len(objects),
                "objects": objects,
            }
        )

    raw_json = {
        "run": {
            "run_id": run_id,
            "pipeline_name": "sam2_video_segmentation",
            "model_name": model_name,
            "created_at": created_at,
            "input_video": filename,
            "fps_used_for_processing": fps,
            "total_frames": prepared_video["analysis_frame_count"],
            "raw_schema_version": "1.0",
        },
        "video_metadata": {
            "filename": filename,
            "path": source_path,
            "width": prepared_video["width"],
            "height": prepared_video["height"],
            "fps": fps,
            "duration_sec": prepared_video["duration_seconds"],
        },
        "frames": frames,
    }

    return raw_json


def export_raw_json_for_video(run_root_folder, video_role, raw_json):
    json_folder = os.path.join(run_root_folder, "json")
    output_json_path = os.path.join(json_folder, f"{video_role}_sam2_raw.json")
    save_json_file(raw_json, output_json_path)
    return output_json_path


def track_two_videos_from_selected_points(
    expert_video_path,
    learner_video_path,
    expert_time_seconds,
    learner_time_seconds,
    expert_point_xy,
    learner_point_xy,
    analysis_mode="full",
    max_seconds=15,
    frame_stride=1,
    output_root="outputs/tmp_tracking_runs",
):
    started_at = time.time()
    created_at = datetime.datetime.now().isoformat()

    run_id = uuid.uuid4().hex
    run_root_folder = os.path.join(output_root, run_id)
    os.makedirs(run_root_folder, exist_ok=True)

    if VIDEO_DEVICE == "cuda":
        maybe_cleanup_cuda()

    expert_prompt = sanitize_prompt_time(
        expert_video_path,
        expert_time_seconds,
        analysis_mode=analysis_mode,
        max_seconds=max_seconds,
        frame_stride=frame_stride,
    )
    if "error" in expert_prompt:
        return {"error": f"Expert: {expert_prompt['error']}"}

    learner_prompt = sanitize_prompt_time(
        learner_video_path,
        learner_time_seconds,
        analysis_mode=analysis_mode,
        max_seconds=max_seconds,
        frame_stride=frame_stride,
    )
    if "error" in learner_prompt:
        return {"error": f"Learner: {learner_prompt['error']}"}

    expert_prepared = prepare_single_video_for_tracking(
        video_path=expert_video_path,
        video_role="expert",
        run_root_folder=run_root_folder,
        analysis_mode=analysis_mode,
        max_seconds=max_seconds,
        frame_stride=frame_stride,
    )
    if "error" in expert_prepared:
        return {"error": f"Expert: {expert_prepared['error']}"}

    learner_prepared = prepare_single_video_for_tracking(
        video_path=learner_video_path,
        video_role="learner",
        run_root_folder=run_root_folder,
        analysis_mode=analysis_mode,
        max_seconds=max_seconds,
        frame_stride=frame_stride,
    )
    if "error" in learner_prepared:
        return {"error": f"Learner: {learner_prepared['error']}"}

    expert_tracking = run_tracking_on_prepared_video(
        prepared_video=expert_prepared,
        selected_local_frame_index=expert_prompt["local_frame_index"],
        point_xy=expert_point_xy,
        run_root_folder=run_root_folder,
        target_obj_id=1,
    )

    learner_tracking = run_tracking_on_prepared_video(
        prepared_video=learner_prepared,
        selected_local_frame_index=learner_prompt["local_frame_index"],
        point_xy=learner_point_xy,
        run_root_folder=run_root_folder,
        target_obj_id=1,
    )

    expert_raw_json = build_raw_json_for_video(
        run_id=run_id,
        created_at=created_at,
        prepared_video=expert_prepared,
        tracking_result=expert_tracking,
        prompt_info=expert_prompt,
        model_name="sam2_hiera_tiny",
    )

    learner_raw_json = build_raw_json_for_video(
        run_id=run_id,
        created_at=created_at,
        prepared_video=learner_prepared,
        tracking_result=learner_tracking,
        prompt_info=learner_prompt,
        model_name="sam2_hiera_tiny",
    )

    expert_raw_json_path = export_raw_json_for_video(
        run_root_folder=run_root_folder,
        video_role="expert",
        raw_json=expert_raw_json,
    )

    learner_raw_json_path = export_raw_json_for_video(
        run_root_folder=run_root_folder,
        video_role="learner",
        raw_json=learner_raw_json,
    )

    processing_time_sec = time.time() - started_at

    warnings = ["score_not_computed"]
    for warning in expert_tracking["warnings"]:
        warnings.append(f"expert_{warning}")
    for warning in learner_tracking["warnings"]:
        warnings.append(f"learner_{warning}")

    return {
        "run": {
            "run_id": run_id,
            "pipeline_name": "sam2_video_point_tracking",
            "processing_time_sec": processing_time_sec,
            "created_at": created_at,
            "component_notes": {
                "sam2_video": f"device={VIDEO_DEVICE}; analysis_mode={analysis_mode}; max_seconds={max_seconds}; frame_stride={frame_stride}; compiled={_VIDEO_PREDICTOR_COMPILED}",
                "prompting": "single_positive_point_per_video",
                "raw_json_export": "completed",
            }
        },
        "expert_video": {
            "path": expert_video_path,
            "annotated_video_path": expert_tracking["annotated_video_path"],
            "raw_json_path": expert_raw_json_path,
            "duration_sec": expert_prepared["duration_seconds"],
            "fps": expert_prepared["fps"],
            "width": expert_prepared["width"],
            "height": expert_prepared["height"],
            "analysis_start_frame_index": expert_prepared["analysis_start_frame_index"],
            "analysis_end_frame_index": expert_prepared["analysis_end_frame_index"],
            "analysis_frame_count": expert_prepared["analysis_frame_count"],
            "requested_time_seconds": expert_prompt["requested_time_seconds"],
            "selected_time_seconds": expert_prompt["selected_time_seconds"],
            "requested_absolute_frame_index": expert_prompt["requested_absolute_frame_index"],
            "selected_absolute_frame_index": expert_prompt["absolute_frame_index"],
            "selected_local_frame_index": expert_prompt["local_frame_index"],
            "frame_stride": frame_stride,
            "selected_point_xy": expert_point_xy,
            "detected_frame_count": expert_tracking["detected_frame_count"],
        },
        "learner_video": {
            "path": learner_video_path,
            "annotated_video_path": learner_tracking["annotated_video_path"],
            "raw_json_path": learner_raw_json_path,
            "duration_sec": learner_prepared["duration_seconds"],
            "fps": learner_prepared["fps"],
            "width": learner_prepared["width"],
            "height": learner_prepared["height"],
            "analysis_start_frame_index": learner_prepared["analysis_start_frame_index"],
            "analysis_end_frame_index": learner_prepared["analysis_end_frame_index"],
            "analysis_frame_count": learner_prepared["analysis_frame_count"],
            "requested_time_seconds": learner_prompt["requested_time_seconds"],
            "selected_time_seconds": learner_prompt["selected_time_seconds"],
            "requested_absolute_frame_index": learner_prompt["requested_absolute_frame_index"],
            "selected_absolute_frame_index": learner_prompt["absolute_frame_index"],
            "selected_local_frame_index": learner_prompt["local_frame_index"],
            "frame_stride": frame_stride,
            "selected_point_xy": learner_point_xy,
            "detected_frame_count": learner_tracking["detected_frame_count"],
        },
        "overall_score": None,
        "metrics": {
            "joint_angle_deviation": None,
            "trajectory_deviation": None,
            "velocity_difference": None,
            "tool_alignment_deviation": None,
            "dtw_cost": None,
            "semantic_similarity": None,
            "optical_flow_similarity": None,
            "extra": {
                "expert_detected_frame_count": expert_tracking["detected_frame_count"],
                "learner_detected_frame_count": learner_tracking["detected_frame_count"],
            }
        },
        "confidences": {
            "overall": None,
            "same_task": None,
            "score": None,
            "explanation": None
        },
        "explanation": {
            "text": "SAM 2 video tracking completed and raw JSON exported.",
            "strengths": [],
            "weaknesses": [],
            "raw_vlm_output": None,
            "structured_notes": {
                "expert_overlay_frame_count": expert_tracking["overlay_frame_count"],
                "learner_overlay_frame_count": learner_tracking["overlay_frame_count"],
                "expert_raw_json_path": expert_raw_json_path,
                "learner_raw_json_path": learner_raw_json_path,
            }
        },
        "warnings": warnings
    }