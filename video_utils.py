import cv2
import os
import shutil


def get_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    return cap


def get_video_info(video_path):
    cap = get_video_capture(video_path)

    if cap is None:
        return {"error": "Impossible d'ouvrir la vidéo"}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    duration_seconds = 0.0
    if fps > 0:
        duration_seconds = frame_count / fps

    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
    }


def read_frame_at_index(video_path, frame_index):
    cap = get_video_capture(video_path)

    if cap is None:
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return None

    if frame_index < 0 or frame_index >= total_frames:
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = cap.read()
    cap.release()

    if not success:
        return None

    return frame


def read_frame_at_time(video_path, time_seconds):
    info = get_video_info(video_path)

    if "error" in info:
        return None

    fps = info["fps"]

    if fps <= 0:
        return None

    frame_index = int(time_seconds * fps)
    max_index = max(0, info["frame_count"] - 1)

    if frame_index > max_index:
        frame_index = max_index

    return read_frame_at_index(video_path, frame_index)


def extract_first_frame(video_path, output_image_path=None):
    frame = read_frame_at_index(video_path, 0)

    if frame is None:
        if output_image_path is not None:
            print("Erreur : impossible de lire la première frame.")
            return False
        return None

    if output_image_path is not None:
        output_dir = os.path.dirname(output_image_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        cv2.imwrite(output_image_path, frame)
        print(f"Première frame sauvegardée dans : {output_image_path}")
        return True

    return frame


def save_frame(frame, output_image_path):
    if frame is None:
        return False

    output_dir = os.path.dirname(output_image_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(output_image_path, frame)
    return True


def save_frame_at_index(video_path, frame_index, output_image_path):
    frame = read_frame_at_index(video_path, frame_index)
    return save_frame(frame, output_image_path)


def save_frame_at_time(video_path, time_seconds, output_image_path):
    frame = read_frame_at_time(video_path, time_seconds)
    return save_frame(frame, output_image_path)


def get_analysis_frame_limit(video_path, analysis_mode="full", max_seconds=15):
    info = get_video_info(video_path)

    if "error" in info:
        return {"error": info["error"]}

    frame_count = info["frame_count"]
    fps = info["fps"]

    if frame_count <= 0:
        return {"error": "Vidéo vide ou nombre de frames invalide"}

    if analysis_mode == "full":
        end_frame_index = frame_count - 1
    elif analysis_mode == "first_n_seconds":
        if fps <= 0:
            return {"error": "FPS invalide"}
        end_frame_index = int(max_seconds * fps) - 1
        end_frame_index = min(end_frame_index, frame_count - 1)
    else:
        return {"error": f"Mode d'analyse inconnu: {analysis_mode}"}

    return {
        "start_frame_index": 0,
        "end_frame_index": end_frame_index,
        "frame_count_to_process": end_frame_index + 1,
    }


def get_analysis_window(video_path, analysis_mode="full", max_seconds=15):
    info = get_video_info(video_path)

    if "error" in info:
        return {"error": info["error"]}

    limits = get_analysis_frame_limit(
        video_path,
        analysis_mode=analysis_mode,
        max_seconds=max_seconds,
    )

    if "error" in limits:
        return {"error": limits["error"]}

    fps = info["fps"]
    start_frame_index = limits["start_frame_index"]
    end_frame_index = limits["end_frame_index"]

    start_time_seconds = 0.0
    end_time_seconds = 0.0

    if fps > 0:
        start_time_seconds = start_frame_index / fps
        end_time_seconds = end_frame_index / fps

    return {
        "start_frame_index": start_frame_index,
        "end_frame_index": end_frame_index,
        "frame_count_to_process": limits["frame_count_to_process"],
        "start_time_seconds": start_time_seconds,
        "end_time_seconds": end_time_seconds,
    }


def ensure_clean_dir(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    os.makedirs(folder_path, exist_ok=True)


def extract_frame_range_to_folder(
    video_path,
    output_folder,
    start_frame_index,
    end_frame_index,
    prefix="frame",
    frame_stride=1,
):
    cap = get_video_capture(video_path)

    if cap is None:
        return {"error": "Impossible d'ouvrir la vidéo"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return {"error": "Vidéo vide ou nombre de frames invalide"}

    if start_frame_index < 0:
        start_frame_index = 0

    if end_frame_index >= total_frames:
        end_frame_index = total_frames - 1

    if start_frame_index > end_frame_index:
        cap.release()
        return {"error": "Intervalle de frames invalide"}

    if frame_stride is None:
        frame_stride = 1

    frame_stride = int(frame_stride)

    if frame_stride <= 0:
        cap.release()
        return {"error": "frame_stride doit être >= 1"}

    ensure_clean_dir(output_folder)

    saved_paths = []
    saved_absolute_frame_indices = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
    current_frame_index = start_frame_index

    while current_frame_index <= end_frame_index:
        success, frame = cap.read()

        if not success:
            break

        if (current_frame_index - start_frame_index) % frame_stride == 0:
            frame_name = f"{current_frame_index:06d}.jpg"
            frame_path = os.path.join(output_folder, frame_name)

            cv2.imwrite(frame_path, frame)
            saved_paths.append(frame_path)
            saved_absolute_frame_indices.append(current_frame_index)

        current_frame_index += 1

    cap.release()

    return {
        "output_folder": output_folder,
        "saved_frame_count": len(saved_paths),
        "saved_paths": saved_paths,
        "saved_absolute_frame_indices": saved_absolute_frame_indices,
        "start_frame_index": saved_absolute_frame_indices[0] if len(saved_absolute_frame_indices) > 0 else None,
        "end_frame_index": saved_absolute_frame_indices[-1] if len(saved_absolute_frame_indices) > 0 else None,
        "frame_stride": frame_stride,
    }
