import cv2


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"error": "Impossible d'ouvrir la vidéo"}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    duration = 0
    if fps > 0:
        duration = frame_count / fps

    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": duration,
    }


def compare_two_videos(expert_video_path, learner_video_path):
    expert_info = get_video_info(expert_video_path)
    learner_info = get_video_info(learner_video_path)

    expert_frame_info = get_first_frame_shape(expert_video_path)
    learner_frame_info = get_first_frame_shape(learner_video_path)

    expert_brightness_info = get_first_frame_mean_brightness(expert_video_path)
    learner_brightness_info = get_first_frame_mean_brightness(learner_video_path)

    print("Comparaison vidéo")
    print(f"Expert : {expert_info}")
    print(f"Learner : {learner_info}")

    return {
        "expert": expert_info,
        "learner": learner_info,
        "expert_first_frame": expert_frame_info,
        "learner_first_frame": learner_frame_info,
        "expert_brightness": expert_brightness_info,
        "learner_brightness": learner_brightness_info,
    }

def get_first_frame_shape(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"error": "Impossible d'ouvrir la vidéo"}

    success, frame = cap.read()
    cap.release()

    if not success:
        return {"error": "Impossible de lire la première frame"}

    height, width = frame.shape[:2]

    return {
        "first_frame_width": width,
        "first_frame_height": height
    }


def get_first_frame_mean_brightness(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"error": "Impossible d'ouvrir la vidéo"}

    success, frame = cap.read()
    cap.release()

    if not success:
        return {"error": "Impossible de lire la première frame"}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())

    return {
        "first_frame_mean_brightness": brightness
    }