from backend.components.mediapipe.wrist_extractor import extract_wrist_from_video


VIDEO_PATH = r"C:\Users\user\Downloads\VIDEO-2026-04-06-19-01-56.mp4"


def main():
    result = extract_wrist_from_video(VIDEO_PATH)

    print("Filename:", result["filename"])
    print("FPS:", result["fps"])
    print("Frame count:", result["frame_count"])
    print("Duration:", result["duration_sec"])

    detected_count = 0

    for frame_data in result["frames"]:
        if frame_data["left_wrist_visible"] or frame_data["right_wrist_visible"]:
            detected_count += 1

    print("\nFrames with at least one detected wrist:", detected_count)

    print("\nFirst 10 frames with a detected wrist:")
    shown = 0
    for frame_data in result["frames"]:
        if frame_data["left_wrist_visible"] or frame_data["right_wrist_visible"]:
            print(frame_data)
            shown += 1
            if shown == 10:
                break


if __name__ == "__main__":
    main()