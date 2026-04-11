from video_utils import (
    get_video_info,
    extract_first_frame,
    read_frame_at_index,
    read_frame_at_time,
    save_frame_at_index,
    get_analysis_frame_limit,
)

video_path = r"C:\Users\User\Desktop\vids sam\expert_30s.mp4"

print("VIDEO INFO:")
print(get_video_info(video_path))

first_frame = extract_first_frame(video_path)
print("FIRST FRAME IS NONE:", first_frame is None)

frame_10 = read_frame_at_index(video_path, 10)
print("FRAME 10 IS NONE:", frame_10 is None)

frame_at_2s = read_frame_at_time(video_path, 2)
print("FRAME AT 2s IS NONE:", frame_at_2s is None)

saved = save_frame_at_index(video_path, 10, "outputs/frame_10.jpg")
print("FRAME 10 SAVED:", saved)

print("FULL MODE:")
print(get_analysis_frame_limit(video_path, analysis_mode="full"))

print("FIRST 15 SECONDS MODE:")
print(get_analysis_frame_limit(video_path, analysis_mode="first_n_seconds", max_seconds=15))