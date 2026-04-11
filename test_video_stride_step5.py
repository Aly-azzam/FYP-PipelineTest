from video_utils import extract_frame_range_to_folder

video_path = r"C:\Users\User\Desktop\vids sam\expert_30s.mp4"

result = extract_frame_range_to_folder(
    video_path=video_path,
    output_folder="outputs/test_stride_frames",
    start_frame_index=0,
    end_frame_index=24,
    frame_stride=3,
)

print(result)