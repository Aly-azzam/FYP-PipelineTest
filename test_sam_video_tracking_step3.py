from sam_runner import track_two_videos_from_selected_points

expert_video_path = r"C:\Users\User\Desktop\vids sam\expert_30s.mp4"
learner_video_path = r"C:\Users\User\Desktop\vids sam\learner_26s.mp4"

expert_time_seconds = 1.00
learner_time_seconds = 1.00

expert_point_xy = (715, 724)
learner_point_xy = (1283, 662)

result = track_two_videos_from_selected_points(
    expert_video_path=expert_video_path,
    learner_video_path=learner_video_path,
    expert_time_seconds=expert_time_seconds,
    learner_time_seconds=learner_time_seconds,
    expert_point_xy=expert_point_xy,
    learner_point_xy=learner_point_xy,
    analysis_mode="first_n_seconds",
    max_seconds=10,
    frame_stride=3,
)

print("RUN NOTES:")
print(result["run"]["component_notes"])

print("EXPERT:")
print(result["expert_video"])

print("LEARNER:")
print(result["learner_video"])