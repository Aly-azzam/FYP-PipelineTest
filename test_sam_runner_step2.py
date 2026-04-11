from sam_runner import prepare_two_videos_for_tracking

expert_video_path = r"C:\Users\User\Desktop\vids sam\expert_30s.mp4"
learner_video_path = r"C:\Users\User\Desktop\vids sam\learner_26s.mp4"

result = prepare_two_videos_for_tracking(
    expert_video_path=expert_video_path,
    learner_video_path=learner_video_path,
    analysis_mode="first_n_seconds",
    max_seconds=15,
)

print(result)