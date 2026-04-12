import json
from pathlib import Path

from sam_runner import track_two_videos_from_selected_points
from sam_metrics import compute_metrics_for_two_videos


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


print("STEP 1: start")

# 1) Main SAM tracking result
result = track_two_videos_from_selected_points(
    expert_video_path="test_videos/expert.mp4",
    learner_video_path="test_videos/learner.mp4",
    expert_time_seconds=1.0,
    learner_time_seconds=1.0,
    expert_point_xy=[297, 139],
    learner_point_xy=[302, 158],
    analysis_mode="first_n_seconds",
    max_seconds=10,
    frame_stride=3,
    output_root="outputs/benchmark_tracking",
)

print("STEP 2: tracking done")

# 2) Load the two raw SAM JSON files and build the same RAW bundle your UI shows
expert_raw_json_path = result["expert_video"]["raw_json_path"]
learner_raw_json_path = result["learner_video"]["raw_json_path"]

raw_json_bundle = {
    "expert_raw_json_path": expert_raw_json_path,
    "learner_raw_json_path": learner_raw_json_path,
    "expert_raw_json": load_json(expert_raw_json_path),
    "learner_raw_json": load_json(learner_raw_json_path),
}

print("STEP 3: raw json bundle done")

# 3) Compute metrics from the two raw JSON files
metrics_result = compute_metrics_for_two_videos(
    expert_raw_json_path=expert_raw_json_path,
    learner_raw_json_path=learner_raw_json_path,
    output_root="outputs/benchmark_tracking",
)

print("STEP 4: metrics done")

# 4) Load the two metrics JSON files and build the same Derived Metrics bundle your UI shows
expert_metrics_json_path = metrics_result["expert_metrics_json_path"]
learner_metrics_json_path = metrics_result["learner_metrics_json_path"]

derived_metrics_bundle = {
    "expert_metrics_json_path": expert_metrics_json_path,
    "learner_metrics_json_path": learner_metrics_json_path,
    "expert_metrics_json": load_json(expert_metrics_json_path),
    "learner_metrics_json": load_json(learner_metrics_json_path),
}

print("STEP 5: derived metrics bundle done")

# 5) Save the same 3 outputs in one easy place for benchmark runs
run_id = result["run"]["run_id"]
run_folder = Path("outputs/benchmark_tracking") / run_id / "json"

main_result_path = run_folder / "main_result_bundle.json"
raw_bundle_path = run_folder / "raw_json_bundle.json"
derived_bundle_path = run_folder / "derived_metrics_bundle.json"

with open(main_result_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

with open(raw_bundle_path, "w", encoding="utf-8") as f:
    json.dump(raw_json_bundle, f, ensure_ascii=False, indent=2)

with open(derived_bundle_path, "w", encoding="utf-8") as f:
    json.dump(derived_metrics_bundle, f, ensure_ascii=False, indent=2)

print("STEP 6: all bundles saved")
print("MAIN RESULT:", main_result_path)
print("RAW JSON SAM 2:", raw_bundle_path)
print("DERIVED METRICS JSON:", derived_bundle_path)
print("DONE")