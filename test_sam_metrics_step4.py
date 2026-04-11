from sam_metrics import export_derived_metrics_json

expert_raw_json_path = r"outputs\tmp_tracking_runs\2026fc83a19d4c7499655134ff517e8f\json\expert_sam2_raw.json"
learner_raw_json_path = r"outputs\tmp_tracking_runs\2026fc83a19d4c7499655134ff517e8f\json\learner_sam2_raw.json"

expert_metrics_json_path = export_derived_metrics_json(expert_raw_json_path)
learner_metrics_json_path = export_derived_metrics_json(learner_raw_json_path)

print("EXPERT METRICS JSON PATH:")
print(expert_metrics_json_path)

print("LEARNER METRICS JSON PATH:")
print(learner_metrics_json_path)