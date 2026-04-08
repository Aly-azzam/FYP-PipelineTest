# MediaPipe Wrist Detection — Implementation, Failure Analysis, and Research Context

This document explains how MediaPipe was implemented in the AugMentor 2.0 pipeline comparison demo, why it fails on our egocentric glove/sleeve craft videos, and what the real output data proves.

---

## 1. What we built

### Goal

Detect the **wrist position** of the expert and learner in egocentric craft videos, frame by frame, so downstream pipeline stages (DTW alignment, VLM explanation) can compare hand motion trajectories.

### Architecture

We built a reusable MediaPipe component at `backend/components/mediapipe/` with two main files:

| File | Role |
|------|------|
| `wrist_extractor.py` | Core per-frame extraction, smoothing, annotation, and video-level orchestration |
| `utils.py` | Helpers: BGR-to-RGB conversion, handedness normalization, jump rejection, gap interpolation |

The pipeline runner at `backend/pipelines/mediapipe_vlm/run.py` calls `extract_wrist_from_video()` for both expert and learner videos and maps the output into the app-wide `PipelineResult` schema.

### Detection strategy

We implemented a **Pose-first, Hands-fallback** approach:

1. **MediaPipe Pose** (`mp.solutions.pose`) runs on each frame and extracts:
   - `LEFT_WRIST` (landmark index 15)
   - `RIGHT_WRIST` (landmark index 16)
   - Filtered by a visibility threshold (default > 0.5)

2. **MediaPipe Hands** (`mp.solutions.hands`) runs on the same frame and extracts:
   - Wrist via `hand_landmarks.landmark[0]`
   - Left/right assignment via handedness classification

3. **Merge rule per side (left/right):**
   - If Pose wrist is valid and visible, use it (source = `"pose"`)
   - Else if Hands wrist is valid, use it (source = `"hands"`)
   - Else no detection (source = `"none"`)

### Post-processing pipeline

After per-frame extraction, the following stages run:

- **EMA smoothing** (alpha = 0.35) — reduces jitter when wrist is detected across consecutive frames
- **Impossible jump rejection** — discards wrist positions that jump more than 0.15 in normalized coordinates between frames (catches false positives)
- **Single-frame gap interpolation** — fills isolated missing frames by averaging the previous and next valid wrist position

### Configuration

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `min_detection_confidence` | 0.5 | Minimum confidence for MediaPipe to report a detection |
| `min_tracking_confidence` | 0.5 | Minimum confidence for frame-to-frame tracking |
| `model_complexity` | 1 | Model size (0 = lite, 1 = full, 2 = heavy) |
| `pose_visibility_threshold` | 0.5 | Minimum Pose landmark visibility to accept a wrist |

---

## 2. Why it fails on our videos

### Our video conditions

All test videos share these characteristics:

- **Egocentric camera angle** — mounted on the head or chest, looking down at the hands
- **Gloves** — the performer wears gloves throughout
- **Long sleeves** — forearms and wrists are partially or fully covered by fabric
- **Craft task** — fine manual work with tools and materials, frequent hand-object overlap

### Root cause analysis

#### MediaPipe Hands fails because it depends on skin appearance

MediaPipe Hands uses a **palm detection model** as its first stage. This model is trained to recognize the visual appearance of bare human palms and fingers. When gloves cover the hand:

- The palm detector cannot find the hand region of interest (ROI)
- Without a valid ROI, the hand landmark model never runs
- No landmarks means no wrist extraction from Hands at all

This is not a threshold tuning issue. The model's training data does not include gloved hands, so the feature representations it learned simply do not match what it sees.

#### MediaPipe Pose fails because it expects a visible body context

MediaPipe Pose is trained on images showing a full or partial human body. The wrist landmarks (15 and 16) are part of a 33-point body skeleton. The model infers wrist position from the visual context of the arm, elbow, shoulder, and torso.

In our egocentric videos:

- The camera sees only the hands, forearms, and work surface
- There is no visible torso, shoulders, or head for the model to anchor its skeleton
- Sleeves and gloves remove the skin-tone cues the model uses for arm tracking
- The Pose model either fails to detect a body entirely, or produces wrist landmarks with very low visibility scores that get filtered out

#### Combined effect

Even with our Pose-first + Hands-fallback strategy, both models fail on most frames because the fundamental visual cues they rely on (skin, body shape, palm texture) are absent.

### Research context

This failure mode is well-documented in the computer vision literature:

- MediaPipe Hands was designed for **bare-hand interaction** in consumer devices (phones, webcams). Its palm detector is trained on datasets like the CMU Panoptic hand dataset and internal Google data, all featuring bare skin. Gloved hands are out of distribution.

- MediaPipe Pose (BlazePose) is designed for **full-body or upper-body** pose estimation from a roughly frontal or side camera angle. Egocentric views where only the hands are visible provide insufficient body context for the skeleton model to anchor landmark positions.

- Zhang et al. (2020) note in the original MediaPipe Hands paper that detection quality degrades significantly under occlusion, unusual lighting, and non-standard hand appearances. Gloves represent an extreme case of non-standard appearance.

- Damen et al. (EPIC-Kitchens, 2018–2022) showed that egocentric hand detection is a known hard problem, with specialized models (e.g., hand-object detectors trained on egocentric data) significantly outperforming general-purpose hand detectors in this setting.

---

## 3. Real output evidence

The following JSON is the actual output from running our `mediapipe_vlm` pipeline on two real egocentric glove videos:

```json
{
  "run": {
    "run_id": "0c63952c436246a5bf5e5b8969064083",
    "pipeline_name": "mediapipe_vlm",
    "processing_time_sec": 163.56303010002011,
    "created_at": "2026-04-08T01:51:53.492550",
    "component_notes": {
      "mediapipe": "mode=wrist_only; expert_detected_wrist_frames=3; learner_detected_wrist_frames=85",
      "vlm": "implemented=False"
    }
  },
  "expert_video": {
    "filename": "ea75104bbf66492da1bc56af2f27f4b1_expert_annotated.mp4",
    "path": "http://127.0.0.1:8000/media/.tmp_outputs/ea75104bbf66492da1bc56af2f27f4b1_expert_annotated.mp4",
    "duration_sec": 25.366,
    "fps": 30.00078845698967,
    "width": 480,
    "height": 368
  },
  "learner_video": {
    "filename": "90625171eb61484abea75f0c4b2172ab_learner_annotated.mp4",
    "path": "http://127.0.0.1:8000/media/.tmp_outputs/90625171eb61484abea75f0c4b2172ab_learner_annotated.mp4",
    "duration_sec": 22.933,
    "fps": 30.000436052849604,
    "width": 480,
    "height": 368
  },
  "overall_score": null,
  "metrics": {
    "joint_angle_deviation": null,
    "trajectory_deviation": null,
    "velocity_difference": null,
    "tool_alignment_deviation": null,
    "dtw_cost": null,
    "semantic_similarity": null,
    "optical_flow_similarity": null,
    "extra": null
  },
  "confidences": {
    "overall": null,
    "same_task": null,
    "score": null,
    "explanation": null
  },
  "explanation": {
    "text": "MediaPipe wrist extraction completed. VLM is not implemented yet.",
    "strengths": [],
    "weaknesses": [],
    "raw_vlm_output": null,
    "structured_notes": {
      "expert_detected_wrist_frames": 3,
      "learner_detected_wrist_frames": 85
    }
  },
  "warnings": [
    "vlm_not_implemented",
    "score_not_computed",
    "wrist_only_extraction"
  ]
}
```

### Field-by-field analysis of why this output is bad

#### Detection rate

| Video | Duration | FPS | Total frames | Detected wrist frames | Detection rate |
|-------|----------|-----|--------------|-----------------------|----------------|
| Expert | 25.4 sec | 30 | ~761 | **3** | **0.4%** |
| Learner | 22.9 sec | 30 | ~688 | **85** | **12.4%** |

The expert video has a **0.4% detection rate**. This means MediaPipe found a wrist in only 3 out of ~761 frames. This is catastrophically low. No meaningful motion trajectory can be built from 3 scattered points.

The learner video at 12.4% is also far too low. A usable trajectory comparison typically needs at least 60-80% coverage with reasonable continuity.

The asymmetry between expert and learner detection rates also means any downstream comparison (DTW alignment, velocity analysis) would be comparing near-empty data against sparse data, producing meaningless results.

#### No score, no metrics, no confidences

Every metric field is `null`. Every confidence field is `null`. The overall score is `null`. This is correct behavior given the pipeline state, but it means the output provides zero quantitative comparison value. The pipeline ran for 163 seconds and produced no usable comparison data.

#### Processing time

The pipeline took **163.6 seconds** (nearly 3 minutes) to process two videos totaling ~48 seconds. This is approximately 3.4x realtime. Most of that time was spent running Pose + Hands inference on every frame, even though the vast majority of frames returned no detection.

#### Warnings

The three warnings confirm the output is incomplete:

- `vlm_not_implemented` — the VLM explanation stage is not wired yet
- `score_not_computed` — no comparison score was generated
- `wrist_only_extraction` — only wrist extraction was attempted, no other metrics

#### Empty explanation

The explanation contains no strengths, no weaknesses, and no VLM analysis. The text is a static placeholder. This output cannot be used for research documentation or pedagogical feedback.

---

## 4. Conclusion

MediaPipe is a well-engineered framework for consumer hand and body tracking under standard conditions. However, it is fundamentally unsuitable as the primary motion extraction method for our specific use case:

| Requirement | MediaPipe capability |
|-------------|---------------------|
| Gloved hands | Not supported (palm detector trained on bare skin) |
| Egocentric camera | Out of distribution for Pose (expects visible body) |
| Sleeve-covered arms | Removes skin-tone cues needed for wrist visibility |
| Robust wrist tracking | Fails below usable threshold on our data |

### What this means for the project

- MediaPipe-based pipelines (`mediapipe_vlm`, `mediapipe_dtw_vlm`, etc.) will produce unreliable or empty data on our actual test videos
- These pipelines should remain in the benchmark to document the failure, but should not be relied on as primary comparison methods
- Pipelines that do not depend on skin-based landmark detection (VLM-only, optical flow, V-JEPA, SAM 2) are more appropriate for our egocentric glove videos

### Keeping MediaPipe in the benchmark

Despite the failure, MediaPipe remains valuable in the project for three reasons:

1. It provides a concrete negative baseline that demonstrates why landmark-based approaches fail under these conditions
2. It validates the project's multi-pipeline comparison design: if all pipelines used MediaPipe internally, the entire system would fail silently
3. The detection rate numbers (0.4%, 12.4%) are useful quantitative evidence for the research report
