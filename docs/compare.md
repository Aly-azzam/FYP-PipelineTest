Below is a **full documentation note** you can save and reuse for your pipeline comparison work in AugMentor 2.0.

# AugMentor 2.0 — Pipeline Comparison Documentation

## 1. Purpose of this comparison

The purpose of this comparison is to evaluate multiple AI pipelines for the **same project task**:

* one **expert craft video**
* one **learner imitation video**
* the system compares them
* the system computes structured motion differences
* the system produces a score
* the system gives a clear explanation of the differences

This comparison is not meant to test random AI models in general.
It is meant to determine which pipeline is the most suitable for the **expert-vs-learner craft evaluation problem** in AugMentor 2.0.

The final result of this comparison should be:

* a **scientific comparison**
* a **set of tables**
* a **set of graphs**
* evidence for choosing the best pipeline for the final project

---

# 2. Important project principle

The project files make one thing very clear:

* **deep learning is used for perception**
* **deterministic logic is used for evaluation and scoring**
* **VLM is used for grounded explanation**
* **the VLM should not be the one deciding the score by itself**

So in the comparison, the best pipeline is not just the one that “sounds smart,” but the one that best supports:

* structured extraction
* reliable comparison
* deterministic metrics
* accurate scoring
* grounded explanation

---

# 3. What exactly is being compared

Each compared pipeline is a different way of solving the same task:

**Expert video + Learner video → structured comparison → score → explanation**

Examples of pipelines:

* VLM
* MediaPipe + VLM
* MediaPipe + YOLO + VLM
* MediaPipe + SAM + V-JEPA + VLM
* MediaPipe + SAM + V-JEPA + DTW + VLM
* other variants such as optical flow, pose, grounding, etc.

The comparison must always be fair:

* same expert video
* same learner videos
* same benchmark set
* same output format
* same evaluation method

---

# 4. What should be fixed before starting

Before running experiments, the following must be fixed and standardized.

## 4.1 Fixed benchmark set

Use one benchmark set for all pipelines.

It should contain:

* **1 expert video**
* several learner videos in the **same context**
* different quality levels of imitation

Recommended:

* 1 expert video
* 5 to 10 learner videos

Example:

* Learner 1 = very close to expert
* Learner 2 = good imitation
* Learner 3 = medium imitation
* Learner 4 = weak imitation
* Learner 5 = poor imitation
* Learner 6 = medium but slower
* Learner 7 = medium but with camera variation

This gives enough variation for meaningful comparison.

## 4.2 Fixed conditions

Keep as many things constant as possible:

* same FPS if possible
* same resolution if possible
* same duration range
* same chapter/task
* same scoring range
* same output JSON schema
* same evaluation scripts

## 4.3 Fixed human reference

You need a human reference to know which pipeline is more accurate.

For each learner video, gather:

* human score out of 100
* human ranking
* optional explanation rating

Best:

* 3 to 5 human evaluators
* compute average score

This becomes your **ground truth reference**.

---

# 5. Main experiment goal

Since you said the tested videos are already **in-context**, then the comparison should focus on:

* similarity evaluation quality
* score accuracy
* explanation quality
* stability
* speed
* practicality

So for this experiment, the domain gate is not the main focus.

---

# 6. What every pipeline must output

Every pipeline must export a **standardized JSON result**.

This is extremely important.

Without a common format, comparison becomes messy and unfair.

## 6.1 Required JSON fields

Each pipeline should save, for every expert-learner pair:

```json
{
  "pipeline_name": "mediapipe_sam_vjepa_dtw_vlm",
  "expert_video_id": "expert_01",
  "learner_video_id": "learner_03",
  "run_id": "run_001",
  "overall_score": 78.4,
  "processing_time_sec": 19.2,
  "metrics": {
    "joint_angle_deviation": 11.2,
    "trajectory_deviation": 0.14,
    "velocity_difference": 0.09,
    "tool_alignment_deviation": 0.05,
    "dtw_cost": 0.21,
    "semantic_similarity": 0.84
  },
  "confidence": 0.82,
  "flags": {
    "low_landmark_confidence": false,
    "missing_tool_frames": false,
    "alignment_warning": false
  },
  "explanation": "Your hand path is close to the expert, but your wrist angle changes too much during the shaping phase."
}
```

## 6.2 Important rule

Even if a pipeline does not use one metric, keep the field.

Example:

* no DTW → `"dtw_cost": null`
* no tool branch → `"tool_alignment_deviation": null`
* no semantic branch → `"semantic_similarity": null`

This makes all pipelines comparable.

---

# 7. Human reference file

You also need a separate reference file.

Example:

```json
{
  "expert_video_id": "expert_01",
  "learner_video_id": "learner_03",
  "human_score": 82,
  "human_rank": 2,
  "explanation_quality_rating": 4.5
}
```

Or in CSV:

| expert_video_id | learner_video_id | human_score | human_rank | explanation_quality_rating |
| --------------- | ---------------- | ----------: | ---------: | -------------------------: |
| expert_01       | learner_01       |          91 |          1 |                        4.7 |
| expert_01       | learner_02       |          80 |          2 |                        4.2 |
| expert_01       | learner_03       |          62 |          4 |                        3.8 |

This file is essential because it lets you compare pipeline output to human judgment.

---

# 8. Best criteria to compare

According to the project files and architecture, these are the best criteria.

## 8.1 Score accuracy

Main question:

**How close is the pipeline score to the human score?**

This is the most important criterion.

Because the project’s core is expert-relative evaluation through deterministic metrics.

### Best metric:

**MAE — Mean Absolute Error**

Formula:

```text
MAE = average of |pipeline_score - human_score|
```

Interpretation:

* lower MAE = better
* lower means closer to human judgment

---

## 8.2 Ranking accuracy

Main question:

**Did the pipeline rank the learner videos in nearly the same order as humans?**

Even if scores are not perfect, good ranking is very valuable.

### Best metric:

* Spearman rank correlation
  or
* simple rank agreement percentage

Interpretation:

* higher = better
* closer to 1 = stronger ranking agreement

---

## 8.3 Internal metric usefulness

Main question:

**Did the pipeline produce meaningful and stable internal metrics?**

Important internal metrics from the project:

* joint angle deviation
* trajectory deviation
* velocity difference
* tool alignment deviation

These are central to your mathematical model.

You should compare whether the pipeline gives:

* interpretable values
* stable values
* useful values for explanation

This is especially important because your system is not just a black-box scorer.

---

## 8.4 Explanation quality

Main question:

**Is the explanation clear, correct, specific, and useful?**

The explanation should:

* mention real differences
* be understandable
* be specific, not vague
* help the learner improve
* remain grounded in metrics

### Best method:

human rating from 1 to 5 on:

* clarity
* correctness
* specificity
* usefulness

Then average them.

---

## 8.5 Reproducibility / stability

Main question:

**If the same pair is run multiple times, does the pipeline return almost the same result?**

Important because the project emphasizes deterministic evaluation.

### Measure:

run each pipeline multiple times on the same pair

Track:

* score variation
* metric variation

A simple form:

```text
variation % = |run1 - run2| / average * 100
```

Interpretation:

* lower variation = better
* ideally within very small range

---

## 8.6 Processing time

Main question:

**How long does the pipeline take per evaluation?**

This matters because a strong pipeline that is too slow may not be practical.

Track:

* total processing time per pair
* average time across all videos

Interpretation:

* lower = faster
* but must be interpreted together with accuracy

---

## 8.7 Robustness

Main question:

**Does the pipeline still perform well when conditions are slightly imperfect?**

Examples:

* slightly slower learner
* small camera shift
* small lighting change
* partial hand occlusion
* tool partially hidden

This criterion is very useful, especially for report discussion.

---

## 8.8 Implementation complexity

Main question:

**How difficult is the pipeline to integrate, maintain, and debug?**

Track:

* number of models
* number of dependencies
* implementation complexity
* runtime complexity
* debugging difficulty

This does not determine accuracy, but it helps justify the final choice.

---

# 9. What data you need to save

For every pipeline and every learner video, save the following:

* pipeline name
* learner video id
* expert video id
* overall score
* processing time
* explanation
* all internal metrics
* optional confidence
* optional warnings

For the human reference, save:

* learner video id
* human score
* human rank
* explanation quality score

Then merge everything into one master table.

---

# 10. Master comparison table

Create one final spreadsheet or CSV with columns like these:

| Pipeline | Expert Video | Learner Video | Pipeline Score | Human Score | Absolute Error | Human Rank | Pipeline Rank | Rank Match | Processing Time | Joint Angle Dev | Trajectory Dev | Velocity Diff | Tool Dev | Explanation Rating |
| -------- | ------------ | ------------- | -------------: | ----------: | -------------: | ---------: | ------------: | ---------- | --------------: | --------------: | -------------: | ------------: | -------: | -----------------: |

This becomes the main analysis table.

---

# 11. Best tables to include in the report

## Table 1 — Pipeline description table

Use this table to define each pipeline.

| Pipeline Name                        | Components                                         | Purpose                             | Expected Strength            |
| ------------------------------------ | -------------------------------------------------- | ----------------------------------- | ---------------------------- |
| VLM                                  | VLM only                                           | semantic comparison and explanation | simple baseline              |
| MediaPipe + VLM                      | hand landmarks + VLM                               | motion comparison + explanation     | stronger kinematics          |
| MediaPipe + YOLO + VLM               | hand + tool + explanation                          | tool-aware comparison               | better craft context         |
| MediaPipe + SAM + V-JEPA + DTW + VLM | segmentation + semantics + alignment + explanation | full multimodal comparison          | strongest research candidate |

This table introduces the experiment.

---

## Table 2 — Per-video results table

Shows raw results for each learner video.

| Pipeline | Learner Video | Pipeline Score | Human Score | Absolute Error | Time |
| -------- | ------------- | -------------: | ----------: | -------------: | ---: |

This is the raw evidence table.

---

## Table 3 — Final summary table

This is your most important summary table.

| Pipeline | MAE ↓ | Rank Correlation ↑ | Explanation Quality ↑ | Avg Time ↓ | Stability ↑ | Complexity | Final Comment |
| -------- | ----: | -----------------: | --------------------: | ---------: | ----------: | ---------- | ------------- |

Interpretation:

* ↓ means lower is better
* ↑ means higher is better

This table is likely the strongest one for the report conclusion.

---

## Table 4 — Internal metric behavior table

Useful if you want deeper technical comparison.

| Pipeline | Angle Metric Quality | Trajectory Metric Quality | Velocity Metric Quality | Tool Metric Quality | Notes |
| -------- | -------------------- | ------------------------- | ----------------------- | ------------------- | ----- |

This helps show why one pipeline scored better.

---

# 12. Best graphs to build

Graphs are very important because they make the final decision much more convincing.

## Graph 1 — MAE bar chart

### Purpose

Show which pipeline is closest to human scoring.

### Axes

* X-axis: pipeline
* Y-axis: MAE

### Interpretation

* lower bar = better accuracy

This should be your **main graph**.

---

## Graph 2 — Ranking agreement bar chart

### Purpose

Show which pipeline best preserves the human ranking order.

### Axes

* X-axis: pipeline
* Y-axis: Spearman correlation or rank agreement

### Interpretation

* higher = better

This is one of the strongest scientific graphs.

---

## Graph 3 — Explanation quality bar chart

### Purpose

Show which pipeline gives the best explanations.

### Axes

* X-axis: pipeline
* Y-axis: average explanation rating

### Interpretation

* higher = better

Very important because your project strongly emphasizes grounded explanation.

---

## Graph 4 — Processing time bar chart

### Purpose

Show runtime cost.

### Axes

* X-axis: pipeline
* Y-axis: average processing time in seconds

### Interpretation

* lower = faster

This graph helps explain tradeoffs.

---

## Graph 5 — Accuracy vs speed scatter plot

### Purpose

Show tradeoff between performance and practicality.

### Axes

* X-axis: average processing time
* Y-axis: accuracy measure, such as inverse MAE or normalized score quality

### Interpretation

* top-left or upper efficient region = strong accuracy with lower time

This is a very strong graph for decision making.

---

## Graph 6 — Per-video line chart

### Purpose

Show whether the pipeline follows human judgment across videos.

### Axes

* X-axis: learner videos
* Y-axis: score

Plot:

* human score line
* pipeline A line
* pipeline B line
* pipeline C line

### Interpretation

* the closer a pipeline follows the human curve, the better

This graph is excellent for showing detailed behavior.

---

## Graph 7 — Stability graph

### Purpose

Show score variation across repeated runs.

Possible forms:

* bar chart of score variance
* error bar chart
* boxplot by pipeline

### Interpretation

* smaller spread = more stable

Useful because determinism matters in your project.

---

## Graph 8 — Radar chart

### Purpose

Give one overall visual summary.

Possible axes:

* accuracy
* ranking
* explanation quality
* speed
* stability
* robustness

### Interpretation

* larger balanced area = stronger overall profile

This graph is attractive, but it should not replace the more scientific bar and line plots.

---

# 13. Best minimal graph set

If you want only the most useful graphs, build these 5:

1. **MAE bar chart**
2. **Ranking agreement bar chart**
3. **Explanation quality bar chart**
4. **Processing time bar chart**
5. **Per-video line chart against human scores**

That is already strong enough for a serious report.

---

# 14. How to calculate each main metric

## 14.1 Absolute error

For each learner video:

```text
absolute_error = |pipeline_score - human_score|
```

## 14.2 MAE

For each pipeline:

```text
MAE = average of all absolute errors
```

## 14.3 Ranking agreement

Compare the ranking of learner videos by:

* human scores
* pipeline scores

Use:

* Spearman correlation
  or
* simple ordered match comparison

## 14.4 Explanation quality

Humans rate each explanation from 1 to 5:

* clarity
* correctness
* specificity
* usefulness

Then average them.

## 14.5 Stability

Run the same pair multiple times and compute:

* score standard deviation
* or percentage variation

## 14.6 Average time

For each pipeline:

```text
average_time = sum(processing_times) / number_of_runs
```

---

# 15. How the workflow should go

## Step 1

Prepare the expert video and learner benchmark set.

## Step 2

Prepare the human reference scores and rankings.

## Step 3

Run Pipeline 1 on all pairs.

## Step 4

Save standardized JSON outputs.

## Step 5

Run Pipeline 2 on all pairs.

## Step 6

Repeat for all pipelines.

## Step 7

Merge all JSON outputs into one analysis table.

## Step 8

Compute:

* MAE
* ranking agreement
* explanation rating average
* average time
* stability

## Step 9

Create tables.

## Step 10

Create graphs.

## Step 11

Interpret results and justify final pipeline choice.

---

# 16. Important comparison rules

To keep the experiment valid:

* use the same benchmark videos for all pipelines
* use the same human reference for all pipelines
* use the same score range
* use the same JSON schema
* do not change evaluation criteria mid-experiment
* do not compare pipelines using different test sets
* do not judge only by explanation quality
* do not judge only by speed
* always balance accuracy and practicality

---

# 17. What to discuss in the final report

Your discussion should answer:

* Which pipeline had the lowest MAE?
* Which pipeline best matched human ranking?
* Which pipeline gave the most useful explanations?
* Which pipeline was fastest?
* Which pipeline was most stable?
* Which pipeline gave the best balance between accuracy and complexity?

Then explain why that pipeline is the best fit for AugMentor 2.0.

---

# 18. Recommended final decision logic

A pipeline should not be chosen only because it is the most complex.

A good final choice should ideally have:

* strong score accuracy
* strong ranking agreement
* grounded explanation
* acceptable time
* stable repeated outputs
* reasonable complexity

So the best final pipeline is the one with the **best balance**, not necessarily the biggest stack.

---

# 19. Suggested folder structure for the experiment

```text
pipeline_benchmark/
│
├── data/
│   ├── expert/
│   ├── learners/
│   └── human_reference.csv
│
├── outputs/
│   ├── vlm/
│   ├── mediapipe_vlm/
│   ├── mediapipe_yolo_vlm/
│   ├── mediapipe_sam_vjepa_vlm/
│   └── mediapipe_sam_vjepa_dtw_vlm/
│
├── merged_results/
│   ├── all_results.csv
│   ├── summary_metrics.csv
│
├── graphs/
│   ├── mae_bar_chart.png
│   ├── ranking_bar_chart.png
│   ├── explanation_bar_chart.png
│   ├── time_bar_chart.png
│   └── per_video_line_chart.png
│
└── report_notes/
    └── interpretation.md
```

---

# 20. Final short checklist

Before starting:

* benchmark set ready
* human scores ready
* JSON schema fixed
* pipelines defined

During experiment:

* same videos for all pipelines
* save every output
* track time
* keep same format

After experiment:

* merge results
* compute MAE
* compute ranking agreement
* rate explanations
* build tables
* build graphs
* interpret and justify final choice

---

# 21. Final conclusion

For this comparison, the most important thing is to treat it as a **structured evaluation experiment**, not just a coding test.

You need:

* standardized JSON outputs
* human reference scores
* clear criteria
* strong tables
* strong graphs

The most important outputs are:

* **MAE table and graph**
* **ranking agreement table and graph**
* **explanation quality table and graph**
* **processing time table and graph**
* **per-video comparison line chart**

These will give you strong proof for selecting the final pipeline.

If you want, I can now convert this into a **clean report-style version with headings and formal wording**, or into a **copy-paste DOCX-ready structure**.



## Final ordered pipelines

1. VLM only
2. Optical Flow + VLM
3. Optical Flow + DTW + VLM
4. V-JEPA + VLM
5. V-JEPA + DTW + VLM
6. SAM 2 + DTW + VLM
7. V-JEPA + SAM 2 + DTW + VLM
8. V-JEPA + Grounded SAM 2 + DTW + VLM
9. Optical Flow + V-JEPA + DTW + VLM

Documented failure baseline: MediaPipe + VLM (see `docs/MediaPipeFail.md`)
---

## Model / component table

| Model / Component | What it does | What it gives to your pipeline | Why it is useful for your project | Main limitation |
| --- | --- | --- | --- | --- |
| **VLM** | Interprets visual content and generates natural-language explanations | Textual feedback, difference description, pedagogical explanation | Good for explaining expert vs learner differences in clear language | Should not be trusted alone for deterministic scoring |
| **MediaPipe** | Detects hand landmarks frame by frame in video | 2D/3D hand keypoints, hand positions, motion features | Core for hand trajectory, joint angle, velocity, and other fine-motor metrics in egocentric craft videos | Can become unstable with occlusion, blur, unusual camera views, or hard hand-tool overlap |
| **DTW** | Aligns two motion sequences that happen at different speeds | Temporal correspondence between expert and learner frames | Very useful because learners will not perform the craft action at exactly the same speed | Adds computation and can still align badly if features are poor |
| **V-JEPA / V-JEPA 2** | Learns high-level video representations and temporal semantics | Semantic similarity, action understanding, video-level context | Useful for understanding whether the learner is doing a similar motion pattern/task, beyond raw geometry; it has shown strong egocentric-video relevance on EPIC-Kitchens-style tasks | Less directly interpretable than explicit geometric metrics |
| **SAM 2** | Segments objects/regions in images and videos and tracks them over time | Masks for tools, materials, active regions, hand-object zones | Useful when you need video-aware segmentation of tool/material interaction, not just hand landmarks; SAM 2 is designed for promptable segmentation in images and videos with streaming memory | Segmentation alone does not tell you if the action quality is good |
| **Grounded SAM 2** | Grounds objects from text prompts and tracks/segments them in video | Text-guided object localization + segmentation + tracking | Strong candidate for egocentric craft videos if you need to find and track a tool/material more reliably than plain detection; combines grounding with SAM 2 video tracking | Heavier and harder to integrate than plain SAM 2 |
| **YOLO** | Detects objects with bounding boxes | Tool/object detections | Can help if visible tools strongly matter in the craft task | Often weaker for small, domain-specific craft tools; gives boxes, not detailed masks |
| **Optical Flow** | Measures pixel-level motion between frames | Dense motion field | Useful as a backup motion signal when landmarks fail or become noisy | Sensitive to noise and less interpretable than explicit hand landmarks |



## VLM

we will compare openai and gemini 3 pro and see who is better
