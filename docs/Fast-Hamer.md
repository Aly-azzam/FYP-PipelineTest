# Technical Monograph: Fast-HaMeR Architecture and Methodology

| Field | Value |
|-------|--------|
| **Subject** | Real-time Monocular 3D Hand Mesh Recovery (HMR) |
| **Author** | Ali Azzam |
| **Date** | April 11, 2026 |

---

## 1. Executive Summary

Fast-HaMeR represents a paradigm shift in 3D hand pose estimation by transitioning from traditional Convolutional Neural Networks (CNNs) to high-capacity Vision Transformers (ViT). While its predecessor, HaMeR, established state-of-the-art accuracy through massive data scaling, the Fast-HaMeR fork optimizes this for real-time inference on consumer GPUs (e.g., NVIDIA RTX 30-series) by employing **Knowledge Distillation (KD)** and **Backbone Pruning**.

---

## 2. Theoretical Foundation: The MANO Model

To understand HaMeR, one must understand the **MANO** (Mesh-based Anthropometric hand mOdel).

The model defines the hand through a function \(M(\beta, \theta)\), where:

- **Shape** (\(\beta \in \mathbb{R}^{10}\)): Captures individual variation (finger length, palm width) using Principal Component Analysis (PCA) on 3D scans.
- **Pose** (\(\theta \in \mathbb{R}^{15 \times 3}\)): Captures the relative rotation of 15 skeletal joints in Axis-Angle representation.

Fast-HaMeR does not only predict points in 2D space; it regresses these specific parameters to deform a template mesh of **778 vertices** and **1538 faces**.

---

## 3. Architecture Deep-Dive

The “Fast” optimization utilizes a **Teacher–Student** framework.

### 3.1 The Transformer Backbone

Traditional CNNs struggle with “global context”—they might see a thumb and a pinky but fail to understand how they relate if a coffee mug is blocking the palm.

- **Self-Attention Mechanism:** Fast-HaMeR uses a Vision Transformer that treats image patches as tokens. This allows every part of the hand image to “attend” to every other part simultaneously.
- **Optimization:** In the “Fast” variant, the original ViT-Huge (632M parameters) is often replaced with a student model such as ConvNeXt-L or a MobileViT backbone, reducing parameter size by up to ~65% while maintaining sub-millimeter accuracy.

### 3.2 Knowledge Distillation (KD)

The training process involves a large **Teacher** model (HaMeR) supervising a smaller **Student** (Fast-HaMeR).

- **Feature-Level Distillation:** The student is encouraged to mimic the internal feature maps of the teacher.
- **Output-Level Distillation:** The student learns to match the teacher’s final 3D vertex predictions, effectively inheriting the teacher’s ability to handle complex occlusions.

---

## 4. Comparative Analysis

| Metric | MediaPipe | OpenPose | Fast-HaMeR |
|--------|-----------|----------|------------|
| **Output Type** | 21 Keypoints | 21 Keypoints | 778 Mesh Vertices |
| **Methodology** | Heatmap Regression | Part Affinity Fields | Iterative Error Feedback (IEF) |
| **Occlusion Handling** | Poor (Glitches) | Moderate | Excellent (Context-Aware) |
| **Hardware Requirement** | Mobile CPU | Mid-tier GPU | High-tier GPU (RTX 3060+) |
| **Accuracy (PA-MPJPE)** | ~12.0 mm | ~10.5 mm | ~7.2 mm |

---

## 5. Strengths (The Research Value)

- **In-the-Wild Robustness:** Trained on the HaMeR-6M dataset; can handle motion blur and extreme lighting.
- **Physical Consistency:** Unlike point-cloud models, the output is a manifold mesh, suitable for physics simulation (e.g., a virtual hand interacting with a virtual object).
- **Cross-View Stability:** Maintains a consistent 3D shape when the camera angle changes rapidly.

---

## 6. Weaknesses & “The Wall”

- **Environment Sensitivity:** Reliance on MMCV and Detectron2 creates a fragile dependency chain. On Windows, this often requires manual wheel building or precise version pinning.
- **Depth Ambiguity:** Like all monocular systems, global translation (exact distance of the hand from the camera) is ambiguous unless a reference object or scale cue is present.
- **High Latency on CPU:** The model is GPU-bound; inference on a modern i9 CPU may remain well below interactive frame rates compared to 30+ FPS on an RTX 3060-class GPU.

---

## 7. Future Research Directions

For a report, the following are plausible **next steps**:

1. **Temporal Smoothing:** Integrate a Transformer-XL or LSTM layer to reduce frame-to-frame jitter in video.
2. **Contact Physics:** Combine HaMeR with a contact–grasp model to distinguish actual touch from proximity.
3. **Quantization (INT8):** Further optimize for deployment on NVIDIA Jetson or other edge devices.

---

## 8. Code-Grounded Research Analysis: How Fast-HaMeR Works End-to-End

This subsection is grounded in the actual repository implementation, not only conceptual summaries. It documents the concrete algorithmic flow used by Fast-HaMeR and the theories that each module operationalizes.

#### A) System objective and formulation in code

Fast-HaMeR solves **monocular 3D hand mesh recovery** as a parametric regression task:

1. Encode an RGB crop containing a hand with a visual backbone.
2. Regress MANO parameters (`global_orient`, `hand_pose`, `betas`) and weak-perspective camera terms.
3. Decode parameters through MANO to produce dense 3D mesh vertices and joints.
4. Project 3D joints to 2D for geometric supervision and evaluation.

In code, this is implemented in `hamer/models/hamer.py` via:

- `create_backbone(cfg)` for visual encoding.
- `build_mano_head(cfg)` for parameter regression.
- `self.mano = MANO(**mano_cfg)` for parametric mesh decoding.
- `perspective_projection(...)` for reprojection to image space.

#### B) Backbone design and representation learning

The repository supports multiple backbones (`vit`, `mobilenet_v3`, `convnext`, `resnet`, `mobilevit`, etc.) through `hamer/models/backbones/__init__.py`. This is the primary efficiency lever in Fast-HaMeR:

- **Teacher regime:** ViT-based high-capacity representation (strong global context).
- **Student regime:** lighter CNN/mobile backbones with lower compute and memory.

Research concept applied: **capacity-accuracy trade-off** under constrained inference budgets. The code exposes this trade-off as a config-level choice (`MODEL.BACKBONE.TYPE`, feature dimensions, spatial resolutions).

#### C) MANO head and Iterative Error Feedback (IEF)

The MANO head (`hamer/models/heads/mano_head.py`) uses a transformer decoder with iterative updates:

- Initial estimates are loaded from MANO mean statistics (`mean_params`).
- A token attends to backbone features through cross-attention.
- Pose, shape, and camera are updated additively for `IEF_ITERS` rounds.

This implements the research idea of **iterative refinement** (IEF): instead of one-shot regression, the model repeatedly corrects estimates, which improves stability under occlusion and ambiguous hand articulation.

#### D) Geometry and losses: what is optimized

Training objective combines geometry-consistent terms in `hamer/models/losses.py` and `hamer/models/hamer.py`:

- **2D keypoint reprojection loss** (`Keypoint2DLoss`): aligns projected joints with labeled image points, confidence-weighted.
- **3D keypoint loss** (`Keypoint3DLoss`): pelvis-centered supervision for pose geometry.
- **MANO parameter loss** (`ParameterLoss`): constrains pose/shape/orientation directly in parameter space.
- **Adversarial term** (optional): discriminator regularization when enabled.

Research concept applied: **multi-objective supervision** over image-space, 3D-space, and parameter-space constraints to reduce degenerate solutions.

#### E) Knowledge Distillation implementation (Fast-HaMeR core contribution)

Distillation is implemented in `hamer/models/efficient_hamer.py` and orchestrated by `train_knowledge_dist.py`:

- Teacher model is loaded once (`load_hamer`) and frozen.
- Student is trained with standard supervised losses **plus** KD losses.
- Distillation mode is controlled by `GENERAL.DISTILLATION` (`outputs`, `features`, or `full`).

Two KD pathways are explicitly coded:

1. **Output distillation** (`OutputDistillation`): student matches teacher 2D/3D keypoints and MANO outputs.
2. **Feature distillation** (`FeatureDistillation`): teacher feature maps are channel/spatially aligned (1x1 projection + interpolation) and matched to student features.

Final optimization target:

- `combined_loss = supervised_loss + KNOWLEDGE_DISTILATION * kd_loss`

Research concept applied: **teacher-guided function approximation**, where a compact student inherits both latent representation structure and output behavior.

#### F) Inference pipeline in practice

The runtime path in `demo.py` is a multi-stage perception stack:

1. Person detection (Detectron2).
2. Whole-body keypoints (ViTPose).
3. Hand box extraction from keypoint confidence.
4. Crop/normalize hand patches (`ViTDetDataset`).
5. Fast-HaMeR forward pass for MANO + camera.
6. Mesh rendering and compositing (PyRender renderer).

This architecture reflects a common **detect-then-reconstruct** paradigm: localization and geometric reconstruction are decoupled, improving modularity and robustness.

#### G) Configuration-driven experimentation

Hydra/YACS configs in `hamer/configs_hydra` and `hamer/configs` provide reproducibility knobs:

- Backbone family and dimensions.
- Distillation mode and KD weights.
- MANO data paths and mean parameters.
- Optimization schedule, logging, and checkpoint cadence.

The practical research value is that ablation studies (e.g., backbone vs KD mode) can be run by config changes rather than code rewrites, which aligns with reproducible ML methodology.

#### H) Technical limitations observed from code + deployment behavior

From both implementation and runtime logs, the key fragility points are:

- Heavy external stack coupling (Detectron2/MMCV/OpenGL/OpenCV variants).
- Platform-sensitive rendering backends (EGL vs Windows OpenGL backend).
- Optional package assumptions in scripts (`torch_tensorrt`, `webdataset`, OpenCV DNN availability).

These constraints support an important research conclusion: **engineering compatibility is a first-class determinant of model usability**, especially for real-time 3D HMR systems on heterogeneous local setups.

---

## 9. Video Processing Extension: From Image Inference to Temporal Pipeline

The original Fast-HaMeR repository is primarily organized as an **image-based inference system**: it expects a folder of images, detects hands, reconstructs MANO geometry, and renders per-image overlays. For practical research usage, however, many target tasks are not isolated-image problems but **temporal action sequences** recorded as videos. This project therefore extends the original image pipeline into a reproducible **video-to-video processing system** without rewriting the core HaMeR model.

The key design principle was methodological conservatism:

- preserve the original inference path as much as possible,
- avoid introducing new heavyweight dependencies,
- keep model inference and downstream analytics separate,
- make the system reproducible from saved artifacts.

This resulted in a layered architecture:

1. `input_video.mp4`
2. temporary frame extraction via `ffmpeg`
3. inference through the working `demo_image.py` path
4. temporary rendered-frame collection
5. output video reconstruction
6. persistent JSON artifact generation

In research terms, this converts Fast-HaMeR from a static monocular hand-reconstruction demo into a **temporal data production pipeline** suitable for downstream motion analysis.

### 9.1 Why the video path is based on `demo_image.py`

Although the repository contains `demo.py`, the practical deployment analysis showed that `demo.py` depends on a heavier stack (`Detectron2`, ViTPose / `mmpose`, and a more fragile rendering path). On the target Windows environment, this created avoidable dependency failures. By contrast, `demo_image.py` already functioned successfully and uses:

- `RTMLib` for whole-body keypoint detection,
- hand extraction from RTMLib keypoints,
- `ViTDetDataset` for crop preparation,
- the HaMeR / Efficient-HaMeR model forward pass,
- `PyTorch3D`-based rendering.

This matters research-wise because the chosen implementation path is not just an engineering convenience. It is a **validity decision**: the documented pipeline must reflect the system that actually runs, not an idealized path that fails in deployment. In reproducible ML systems, the executable pathway is the real methodology.

### 9.2 Wrapper architecture in `run_video_hamer.py`

The video wrapper does not retrain, redesign, or reinterpret HaMeR. Instead, it acts as an **orchestration layer** around the existing image pipeline.

Its stages are:

#### A) Frame extraction

`ffmpeg` extracts frames from the source video at a user-selected processing FPS. This is an explicit temporal sampling decision:

- lower FPS reduces runtime and storage,
- higher FPS preserves more motion detail,
- the chosen FPS becomes the temporal basis for downstream timestamps and derivatives.

Frames are written temporarily with deterministic naming (`frame_%06d.jpg`), which ensures stable chronological ordering.

#### B) Image-folder inference reuse

The wrapper then calls `demo_image.py` using `sys.executable`, so it runs inside the same active Python environment. This avoids cross-environment drift and preserves compatibility with the already-working local stack.

The input to `demo_image.py` is the extracted frame folder; the output is a rendered image folder. The wrapper therefore reuses the original repository’s **folder-of-images abstraction** rather than replacing it with a custom video-native model path.

#### C) Output frame selection

The preferred output type is the full-frame rendered overlay produced by `demo_image.py`. These overlays preserve both:

- the original scene context,
- the reconstructed hand mesh visualization.

If a frame has no usable hand render, the wrapper explicitly falls back to the original extracted frame. This is important for temporal continuity: the output video remains playable and aligned frame-to-frame even when detection fails.

#### D) Video reconstruction

Finally, `ffmpeg` reconstructs the final `.mp4` using a broadly compatible codec and pixel format. This keeps the output usable for later inspection, annotation, or side-by-side comparison with other pipelines.

#### E) Temporary data discipline

All intermediate artifacts are created inside a temporary working directory and deleted in a `finally` block. The pipeline intentionally preserves only:

- the original input video,
- the final output video,
- permanent JSON outputs.

This separation is methodologically useful because it distinguishes:

- **ephemeral computation state** from
- **persistent experiment artifacts**.

### 9.3 Temporal semantics of the video extension

It is important to note that this extension does **not** make Fast-HaMeR a temporal model in the architectural sense. The model still performs per-frame monocular inference. There is no recurrent state, no transformer memory across frames, and no explicit temporal smoothing inside the model.

Instead, the project creates a **temporally indexed inference dataset** by:

- sampling frames in order,
- processing them sequentially,
- rebuilding outputs into time-consistent video and JSON artifacts.

Thus, the temporal layer is an **analysis and orchestration layer**, not a temporal network layer. This distinction is academically important because it avoids overstating what the model itself computes.

---

## 10. Structured JSON Outputs as Research Artifacts

One of the most important extensions in this project is the introduction of stable JSON outputs. These JSON files transform Fast-HaMeR from a visual demo into a system that produces **machine-readable experimental evidence**.

The project now produces three persistent JSON layers:

1. summary run JSON,
2. raw hand-data JSON,
3. derived metrics JSON.

Each layer serves a different epistemic role.

### 10.1 Summary JSON: experiment-level reporting

The first JSON layer is the summary run report generated by `run_video_hamer.py`. Its purpose is not motion analysis; its purpose is **experiment bookkeeping** and cross-pipeline comparability.

This file records:

- run identity,
- input/output video metadata,
- processing time,
- component notes,
- explicit nulls for unsupported comparison metrics,
- warnings and execution caveats.

The explicit use of `null` is methodologically significant. It prevents the system from inventing scores that Fast-HaMeR does not natively compute. This preserves scientific honesty and keeps the schema interoperable with richer pipelines that may later provide task similarity, semantic interpretation, or external scoring.

### 10.2 Raw hand-data JSON: pre-render geometric evidence

The second JSON layer is the most important one for downstream analysis. It exports **real inference-time hand data before rendering**, rather than reverse-engineering information from images.

This raw JSON is produced from the working `demo_image.py` path and contains:

- run metadata,
- video metadata,
- a frame-continuous `frames` list,
- per-frame `hands_detected`,
- per-hand structured outputs.

Per detected hand, the default export includes:

- `hand_index`
- `hand_side`
- `bbox_xyxy`
- `bbox_confidence`
- `joints_2d`
- `joints_2d_scores`
- `joints_3d_cam`
- `cam_t_full`

Optionally, behind `--export_vertices`, it can also include:

- `vertices_3d_cam`

This file is particularly valuable because it preserves the true model-side geometry in a reusable form. It makes later analyses possible without rerunning the full network, which is essential for reproducibility, ablation, and cross-method comparison.

### 10.3 Derived metrics JSON: analytic layer separated from raw export

The third JSON layer is generated by `compute_derived_metrics.py`. This script consumes the raw hand-data JSON and produces a separate derived-metrics artifact. The separation is deliberate:

- raw JSON stores what the model predicted,
- derived JSON stores what the researcher computes from those predictions.

This is a strong research design because it preserves the boundary between:

- **primary data** and
- **secondary interpretation**.

As a result, derived metrics can be recomputed later with improved formulas, different thresholds, or other comparative models, without touching the original model outputs.

---

## 11. How Raw Hand Data Is Exported from the Working Inference Path

The raw export logic is inserted into the actual `demo_image.py` inference pathway, not into the renderer output and not into post-hoc image analysis.

### 11.1 Detector-side information

`demo_image.py` uses RTMLib’s whole-body detector/tracker to obtain keypoints and confidence scores for the frame. From those body outputs, left-hand and right-hand subsets are selected using the OpenPose-style index ranges:

- `LEFT_HAND_INDICES = 91..111`
- `RIGHT_HAND_INDICES = 112..132`

For each hand candidate:

- 2D hand keypoints are extracted,
- keypoint confidence scores are extracted,
- `keypoints_to_bbox()` computes a square hand bounding box,
- mean confidence over valid keypoints becomes `bbox_confidence`.

This stage produces the detector-side 2D evidence in full-frame pixel coordinates. These are especially useful because they remain interpretable in the original image space.

### 11.2 Crop normalization and model input

The resulting hand boxes are passed into `ViTDetDataset`, which constructs normalized hand crops and associated metadata such as:

- `box_center`
- `box_size`
- `img_size`
- `right`
- `personid`

This stage is a classical **geometric normalization** step: the model does not receive the whole image directly, but a canonicalized hand crop plus metadata needed to lift crop-space predictions back into full-frame space.

### 11.3 HaMeR forward outputs

The model forward pass returns a dictionary containing, among other fields:

- `pred_cam`
- `pred_mano_params`
- `pred_cam_t`
- `focal_length`
- `pred_keypoints_3d`
- `pred_vertices`
- `pred_keypoints_2d`

For raw export, the most valuable fields are:

- `pred_keypoints_3d` for compact skeletal analysis,
- `pred_vertices` for full mesh export when explicitly requested,
- camera translation terms needed to move from crop-relative to full-frame camera coordinates.

### 11.4 Camera translation and coordinate lifting

The wrapper uses `cam_crop_to_full()` to transform crop-relative camera outputs into `pred_cam_t_full`. This is a crucial geometric step. Without it, the 3D quantities would remain tied to the crop coordinate system and would be less useful for frame-to-frame motion analysis.

The exported `joints_3d_cam` are built from:

- handedness-corrected `pred_keypoints_3d`,
- plus `cam_t_full`.

Likewise, when vertices are exported, the handedness correction is applied first and then the full camera translation is added.

This means the raw JSON does not simply store latent network outputs; it stores **camera-space geometric quantities** that are much more suitable for temporal analysis.

### 11.5 Frame continuity and timestamp reconstruction

The raw JSON preserves one record for every processed frame, even when no hands are detected:

- `hands_detected = 0`
- `hands = []`

This is methodologically important because missing detections are not hidden. They remain part of the time series and can later support:

- robustness metrics,
- continuity analysis,
- failure-mode analysis.

Timestamps are reconstructed as:

`timestamp_sec = frame_index / fps_used_for_processing`

This is appropriate because the wrapper defines the processing FPS explicitly during frame extraction. Therefore, the downstream time axis is well defined by the sampling policy of the experiment.

---

## 12. Derived Motion/Posture Metrics Layer

The first derived-metrics layer was intentionally designed as a **strong but minimal** analytic layer. It avoids premature complexity while still extracting useful temporal and geometric descriptors from the raw JSON.

### 12.1 Coverage and robustness metrics

At summary level, the system computes:

- `total_frames`
- `frames_with_detection`
- `frames_without_detection`
- `detection_rate`
- `longest_detection_streak`
- `first_detected_frame`
- `last_detected_frame`

These are not merely bookkeeping statistics. They quantify the operational robustness of the pipeline under real video conditions and provide a baseline for comparison with other hand-tracking systems.

### 12.2 Wrist motion as a compact temporal proxy

The wrist joint is used as the primary motion anchor because it is:

- present in every 21-joint hand layout,
- relatively stable,
- meaningful for global hand displacement analysis.

From `joints_3d_cam`, the derived layer computes:

- per-frame wrist position,
- wrist velocity,
- wrist acceleration.

At summary level, it reports:

- mean wrist speed,
- max wrist speed,
- mean wrist acceleration,
- max wrist acceleration,
- wrist trajectory length.

These values are computed in model-space units per second, using the reconstructed timestamps from the processing FPS.

### 12.3 Fingertip motion metrics

To capture distal articulation and motion intensity, the first implementation includes:

- thumb tip,
- index tip,
- middle tip.

For these joints, the system computes per-frame speeds and summary speed statistics. This is a useful compromise between compactness and expressiveness: fingertips carry much of the behavior relevant to grasping, manipulation, and gesture change, but exporting all possible higher-order descriptors immediately would have overcomplicated the first layer.

### 12.4 Angle metrics

The current implementation includes a small set of interpretable, geometry-based angles:

- index PIP flexion angle,
- middle PIP flexion angle,
- thumb-index spread angle.

The first two are measured as joint-centered angles between adjacent phalange segments, while thumb-index spread is computed at the wrist from the directions toward thumb tip and index tip.

This limited set was chosen because it is:

- easy to interpret,
- grounded directly in exported 3D joints,
- sufficient to characterize basic posture change,
- not yet a full speculative biomechanical model.

### 12.5 Hand openness metric

The first openness measure is defined as the **mean fingertip-to-wrist distance** across all five fingertips. This is simple, robust, and useful:

- larger values suggest a more open or extended hand,
- smaller values suggest a more closed or contracted posture.

It is not a full grasp taxonomy, but it is a defensible first-order descriptor of hand spread.

### 12.6 Trajectory metrics

The derived layer currently reports:

- wrist trajectory length,
- index fingertip trajectory length.

These are computed as cumulative Euclidean displacement across consecutive valid frames. They summarize how much the hand or a key fingertip traveled through the reconstructed 3D trajectory over the sampled sequence.

### 12.7 Smoothness metric

The first smoothness implementation uses a jerk-based approximation from wrist motion:

- position -> velocity -> acceleration -> jerk magnitude

This is explicitly documented as an approximation, not as a full clinical motion-quality index. Nevertheless, it is valuable because jerk is a standard engineering proxy for movement smoothness:

- lower jerk suggests smoother motion,
- higher jerk suggests abrupt or noisy motion.

This makes it useful both for behavioral interpretation and for diagnosing unstable model outputs.

---

## 13. Reliability Safeguards and Honest Failure Handling

An important research lesson from this implementation is that monocular 3D recovery can produce occasional **degenerate camera estimates**. In practice, some frames produced extreme `cam_t_full` depth values, which would have caused physically meaningless velocities and trajectory spikes.

To prevent this from corrupting the derived metrics layer, `compute_derived_metrics.py` applies an explicit outlier safeguard:

- wrist coordinate magnitudes above a plausibility threshold are treated as degenerate,
- the affected frame pairs are skipped for motion-based derivatives,
- warnings are recorded in the output JSON.

This is a strong example of research honesty. The system does not pretend that all frames are equally reliable; instead, it preserves the raw data and documents where derivative computations become unsafe.

Likewise, frames with missing detections are preserved as missing rather than imputed. This is important because imputation would blur the distinction between:

- model confidence failure,
- real hand absence,
- analytic uncertainty.

---

## 14. Scientific Value of the Multi-Layer Output Design

The final architecture now produces four persistent outputs per run:

1. output video,
2. summary JSON,
3. raw hand-data JSON,
4. derived metrics JSON.

This design is valuable because each artifact answers a different research question:

- **output video**: What did the system visually reconstruct?
- **summary JSON**: What happened during the run at experiment level?
- **raw JSON**: What exact geometric hand data did the model produce?
- **derived metrics JSON**: What motion/posture descriptors can be computed from that geometry?

This layered structure makes the pipeline extensible for comparison with:

- MediaPipe,
- Optical Flow pipelines,
- V-JEPA or self-supervised temporal models,
- hybrid systems.

Crucially, it enables comparison without collapsing everything into a single opaque score. That is good scientific design: preserve intermediate evidence, separate raw from derived quantities, and let future evaluation criteria evolve independently.

---

## 15. Limitations and Future Research Directions for the Video/JSON Extension

Although the pipeline is now substantially more useful, several limits remain and should be documented clearly.

### 15.1 Current limitations

- The model is still frame-wise; temporal consistency is not learned inside HaMeR.
- The current `demo_image.py` path effectively uses the first detected person and supports at most one left hand plus one right hand per frame.
- Hand identity across long sequences is side-based, not a full temporal tracking identity.
- Monocular depth remains ambiguous and can destabilize motion derivatives.
- Derived metrics are descriptive, not normative: they describe movement but do not yet score task quality.

### 15.2 Natural next research extensions

The new architecture makes several next steps possible:

- temporal smoothing of raw joints before differentiation,
- richer biomechanical angle sets for all fingers,
- thumb-index pinch or grip aperture metrics,
- continuity and dropout analysis,
- cross-model metric normalization,
- late-stage semantic interpretation on top of geometry, if desired later.

The important point is that these extensions no longer require rerunning the whole model if the raw JSON already exists. This is exactly why the separation between raw export and derived metrics is methodologically powerful.

---

## 16. Closing Research Interpretation

From a research perspective, this project no longer functions merely as a Fast-HaMeR demo deployment. It now constitutes a **structured monocular 3D hand-analysis pipeline**. The contribution is not a new neural architecture, but an experimentally sound systems layer around the existing model:

- video orchestration,
- reproducible artifact generation,
- explicit raw geometric export,
- honest derived-metrics computation,
- clear separation between inference and interpretation.

That transformation is significant. In many applied ML projects, the gap between “a model runs” and “a model supports research analysis” is large. The work documented here bridges that gap by converting Fast-HaMeR into a pipeline that can support not only visualization, but also quantitative hand-motion study and future multi-method comparison.

---

*Ali-Ahmad-Rafic*
