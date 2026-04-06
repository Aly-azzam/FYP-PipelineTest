# FYP-PIPELINETEST — Folder Responsibility Explanation

## Purpose of the project structure

This structure is designed for a pipeline comparison demo application for AugMentor. The goal is not to build the final full platform, but to build a clean benchmark demo where different candidate pipelines can be tested fairly on the same task:

- 1 expert video
- 1 learner video
- 1 selected pipeline
- **output** = score, VLM explanation, and full JSON result

The structure is organized so that:

- the frontend stays simple
- reusable model logic is separated into components
- each candidate pipeline has its own folder
- shared backend utilities are centralized
- outputs are saved in an organized way
- documentation is easy to maintain

---

## Top-level folders

### `frontend/`

This folder contains the demo user interface.

Its role is to provide a very simple interaction layer where the user can:

- upload or select the expert video
- upload or select the learner video
- choose which pipeline to run
- launch the comparison
- display the returned score, explanation, and JSON result

This folder is intentionally minimal because the project is a demo benchmark app, not a full frontend product.

#### Files inside `frontend/`

| File | Role |
|------|------|
| `index.html` | Main page structure |
| `style.css` | Visual styling of the page |
| `app.js` | Frontend logic and communication with backend |

### `backend/`

This folder contains the main application logic.

Its role is to:

- receive the input videos and selected pipeline
- call the correct pipeline
- coordinate processing
- return the final result to the frontend
- save outputs for later comparison and reporting

The backend is the core of the project because it contains both the reusable components and the candidate pipelines.

### `docs/`

This folder contains the manual project documentation.

Its role is to help the team keep track of:

- which videos were tested
- notes about each pipeline
- experiment observations and progress

This folder is useful because the number of videos is small and the experiments are being documented manually.

---

## Backend subfolders

### `backend/components/`

This folder contains the reusable AI / processing building blocks.

Each folder inside `components` corresponds to one model or one major technical module. These components are not full pipelines by themselves. Instead, they are the individual pieces that pipelines combine.

The idea is:

- write the logic of each component once
- reuse it in many pipelines
- keep comparison fair
- avoid duplicating code

#### Responsibility of each component folder

##### `backend/components/vlm/`

Responsible for all logic related to the Vision-Language Model.

Its job may include:

- preparing VLM prompts
- sending image/video inputs to the VLM
- receiving and parsing VLM outputs
- formatting explanation results

This component is used in all pipelines that require natural-language analysis or explanation.

##### `backend/components/mediapipe/`

Responsible for all logic related to MediaPipe-based hand / motion extraction.

Its job may include:

- extracting hand landmarks
- generating structured motion features
- preparing pose-based data for downstream comparison

This component is the main fine-motor perception block.

##### `backend/components/dtw/`

Responsible for Dynamic Time Warping logic.

Its job may include:

- aligning expert and learner motion sequences
- handling speed differences between performances
- returning temporal correspondence or alignment cost

This component is used in pipelines that need robust temporal comparison.

##### `backend/components/vjepa/`

Responsible for all logic related to V-JEPA / video semantic representation.

Its job may include:

- extracting high-level video representations
- measuring semantic similarity between clips
- adding a more contextual video understanding layer beyond raw geometry

This component is useful for pipelines that want semantic understanding in addition to motion structure.

##### `backend/components/sam2/`

Responsible for all logic related to SAM 2 segmentation.

Its job may include:

- segmenting relevant video regions
- identifying important objects or interaction areas
- tracking masks across frames if needed

This component is useful when the pipeline needs detailed video region understanding rather than only hand landmarks.

##### `backend/components/grounded_sam2/`

Responsible for all logic related to Grounded SAM 2.

Its job may include:

- grounding text prompts to objects/regions
- locating specific tool/material targets
- segmenting and tracking them more precisely than generic segmentation

This component is separated from normal SAM 2 because it represents a distinct technical capability.

##### `backend/components/optical_flow/`

Responsible for optical flow computation.

Its job may include:

- calculating pixel-level motion between frames
- capturing motion patterns when landmark-based methods are insufficient
- providing complementary movement information

This component is useful for motion-sensitive pipelines that want a lower-level movement signal.

##### `backend/components/common/`

Responsible for shared helper logic used by multiple components.

Its job may include:

- shared preprocessing helpers
- common data conversions
- reusable utility functions for component-level operations

This folder exists to prevent repetition across component modules.

---

### `backend/pipelines/`

This folder contains the actual candidate pipelines being compared.

Each folder inside `pipelines` represents one complete experimental pipeline.

A pipeline is not a single model. A pipeline is a combination of components arranged in a specific order to process the videos and produce the final result.

This folder is the heart of the comparison system because it defines the different benchmark candidates.

#### Responsibility of each pipeline folder

##### `backend/pipelines/vlm_only/`

Pipeline that relies only on the VLM.

Its role is to serve as the simplest baseline:

- expert video + learner video
- direct multimodal comparison
- direct score / explanation / JSON output

##### `backend/pipelines/mediapipe_vlm/`

Pipeline that combines:

- MediaPipe
- VLM

Its role is to use structured hand/motion features together with a language explanation layer.

##### `backend/pipelines/mediapipe_dtw_vlm/`

Pipeline that combines:

- MediaPipe
- DTW
- VLM

Its role is to compare motion more robustly by aligning expert and learner sequences before explanation.

##### `backend/pipelines/mediapipe_vjepa_vlm/`

Pipeline that combines:

- MediaPipe
- V-JEPA
- VLM

Its role is to mix structured motion information with semantic video representation.

##### `backend/pipelines/mediapipe_vjepa_dtw_vlm/`

Pipeline that combines:

- MediaPipe
- V-JEPA
- DTW
- VLM

Its role is to include both motion alignment and semantic understanding.

##### `backend/pipelines/mediapipe_sam2_dtw_vlm/`

Pipeline that combines:

- MediaPipe
- SAM 2
- DTW
- VLM

Its role is to enrich structured motion comparison with segmentation-based scene/object understanding.

##### `backend/pipelines/mediapipe_vjepa_sam2_dtw_vlm/`

Pipeline that combines:

- MediaPipe
- V-JEPA
- SAM 2
- DTW
- VLM

Its role is to build a stronger multimodal comparison using motion, segmentation, semantics, alignment, and explanation together.

##### `backend/pipelines/mediapipe_vjepa_grounded_sam2_dtw_vlm/`

Pipeline that combines:

- MediaPipe
- V-JEPA
- Grounded SAM 2
- DTW
- VLM

Its role is to provide the most advanced grounded multimodal pipeline in the benchmark.

##### `backend/pipelines/mediapipe_optical_flow_dtw_vlm/`

Pipeline that combines:

- MediaPipe
- Optical Flow
- DTW
- VLM

Its role is to test whether optical flow adds useful motion information beyond landmark extraction.

---

### `backend/services/`

This folder contains the shared backend support logic.

Unlike components, this folder does not represent AI models. Unlike pipelines, it does not define candidate benchmark combinations.

Its role is to provide common backend operations needed by the application.

#### Responsibility of each service file

##### `pipeline_dispatcher.py`

Responsible for choosing and launching the correct pipeline based on the user’s selection.

It acts as the routing layer between the frontend request and the pipeline folders.

##### `video_loader.py`

Responsible for reading, validating, or preparing video inputs before they are passed into the selected pipeline.

It centralizes video input handling instead of repeating it inside every pipeline.

##### `result_saver.py`

Responsible for saving the outputs of each run.

Its job may include:

- saving final JSON results
- organizing result files
- creating a consistent output history for later analysis

##### `timing_service.py`

Responsible for tracking runtime and processing duration.

Its job may include:

- start/end timing
- reporting total pipeline execution time
- helping benchmark pipeline speed

---

### `backend/schemas/`

This folder contains the common data structures / result definitions.

Its role is to ensure that every pipeline returns outputs in a consistent format.

This is extremely important for fairness, because even if pipelines are internally different, their final returned structure should remain standardized.

#### Files inside `schemas/`

##### `result_schema.py`

Responsible for defining the structure of the final benchmark result.

This may include fields such as:

- pipeline name
- score
- explanation
- metrics
- warnings
- runtime
- full JSON structure

##### `pipeline_schema.py`

Responsible for defining pipeline-related structures or metadata.

This may include:

- accepted pipeline names
- expected pipeline inputs
- internal pipeline contract format

---

### `backend/outputs/`

This folder stores the saved outputs of executed runs.

Its role is to preserve benchmark results for later use in:

- graphs
- tables
- report analysis
- experiment review

Each run can produce artifacts such as:

- full JSON result
- score summary
- explanation output
- timing result

This folder is important because the benchmark is not only about displaying a result live, but also about keeping the results for comparison and documentation.

---

## Root backend files

### `backend/app.py`

This is the main backend entry point.

Its role is to:

- start the backend application
- expose the route(s) needed by the frontend
- connect frontend requests to the backend services and pipelines

### `backend/__init__.py`

This marks the backend as a Python package.

Its role is structural rather than functional.

---

## Docs files

### `docs/tested_videos.md`

Used to document which expert and learner videos were tested.

This helps keep track of:

- video names
- test pairings
- notes about what each video represents

### `docs/pipeline_notes.md`

Used to document observations and notes about each pipeline.

This may include:

- strengths
- weaknesses
- known issues
- configuration notes

### `docs/experiment_log.md`

Used to record the progress of experiments over time.

This may include:

- what was tested
- what worked
- what failed
- what needs to be improved next

---

## Final summary

In simple terms:

| Location | Role |
|----------|------|
| `frontend/` | Demo interface |
| `backend/components/` | Reusable technical building blocks |
| `backend/pipelines/` | Complete candidate pipelines built from those blocks |
| `backend/services/` | Shared backend support logic |
| `backend/schemas/` | Common output structure |
| `backend/outputs/` | Saved benchmark results |
| `docs/` | Manual project documentation |
