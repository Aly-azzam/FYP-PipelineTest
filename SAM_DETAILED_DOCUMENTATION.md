# SAM Video Compare Demo — Detailed Documentation

## 1. Project Goal

This work is a prototype interface for comparing two videos:
- one **expert** video
- one **learner** video

The objective of the prototype is to test **SAM 2** on visual object segmentation in a simple human-in-the-loop setup.

At the current stage, the prototype:
1. loads two videos,
2. extracts the **first frame** from each video,
3. lets the user click the object of interest in each frame,
4. applies **SAM 2** segmentation on both frames,
5. displays overlay results,
6. computes simple comparison metrics.

This is **not yet** the final video-wide pipeline. It is an initial validation stage before extending the same logic to multiple frames or the full video.

---

## 2. High-Level Development Logic

The work was developed progressively in small validated steps.

The logic followed was:
1. create a clean local project structure,
2. verify Python environment and basic libraries,
3. test video reading,
4. build a simple Gradio interface,
5. verify that two uploaded videos can be read,
6. install and test SAM 2 on a synthetic image,
7. confirm mask generation and overlay creation,
8. integrate SAM into the interface,
9. move from automatic center-click prompting to **manual user-selected point prompting**,
10. compute simple comparison metrics,
11. package the code for GitHub sharing.

---

## 3. Final Current Scope

### What the prototype currently does
- Uploads **two videos**.
- Extracts the **first frame** of each video.
- Displays those frames in the interface.
- Lets the user click one point on the target object in each image.
- Applies **SAM 2 image segmentation** using that point prompt.
- Generates overlays in green.
- Computes simple quantitative metrics:
  - mask area,
  - mask centroid,
  - bounding box.

### What the prototype does not yet do
- Track the object through the full video.
- Process all frames or the first 10 seconds.
- Detect the object automatically without user guidance.
- Compute semantic/action-level scores.
- Compute DTW, optical flow, wrist trajectories, or motion comparison over time.
- Produce the final JSON schema requested by the other pipeline.

---

## 4. Project Structure

The working project contains the following important files:

- `app.py`
  - Main Gradio application.
  - Handles UI, frame extraction, click selection, SAM execution, and metric display.

- `sam_image_utils.py`
  - Loads SAM 2.
  - Contains the segmentation function based on point prompting.

- `video_utils.py`
  - Early utility functions for opening videos and extracting the first frame.

- `compare_utils.py`
  - Early/basic comparison helper functions.
  - Some parts became superseded by logic integrated into `app.py`.

- `sam_runner.py`
  - Placeholder file created during the earliest prototype stage.

- `test_extract.py`
  - Small test script for validating first-frame extraction.

- `test_sam_image.py`
  - Early SAM loading test.

- `requirements.txt`
  - Basic dependency list for project setup.

- `README.md`
  - Short repository-level usage note.

- `.gitignore`
  - Excludes heavy/local/generated folders from the lightweight branch.

---

## 5. Environment and Setup Work Completed

### 5.1 Virtual environment
A local Python virtual environment was created for isolated dependency management.

### 5.2 Installed packages
The environment included installation of:
- `opencv-python`
- `numpy`
- `gradio`
- `torch`
- `torchvision`
- `hydra-core`
- `omegaconf`
- `iopath`
- `Pillow`
- `tqdm`

### 5.3 SAM 2 setup
SAM 2 was cloned locally and installed in editable mode during development.

A checkpoint was downloaded:
- `sam2_hiera_tiny.pt`

---

## 6. Important Technical Issue Encountered with SAM 2

A naming conflict happened because the local repository folder was named `sam2`, while the installed Python package was also named `sam2`.

### Problem
Python raised an error indicating shadowing/conflict between:
- the local folder name
- the installed package import path

### Fix applied
The local repo folder was renamed to:
- `sam2_repo`

Then SAM 2 was reinstalled from that renamed folder.

This was an important fix because without it, imports such as:
- `from sam2.build_sam import build_sam2`
would fail or behave ambiguously.

---

## 7. First Functional Validation Before Full Interface Work

Before connecting everything to the UI, SAM 2 was validated on a synthetic image.

### Synthetic test image
A white image was created with a red rectangle.

### Test goal
- load SAM 2,
- give it a point prompt at the center of the rectangle,
- verify that masks are produced,
- verify that the best mask can be selected,
- verify that a mask file and overlay file can be saved.

### Validation outputs
The prototype successfully produced:
- multiple masks,
- confidence scores,
- a selected best mask,
- `best_mask.png`,
- `overlay.png`.

This proved that SAM was working correctly before being integrated into the interface.

---

## 8. Interface Evolution

The interface did **not** start directly as the final current version.

### 8.1 Earliest Gradio version
The earliest interface only:
- uploaded two videos,
- called a placeholder comparison function,
- displayed simple text output.

### 8.2 Metadata comparison phase
The interface then evolved to compare simple video metadata such as:
- frame count,
- fps,
- width,
- height,
- duration.

Additional checks were added for:
- first-frame shape,
- first-frame brightness.

### 8.3 First SAM integration
The next version used SAM on the **first frame** of each uploaded video, but it used the **image center** as the point prompt.

This worked technically, but segmentation quality was inconsistent because the center of the image was often not on the real object of interest.

### 8.4 Manual click-based prompting
To fix that limitation, the interface was rebuilt with `gr.Blocks` instead of the simpler `gr.Interface` form.

The final current UI logic became:
1. upload expert and learner videos,
2. click **“Load first frames”**,
3. display the first frame of each video,
4. let the user click one point on the object in each image,
5. click **“Run SAM with selected points”**,
6. display both overlays and comparison metrics.

This was a major improvement because manual prompting drastically improved mask consistency.

---

## 9. Why Manual Point Prompting Was Necessary

When using only the center of the image, SAM often segmented:
- a large incorrect region,
- only a small part of the target object,
- or background-adjacent regions.

Manual point prompting solved that problem by letting the user click directly on the desired object.

### Practical object used in testing
The object selected in the test videos was the **blue tool** held by the person.

### Recommended click strategy
The most stable click location was:
- the **center of the blue body of the tool**,
- not on the flame,
- not on the metal part,
- not on the glove,
- not on the smoke.

---

## 10. Video Selection Work and Test Video Preparation

Originally, long videos were considered for testing.

### Problem with long videos
Long videos are less practical for this prototype because:
- upload is heavier,
- processing is slower,
- manual validation is harder,
- CPU-only SAM is costly.

### Decision made
Shorter clips were preferred.

The videos used in testing were trimmed using VLC into shorter excerpts that had:
- a better first frame,
- clear visibility of the object,
- good hand/object visibility,
- less irrelevant camera drift.

This significantly improved SAM segmentation consistency.

---

## 11. Core Functions Implemented in the Current Prototype

### 11.1 `extract_first_frame(video_path)`
Purpose:
- opens a video,
- reads its first frame,
- returns that frame.

### 11.2 `save_first_frame(video_path, output_path)`
Purpose:
- extracts first frame,
- saves it as an image file,
- returns the saved path.

### 11.3 `mask_to_overlay(image_bgr, mask)`
Purpose:
- copies the original image,
- paints the segmented mask region in green,
- returns the overlay image.

### 11.4 `prepare_preview_frames(expert_video, learner_video)`
Purpose:
- extracts and saves first frames for both videos,
- returns preview paths for Gradio display.

### 11.5 `get_select_coords(evt)`
Purpose:
- captures click coordinates from the image inside Gradio,
- returns the clicked `(x, y)` point.

### 11.6 `segment_image_from_point(image_bgr, x, y)`
Implemented in `sam_image_utils.py`.

Purpose:
- converts BGR image to RGB,
- sets the image in SAM,
- creates a positive point prompt,
- runs segmentation,
- selects the highest-score mask,
- returns the chosen mask.

### 11.7 `run_sam_with_points(...)`
Purpose:
- reads the first frame of each video,
- verifies click points exist,
- applies SAM using the selected points,
- creates overlays,
- computes metrics,
- returns overlay images and result text.

---

## 12. Metrics Implemented So Far

Three simple comparison metrics were implemented.

### 12.1 Mask area
Function:
- `mask_area(mask)`

Meaning:
- number of foreground pixels in the mask.

Comparison function:
- `compare_mask_areas(expert_mask, learner_mask)`

Output includes:
- expert mask area,
- learner mask area,
- absolute difference.

### 12.2 Mask centroid
Function:
- `mask_centroid(mask)`

Meaning:
- mean center of the segmented pixels.

Comparison function:
- `compare_mask_centroids(expert_mask, learner_mask)`

Output includes:
- expert centroid,
- learner centroid,
- `(dx, dy)` offset.

### 12.3 Bounding box
Function:
- `mask_bbox(mask)`

Meaning:
- smallest rectangle containing the mask.

Comparison function:
- `compare_mask_bboxes(expert_mask, learner_mask)`

Output includes:
- expert bbox,
- learner bbox.

---

## 13. Interpretation of the Current Numbers

Example output:

- `Surface expert = 44564`
- `Surface learner = 46127`
- `Difference = 1563`
- `Centre expert = (702, 592)`
- `Centre learner = (1241, 593)`
- `Décalage = (539, 1)`
- `BBox expert = (581, 342, 852, 856)`
- `BBox learner = (1046, 375, 1423, 816)`

### Meaning
- **Surface** = size of segmented region.
- **Difference** = absolute difference between object sizes.
- **Centre** = geometric center of segmented object.
- **Décalage** = position offset from expert to learner.
- **BBox** = spatial extent of the object in the frame.

### Limitation
These metrics are valid only if SAM segments approximately the **same semantic object** in both images.
If one mask covers the whole object and the other only a small part, comparison becomes misleading.

This issue was observed earlier and improved by selecting better first frames and more accurate click points.

---

## 14. Important Practical Observation from Testing

A major insight from testing was:

> The quality of comparison depends heavily on the consistency of SAM segmentation between both videos.

### Bad case observed earlier
- expert mask covered almost the full object,
- learner mask covered only a small region.

This produced:
- very large area differences,
- misleading centroid values,
- poor comparative interpretation.

### Improved case after better frame selection
Once a better learner video / better first frame was used:
- SAM segmented almost the whole object in both videos,
- area values became close,
- output became much more credible.

This was an important practical lesson:
- good prompting and good frame quality matter a lot.

---

## 15. Current Interface Workflow

The current user workflow is:

1. Run the application:
   - `python app.py`

2. Open Gradio locally.

3. Upload:
   - expert video,
   - learner video.

4. Click:
   - **Load first frames**

5. Click once on the target object in:
   - expert first frame,
   - learner first frame.

6. Confirm coordinate text updates.

7. Click:
   - **Run SAM with selected points**

8. Observe:
   - expert overlay,
   - learner overlay,
   - comparison text.

---

## 16. Git and Repository Work Completed

A Git repository was initialized locally for the project.

### Branch created
- `sam-work`

### Remote connected
- `Aly-azzam/FYP-PipelineTest`

### Lightweight push strategy used
A `.gitignore` was created to exclude heavy and machine-specific items.

Excluded items included:
- `.venv/`
- `__pycache__/`
- `outputs/`
- `checkpoints/`
- `sam2_repo/`
- `*.pyc`

### Purpose of the lightweight branch
The goal of the first pushed branch was to:
- share code,
- show progress,
- avoid pushing large binaries or machine-specific files.

---

## 17. Why the First GitHub Branch Was Not Fully Runnable Immediately

Although the code branch was successfully pushed, it did **not** include all runtime assets.

### Missing from lightweight branch
- local virtual environment,
- SAM checkpoint file,
- local cloned SAM repository,
- generated outputs,
- other heavy local items.

### Consequence
A collaborator could:
- read the code,
- understand the workflow,
- inspect the structure,

but would still need to:
- install dependencies,
- install SAM 2,
- download the checkpoint,
- then run the project.

---

## 18. Documentation Files Added for Sharing

Two key sharing documents were created:

### `requirements.txt`
Contains the main Python dependencies.

### `README.md`
Contains a concise summary of:
- project purpose,
- install steps,
- SAM install reminder,
- checkpoint placement,
- launch command,
- current limitation.

---

## 19. Important GitHub Constraint for a “Full” Branch

A later requirement appeared: create a new branch that includes more of the project so that a teammate can work more directly on it.

However, this must consider practical GitHub limits and repo hygiene.

### Key constraint
Pushing literally **everything** is not always practical or allowed because:
- the SAM checkpoint file is large,
- virtual environments are machine-specific,
- repository size can become problematic,
- some large files may require Git LFS.

### Best practical definition of a “full runnable branch”
A more realistic “complete” branch should include:
- source code,
- documentation,
- `requirements.txt`,
- `README.md`,
- optionally `sam2_repo/`,
- optionally checkpoint via Git LFS,
- optionally sample data if allowed by size.

It should **not** normally include `.venv/`.

---

## 20. Why `.venv` Should Normally Not Be Pushed

Even if the goal is “everything included,” a Python virtual environment is usually the wrong thing to push because:
- it is OS-specific,
- it may contain absolute paths tied to one machine,
- it is large,
- it is not the standard way to share Python projects.

The correct sharing method is:
- push source code,
- push dependency manifest,
- document installation,
- push large model files with Git LFS if needed.

---

## 21. Discussion About Next Technical Direction

After reviewing the current result, it was concluded that the current work is **not sufficient** as a final project step.

### New requirement identified
Move from:
- segmentation on only the first frame

to:
- segmentation on the full video,
- or at least the first 10 seconds.

### Additional desired improvement
Automatic object detection was also discussed, specifically:
- automatically selecting the object being held in the hand,
- even if the object changes between videos.

### Practical conclusion reached
This is possible only partially.

A realistic roadmap is:
1. first: user click on first frame, then track through video,
2. later: semi-automatic object initialization,
3. only after that: more ambitious automatic object detection.

---

## 22. What Was Explained About Automatic Detection

It was clarified that **SAM alone does not automatically know which object is “the important one.”**

SAM is mainly a **prompted segmentation system**.

So for true automatic object selection in new arbitrary videos, another upstream logic would be needed, for example:
- detect hands,
- detect object candidates near hands,
- choose the likely manipulated object,
- initialize SAM from that object.

This means full automation is a separate research/engineering layer on top of SAM.

---

## 23. JSON Output Requirement Introduced by Teammate

A new requirement was introduced:
- the output should eventually follow a JSON structure similar to another pipeline’s response.

The target shape includes sections such as:
- `run`
- `expert_video`
- `learner_video`
- `overall_score`
- `metrics`
- `confidences`
- `explanation`
- `warnings`

### Current state relative to that requirement
The prototype does **not yet** produce this JSON.

### What is already available to populate JSON later
Current SAM prototype can already provide values such as:
- mask areas,
- centroid positions,
- centroid offset,
- bounding boxes,
- notes about manual prompting,
- pipeline name,
- explanatory warnings.

So the current text output can later be converted into structured JSON.

---

## 24. Recommended Next Step After This Documentation

The next strong engineering step is:

### Video-wide version
Apply segmentation not only to the first frame but to:
- the whole video if short,
- otherwise the first 10 seconds.

### Practical reason
This is more meaningful than current first-frame-only logic, while still remaining achievable.

### Expected work involved
That next phase will require:
- reading multiple frames,
- selecting a frame budget,
- applying SAM frame by frame,
- generating overlays across time,
- producing output videos,
- then extending metrics over multiple frames.

---

## 25. Summary of What Has Been Achieved

This work successfully established a complete prototype chain:

1. local project setup,
2. video reading validation,
3. Gradio UI prototype,
4. SAM 2 installation and debugging,
5. SAM image segmentation validation,
6. integration of SAM into the interface,
7. manual point-based prompting,
8. overlay generation,
9. metric computation,
10. GitHub sharing on a dedicated branch,
11. collaborator-oriented documentation setup.

This means the project already has a **real working SAM-based prototype**, even though it is still limited to first-frame comparison.

---

## 26. Current Strengths of the Prototype

- Real SAM 2 integration works.
- Interface is interactive and demonstrable.
- Manual point prompting improves segmentation quality significantly.
- Metrics are interpretable and easy to explain.
- The codebase is already structured enough for extension.
- The workflow is now documented and shareable.

---

## 27. Current Limitations of the Prototype

- First-frame only.
- Manual point prompt required.
- No temporal tracking across video.
- No automatic object detection.
- No action-level comparison.
- No semantic/VLM explanation.
- No final JSON output yet.
- No robust scoring logic yet.

---

## 28. Recommended Development Roadmap

### Phase 1 — already achieved
- first-frame SAM comparison with manual point prompts.

### Phase 2 — immediate next target
- process the first 10 seconds of both videos,
- generate overlay video outputs.

### Phase 3 — structured output
- replace plain result text with JSON response schema.

### Phase 4 — semi-automatic initialization
- reduce or remove manual point selection where possible.

### Phase 5 — stronger comparison logic
- multi-frame metrics,
- trajectory comparison,
- motion-based or action-based scoring.

---

## 29. One-Sentence Status Statement

This project currently provides a working **SAM 2 first-frame video comparison prototype with manual click prompting, overlay visualization, and basic quantitative mask comparison metrics**, and is ready to be extended toward video-wide processing and structured JSON outputs.