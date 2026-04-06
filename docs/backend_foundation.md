# Backend foundation — implementation notes

This document records what was implemented in the AugMentor pipeline comparison demo **before** any real AI components (MediaPipe, VLM, DTW, etc.), and why that order matters.

---

## 1. Common result schema

**File:** `backend/schemas/result_schema.py`

This file defines the **global result contract** for the whole app.

It was implemented early because every pipeline, no matter how simple or advanced, must return its output in the **same structure**.

The schema currently includes:

### `VideoMeta`

Stores metadata about each video, such as:

- `filename`
- `path` (optional)
- optional `duration_sec`
- optional `fps`
- optional `width` and `height`

### `Metrics`

Stores optional quantitative metrics that may or may not exist depending on the pipeline, such as:

- joint angle deviation (`joint_angle_deviation`)
- trajectory deviation (`trajectory_deviation`)
- velocity difference (`velocity_difference`)
- tool alignment deviation (`tool_alignment_deviation`)
- DTW cost (`dtw_cost`)
- semantic similarity (`semantic_similarity`)
- optical flow similarity (`optical_flow_similarity`)
- extra custom metrics via an optional `extra` dictionary

### `Confidences`

Stores optional confidence values:

- `overall`
- `same_task`
- `score`
- `explanation`

### `Explanation`

Stores the explanation layer, including:

- main explanation text (`text`)
- `strengths` and `weaknesses` (lists)
- optional `raw_vlm_output`
- optional `structured_notes`

### `RunMeta`

Stores metadata about the execution itself, including:

- `run_id` (UUID hex, default-generated)
- `pipeline_name` (string, aligned with pipeline identity)
- optional `processing_time_sec`
- `created_at` (UTC timestamp, default-generated)
- optional `component_notes`

### `PipelineResult`

This is the **final root schema** returned by any pipeline. It combines:

- `run` metadata
- expert and learner `VideoMeta`
- `overall_score` (optional)
- `metrics`
- `confidences`
- `explanation`
- `warnings` (list of strings)

### Why this file mattered

It defines the **output shape of the entire system**. Without it, each pipeline could return a different result format, which would make frontend integration, benchmarking, graph generation, and report analysis inconsistent.

---

## 2. Pipeline schema layer

**File:** `backend/schemas/pipeline_schema.py`

This file defines **pipeline identity**, **selection**, **input contract**, and lightweight **execution/descriptor** shapes.

Its purpose is to standardize:

- which pipelines officially exist
- how a pipeline is selected (`PipelineSelection`; useful when the body is only a choice)
- what input is passed to the backend before dispatching (`PipelineInput`)
- what lightweight execution metadata looks like (`PipelineExecutionMeta`)
- optional human/registry-facing descriptors (`PipelineDescriptor`)

### `PipelineName`

A strict **string enum** of all supported internal pipeline names:

- `vlm_only`
- `mediapipe_vlm`
- `mediapipe_dtw_vlm`
- `mediapipe_vjepa_vlm`
- `mediapipe_vjepa_dtw_vlm`
- `mediapipe_sam2_dtw_vlm`
- `mediapipe_vjepa_sam2_dtw_vlm`
- `mediapipe_vjepa_grounded_sam2_dtw_vlm`
- `mediapipe_optical_flow_dtw_vlm`

### `PipelineSelection`

Minimal schema: selected `pipeline_name` only.

### `PipelineInput`

App-level input contract before a pipeline runs:

- `pipeline_name`
- `expert_video_path`
- `learner_video_path`
- optional `config` (`Dict[str, Any]`)

### `PipelineExecutionMeta`

Lightweight execution status metadata:

- `pipeline_name`
- `success`
- optional `message`
- optional `error`
- optional `processing_time_sec`

### `PipelineDescriptor`

Optional structure for name, description, and `components_used` (list of strings).

### Why this file mattered

The backend needs a **strict, validated** way to identify pipelines. This prevents naming drift and gives the dispatcher and app entrypoint a single contract.

---

## 3. Pipeline dispatcher

**File:** `backend/services/pipeline_dispatcher.py`

This is the **central routing layer** between the app entrypoint and pipeline implementations.

### Current responsibilities

- Accept validated `PipelineInput`
- Resolve the selected pipeline in a **registry** (`Dict[PipelineName, Runner]`)
- Call the corresponding runner (`PipelineInput` → `PipelineResult`)
- Return a valid `PipelineResult`

Because real pipeline modules are not implemented yet, the dispatcher uses **placeholder runners** for every registered name. It does **not** raise during normal dispatch for “not implemented”; it returns a structured placeholder result.

### What the dispatcher does today

- Extracts filenames from expert/learner paths (Windows- and POSIX-friendly via `PureWindowsPath` / `PurePosixPath`)
- Builds valid `VideoMeta` and `RunMeta`
- Returns a placeholder `PipelineResult` with:
  - correct `pipeline_name` on `RunMeta`
  - expert and learner video metadata
  - `overall_score=None`
  - explanation text: pipeline is registered but not implemented yet
  - warning: `pipeline_not_implemented`

### Registry behavior

The registry maps each `PipelineName` to a callable. Today all entries use placeholders generated by `_make_placeholder_runner`. Later, individual keys can be reassigned to real `run` functions imported from `backend/pipelines/<name>/`.

### Why this file mattered

It establishes **execution architecture** before components: the app can already accept a pipeline name, route it, and return a **standardized** result.

---

## 4. Backend entrypoint

**File:** `backend/app.py`

FastAPI application entrypoint for the demo API.

### Routes

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Returns `{"status": "ok"}` |
| `GET` | `/pipelines` | Returns `{"pipelines": [...]}` from `PipelineName` (single source of truth) |
| `POST` | `/compare` | JSON body = `PipelineInput`; calls `dispatch_pipeline`; `response_model=PipelineResult` |

### Additional behavior

- **CORS:** `CORSMiddleware` with open origins/methods/headers for local dev
- **Errors:** `ValueError` from the dispatcher → HTTP **400** with `HTTPException(detail=str(exc))`
- **Local run:** minimal `if __name__ == "__main__"` block running `uvicorn` with reload on `127.0.0.1:8000`

### Why this file mattered

It completes the initial path:

**client → FastAPI → `dispatch_pipeline` → `PipelineResult`**

even before real model code exists.

---

## 5. What is already working conceptually

At this stage, the foundation includes:

- Agreed **project layout** (components, pipelines, services, schemas, outputs)
- **Standardized result format** (`PipelineResult`)
- **Standardized pipeline list** (`PipelineName`)
- **Request validation** via `PipelineInput` on `/compare`
- **Routing** via the dispatcher registry
- **Valid placeholder results** for every registered pipeline

So the **core architecture is ready** before implementing reusable AI modules.

---

## 6. Why this order mattered

Implementation intentionally did **not** start with MediaPipe, VLM, or SAM.

Order used:

1. Result schema  
2. Pipeline schema  
3. Dispatcher  
4. App entrypoint  

That order defines:

- system **contract** (inputs/outputs)
- **execution flow** and routing
- **consistent** JSON for the frontend and benchmarks

Building components first risks inconsistent return shapes, unfair comparisons, and messy frontend work.

---

## 7. Current project status

### Done (this phase)

- Folder structure for backend demo
- `backend/schemas/result_schema.py`
- `backend/schemas/pipeline_schema.py`
- `backend/services/pipeline_dispatcher.py`
- `backend/app.py`

### Not started yet

- Reusable AI components under `backend/components/`
- Real runners under `backend/pipelines/`
- Frontend wired to this API (paths + `fetch` / `POST /compare`)
- `result_saver`, `timing_service`, `video_loader` implementations
- Persisted outputs beyond placeholders
- Graphs / benchmark analysis scripts

---

## 8. What comes next

The next step is the **first reusable component**, ideally **VLM**, because:

- the first baseline is `vlm_only`
- it is the shortest path to a **first real** end-to-end result
- it exercises the full stack early

Suggested next phase:

1. Implement `backend/components/vlm/` (reusable VLM interface)
2. Implement `backend/pipelines/vlm_only/` and a real `run(pipeline_input) -> PipelineResult`
3. Register that runner in `REGISTRY` for `PipelineName.VLM_ONLY`
4. Run `/compare` against real videos and inspect a non-placeholder `PipelineResult`

---

## 9. Summary

So far, **no AI models** are implemented, but the **backend skeleton** of the benchmark demo is in place.

Future pipelines will:

- plug into the **same app**
- follow the **same schemas**
- be **dispatched** the same way
- return results in the **same structure**

That keeps comparison **fair**, integration **clean**, and the system **scalable** before the first real component lands.
