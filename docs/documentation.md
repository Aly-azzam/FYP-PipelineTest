# V-JEPA (`vjepa_only`) — Pipeline documentation

This document explains how **V-JEPA 2.1** is integrated in this project, what data it produces, how scores are derived, and known limitations. It reflects the implementation in `backend/components/vjepa/` and `backend/pipelines/vjepa_only/`.

---

## 1. Purpose

The **`vjepa_only`** pipeline compares an **expert** video and a **learner** video using **self-supervised video embeddings** (no external VLM for the core score). It outputs:

- A `PipelineResult` (score, `semantic_similarity`, explanation, warnings, etc.)
- JSON artifacts under `backend/outputs/json/vjepa_only/` (main result, derived metrics, optional raw embedding debug)

The goal is **semantic similarity** between temporally segmented clips, not a validated human skill rubric.

---

## 2. What V-JEPA is (big picture)

**V-JEPA** (Video Joint-Embedding Predictive Architecture) is a **neural network trained on large-scale video without manual labels**. It learns representations that capture appearance and motion patterns in short clips.

In this project we use only the **encoder**:

- **Input:** a stack of RGB frames (sampled from a video segment).
- **Output:** one **embedding vector** per segment (a fixed-length list of floats).

We do **not** use the full pretraining objective at inference time—only the frozen encoder produces embeddings.

---

## 3. How V-JEPA is loaded (correct setup)

Implementation: `backend/components/vjepa/service.py`.

- **Torch Hub** loads `vjepa2_preprocessor` and the encoder (e.g. `vjepa2_1_vit_base_384`) with **`pretrained=False`** to avoid the upstream broken localhost checkpoint URL.
- **Weights** are loaded from Meta’s public CDN (`dl.fbaipublicfiles.com/vjepa2/`) or from `VJepaConfig.checkpoint_path` / `checkpoint_url` if set.
- Encoder weights are read from the checkpoint keys expected by the published models (`ema_encoder` / `target_encoder`, with fallbacks).
- On failure, the service can fall back to a **chunk-based RGB heuristic** (not semantically comparable to real V-JEPA); runs then include `temporary_embedding_fallback` in warnings when applicable.

**Configuration:** `backend/components/vjepa/config.py` (`hub_repo`, `model_name`, `frames_per_clip`, `num_segments`, device, etc.).

---

## 4. End-to-end flow in this codebase

### 4.1 Segmenting each video

- Each video is split into **`num_segments`** equal parts along the **original timeline** (default **8**).
- **Per segment**, `sample_uniform_frames` in `backend/components/vjepa/sampler.py` draws **`frames_per_clip`** frames (default **30**) uniformly between that segment’s start and end frame indices.

### 4.2 Encoding one segment

For each segment’s frames (`service.py`, `_encode_frames`):

1. **Preprocessor** converts frames to the tensor format expected by the model.
2. **Encoder forward pass** runs with `torch.no_grad()`.
3. **Pooling:** If the encoder returns a 3D token tensor `[batch, tokens, dim]`, embeddings are **`mean`-pooled over tokens** to produce one vector per clip.
4. Result is a **NumPy float32 vector** (one embedding per segment).

### 4.3 Comparing expert vs learner

Implementation: `backend/pipelines/vjepa_only/__init__.py`.

1. Build an **N×N similarity matrix** where each cell is **cosine similarity** between one expert segment embedding and one learner segment embedding (`backend/components/vjepa/similarity.py`).
2. **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`) on **cost = 1 − similarity** finds a **one-to-one** assignment of expert segments to learner segments (no duplicate learner segment).
3. **Scoring heuristics** (re-centering, bottom-k blend, gap/STD penalties, nonlinear mapping) combine matched similarities into **`semantic_similarity`** and **`overall_score`**. These rules are **temporary calibration**, not a validated rubric.

---

## 5. Cosine similarity (simple explanation)

- Each embedding is a **vector** (high-dimensional arrow).
- **Cosine similarity** measures how **aligned** two vectors are in direction (after normalization used in the dot-product formula), typically in **[-1, 1]**.
- **High cosine** means the two clips are **close in the model’s feature space**—not automatically “the same task” or “human-related content.”
- Many **unrelated** real-world videos still get **high** cosines (e.g. 0.92–0.98) because they share **lighting, scene type, motion texture, color**, etc. This is a **known limitation** of generic embeddings + cosine for fine-grained “same vs different.”

---

## 6. What V-JEPA gives you (data and information)

### 6.1 Direct model outputs

| Item | Description |
|------|-------------|
| **Segment embeddings** | One vector per temporal segment per video (not exported in full by default). |
| **Cosine similarities** | Scalars between pairs of segment embeddings (full matrix + matched pairs). |
| **Embedding norms** | Optional diagnostics (e.g. in derived JSON / per-segment metrics) to verify vectors are non-degenerate. |

### 6.2 What is not produced by V-JEPA alone

- No **semantic labels** (e.g. “cooking,” “correct technique”).
- No **guarantee** that unrelated videos get low similarity.
- No **pose / skeleton / tool** structure unless another component provides it.

---

## 7. JSON outputs (what to look for)

Artifacts are written under `backend/outputs/json/vjepa_only/`:

| File pattern | Contents |
|--------------|----------|
| `vjepa_<run_id>.json` | Main `PipelineResult`: score, `metrics.semantic_similarity`, `metrics.extra` (matrix, matches, raw/adjusted stats, penalties), explanation, warnings. |
| `vjepa_<run_id>_derived_metrics.json` | Run metadata, paths, summary metrics, embedding mode (`real_encoder_used`, `fallback_used`, model, frames, segments), per-segment rows, `video_debug`. |
| `vjepa_<run_id>_raw_embeddings.json` | Optional; controlled by config — previews/norms, not necessarily full vectors. |

**Trust signals for “real V-JEPA ran”:**

- `embedding_mode.real_encoder_used: true`
- `embedding_mode.fallback_used: false`
- No `temporary_embedding_fallback` in `warnings`

---

## 8. Scoring calibration (why scores can look wrong)

The pipeline applies **heuristic** steps after raw cosine:

- **Re-centering:** Raw cosines are shifted/scaled using a **baseline** (e.g. fixed value in code such as `0.96`) so that small differences in the high-similarity band map to a usable **[0, 1]** range before penalties and power mapping.
- **Penalties** (gap between mean and min, segment spread) and **nonlinear mapping** (e.g. gamma on adjusted similarity) control how harsh the final **0–100** score is.

If **all** matched raw similarities fall **below** the chosen baseline, adjusted values can **clamp to zero**, producing a **0** score even when raw cosines are still “high” in absolute terms. Interpretation should always consider **`segment_similarities_raw`** and the **similarity matrix**, not only `overall_score`.

---

## 9. Known limitations (summary)

1. **Cosine saturation:** Unrelated clips can still show **high** cosine in V-JEPA space.
2. **Temporal alignment:** Hungarian matching allows **different segment order**; high similarity does not mean “same timeline.”
3. **Heuristic score:** The **0–100** number is **not** a calibrated exam grade; it is a **project-specific** combination of similarities and penalties.
4. **CPU vs GPU:** Encoder runs on CPU if CUDA is unavailable; results are the same numerically, but slower.

---

## 10. Key source files

| Path | Role |
|------|------|
| `backend/components/vjepa/service.py` | Load V-JEPA, segment sampling, encode frames, optional fallback. |
| `backend/components/vjepa/config.py` | Model and segment/frame settings. |
| `backend/components/vjepa/sampler.py` | Uniform frame sampling with optional frame range. |
| `backend/components/vjepa/similarity.py` | Cosine similarity and helpers. |
| `backend/pipelines/vjepa_only/__init__.py` | Matrix, Hungarian match, scoring, JSON export. |

---

## 11. Revision note

This documentation captures the **behavior and design** of the `vjepa_only` integration (segment-wise comparison, Hungarian matching, re-centering, penalties, gamma scoring). **Constants and weights** in `vjepa_only/__init__.py` may change over time; always refer to the code for exact formulas.
