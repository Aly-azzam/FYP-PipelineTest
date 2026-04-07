# VLM Only Pipeline Documentation

## 1. Research Documentation

### 1.1 Pipeline Name
`vlm_only`

### 1.2 Objective
The objective of this pipeline is to evaluate whether a Vision-Language Model (VLM), using only the two raw input videos, can:
- determine whether the expert and learner videos correspond to the same task,
- estimate the learner’s overall performance relative to the expert,
- identify strengths and weaknesses,
- describe the main visual differences between the two performances,
- generate a detailed explanation useful for qualitative analysis and report writing.

This pipeline is intentionally designed as the **semantic baseline** of the benchmark.  
It does **not** use structured motion extraction, hand landmarks, DTW alignment, optical flow, or deterministic metric computation.

### 1.3 Input and Output Definition
**Inputs**
- 1 expert raw video
- 1 learner raw video

**Outputs**
- estimated score out of 100
- same-task judgment
- confidence values
- strengths
- weaknesses
- key visual differences
- final verdict
- full explanation text

### 1.4 Experimental Setup
The pipeline was tested twice on the same expert/learner pair:
- `expert.mp4`
- `learner.mp4`

The two runs were executed through the `vlm_only` pipeline using the same raw video paths.  
The purpose of repeating the same run was to observe the reproducibility and stability of the VLM-only baseline.

### 1.5 Run Results Summary

| Metric | Run 1 | Run 2 |
|---|---:|---:|
| Pipeline | `vlm_only` | `vlm_only` |
| Overall Score | 65 | 65 |
| Same Task Label | true | true |
| Same Task Confidence | 1.00 | 1.00 |
| Overall Confidence | 0.90 | 0.95 |
| Processing Time (s) | 85.797 | 95.829 |
| Warnings | none | none |

### 1.6 Main Consistent Findings
Across both runs, the VLM-only pipeline produced a highly stable high-level judgment:

- It correctly identified that both videos represent the same task.
- It gave the same estimated score in both runs: **65/100**.
- It consistently identified the learner’s main weakness as:
  - excessive smoke generation,
  - weaker motion fluidity,
  - less precise and less clean cutting than the expert.
- It consistently recognized that the learner understood the task and used the correct tools, but still lacked the finesse and efficiency shown by the expert.

This suggests that the VLM-only baseline is relatively stable for:
- coarse task recognition,
- global score estimation,
- high-level qualitative assessment.

### 1.7 Differences Observed Between the Two Runs
Although the two runs were strongly aligned in the final judgment, they were not identical in wording or detail.

#### Run 1 emphasized:
- the expert’s explicit demonstration of “what happens if the tool stays planted,”
- the idea of **delicacy** in handling the hot knife,
- smoke as evidence of too much dwelling time or pressure,
- the learner’s correct use of safety tools and understanding of the process.

#### Run 2 emphasized:
- smoother and more continuous movement in the expert,
- slightly wavering lines and less crisp corners in the learner,
- more visible issues on curves,
- less stable coordination between the guiding hand and the cutting tool,
- the learner’s lower efficiency and more charred edges.

### 1.8 Interpretation of Variability
The repeated runs show that the VLM-only pipeline is **stable in decision**, but **variable in language**.

#### Stable aspects
- same score,
- same-task label,
- same-task confidence,
- same broad performance diagnosis.

#### Variable aspects
- wording of strengths and weaknesses,
- which visual details are emphasized,
- level of specificity in the explanation,
- overall confidence value (0.90 vs 0.95).

This means the VLM-only pipeline is **not deterministic in explanation phrasing**, but it appears **reasonably reproducible at the semantic judgment level**.

### 1.9 Strengths of the VLM-Only Pipeline
The experiments suggest several important strengths:

1. **Strong same-task recognition**  
   The model correctly and confidently identified that both videos correspond to the same task in both runs.

2. **Stable score estimate at high level**  
   The estimated score remained identical across repeated inference on the same video pair.

3. **Useful qualitative explanation**  
   The long explanation is rich enough to support report writing and human interpretation.

4. **Good semantic interpretation of visible performance issues**  
   The model consistently associated visible smoke with weaker execution quality and less refined technique.

5. **No need for handcrafted metrics**  
   As a first baseline, it can produce meaningful output directly from raw video.

### 1.10 Limitations of the VLM-Only Pipeline
Despite its strengths, the VLM-only pipeline has major limitations:

1. **No deterministic motion metrics**  
   It cannot provide trustworthy values for:
   - velocity difference,
   - trajectory deviation,
   - joint angle deviation,
   - DTW cost,
   - tool alignment deviation.

2. **Non-deterministic explanation phrasing**  
   The wording changes between repeated runs, even when the final decision stays similar.

3. **Confidence is model-generated, not measured**  
   Confidence values are useful indicators, but they are not grounded in deterministic evaluation logic.

4. **No structured geometric evidence**  
   The model explains what it visually infers, but it does not provide the measurable engineering evidence needed for rigorous skill evaluation.

5. **Possible over-reliance on visual salience**  
   The model strongly focused on smoke, which is useful, but this could also mean that highly visible cues may dominate the analysis.

### 1.11 Research Conclusion
The `vlm_only` pipeline functions well as a **semantic and qualitative baseline**.

It is capable of:
- recognizing the task,
- giving a stable coarse score,
- generating useful long-form explanations,
- identifying important visible weaknesses.

However, it is not sufficient on its own for a robust engineering evaluation framework, because it lacks:
- deterministic motion metrics,
- structured temporal alignment,
- measurable kinematic evidence.

Therefore, the VLM-only pipeline should be treated as:
- a **baseline for comparison**,
- a **qualitative reasoning layer**,
- and a reference point for testing whether structured perception modules such as MediaPipe improve the evaluation.

---

## 2. Benchmark Comparison Documentation

### 2.1 Purpose in the Pipeline Benchmark
This section documents the `vlm_only` pipeline in a format intended for later comparison against:
- `mediapipe_vlm`
- `mediapipe_dtw_vlm`
- `mediapipe_vjepa_vlm`
- and other stronger pipelines.

The role of `vlm_only` in the benchmark is to answer:

> What can a VLM do when it receives only the two raw videos, without any structured motion data?

### 2.2 Benchmark Role
**Pipeline role:** baseline semantic comparator

**What it contributes**
- same-task judgment
- estimated score
- qualitative strengths/weaknesses
- natural-language explanation

**What it does not contribute**
- deterministic metrics
- landmark-based comparison
- alignment cost
- motion trajectories
- velocity computation
- tool position metrics

### 2.3 Repeated-Run Stability Table

| Aspect | Run 1 | Run 2 | Stability Assessment |
|---|---|---|---|
| Overall Score | 65 | 65 | High |
| Same Task Label | true | true | High |
| Same Task Confidence | 1.00 | 1.00 | High |
| Overall Confidence | 0.90 | 0.95 | Medium-High |
| Main Weakness | smoke / slow dwelling / weak fluidity | smoke / weaker fluidity / poorer edge quality | High |
| Explanation Wording | different | different | Medium |
| Fine-detail emphasis | delicacy, pressure, smoke | curves, edges, coordination, smoke | Medium |

### 2.4 Core Comparative Observations
The repeated-run behavior suggests that `vlm_only` is likely to be strong in:
- semantic consistency,
- qualitative assessment,
- task-level comparison.

But it is likely to be weaker in:
- reproducibility of exact phrasing,
- metric-based benchmarking,
- deterministic technical scoring.

This is precisely why it is useful as a first pipeline:
it establishes what the VLM can do **before** adding structured perception or motion analysis.

### 2.5 Expected Comparison Value Against Later Pipelines
When compared later to pipelines such as `mediapipe_vlm`, the most important questions will be:

1. Does adding MediaPipe improve score reliability?
2. Does adding structured hand data reduce explanation variability?
3. Does the next pipeline produce more grounded weaknesses?
4. Can later pipelines provide measurable technical metrics that the VLM-only baseline cannot provide?
5. Does the score remain similar, or does structured motion evidence significantly change the evaluation?

### 2.6 Benchmark Conclusion
The `vlm_only` pipeline should be kept in the benchmark as the **reference semantic baseline**.

It is useful because it establishes:
- how far raw-video VLM reasoning can go by itself,
- how stable the model is across repeated runs,
- and what qualitative insight is already available without hand-crafted motion analysis.

Later pipelines should be evaluated not only on whether they improve the score, but also on whether they:
- increase reproducibility,
- provide grounded structured evidence,
- and improve the quality of explanation beyond what the VLM already achieves on its own.

---

## 3. Final Summary
The two `vlm_only` runs show that the pipeline is **strongly consistent at the high level** and **moderately variable at the language/detail level**.

This makes it a valid and valuable first baseline for the benchmark.

In summary, `vlm_only` is:
- useful for semantic understanding,
- useful for long-form explanation,
- useful for initial report material,
- but insufficient for precise, deterministic skill evaluation.

That makes it the correct first baseline to compare against the next pipeline:
**`mediapipe_vlm`**.
