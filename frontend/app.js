const expertInput = document.getElementById("expert-video");
const learnerInput = document.getElementById("learner-video");
const expertPreview = document.getElementById("expert-preview");
const learnerPreview = document.getElementById("learner-preview");
const expertHint = document.getElementById("expert-filename-hint");
const learnerHint = document.getElementById("learner-filename-hint");
const pipelineSelect = document.getElementById("pipeline-select");
const compareBtn = document.getElementById("compare-btn");
const compareNote = document.getElementById("compare-note");
const resultPanel = document.getElementById("result-panel");
const resultPipeline = document.getElementById("result-pipeline");
const resultStatus = document.getElementById("result-status");
const resultScore = document.getElementById("result-score");
const resultScoreLabel = document.getElementById("result-score-label");
const resultScoreValue = document.getElementById("result-score-value");
const resultMetrics = document.getElementById("result-metrics");
const resultSemanticSimilarity = document.getElementById("result-semantic-similarity");
const resultSegmentMetrics = document.getElementById("result-segment-metrics");
const resultMeanSegmentSimilarity = document.getElementById("result-mean-segment-similarity");
const resultMinSegmentSimilarity = document.getElementById("result-min-segment-similarity");
const resultSegmentConsistency = document.getElementById("result-segment-consistency");
const resultDebugMetrics = document.getElementById("result-debug-metrics");
const resultSegmentVariance = document.getElementById("result-segment-variance");
const resultWeightedSimilarity = document.getElementById("result-weighted-similarity");
const resultWorstSegmentIndex = document.getElementById("result-worst-segment-index");
const resultFallback = document.getElementById("result-fallback");
const resultText = document.getElementById("result-text");
const resultStrengths = document.getElementById("result-strengths");
const resultWeaknesses = document.getElementById("result-weaknesses");
const resultMeta = document.getElementById("result-meta");
const resultWarnings = document.getElementById("result-warnings");

// Must match uvicorn: `python -m uvicorn backend.app:app --host 127.0.0.1 --port 8001`
// If pipelines are missing (e.g. no vjepa_only), you are usually hitting a stale server on another port.
const API_BASE = "http://127.0.0.1:8001";
const PIPELINES_ENDPOINT = `${API_BASE}/pipelines`;
const COMPARE_ENDPOINT = `${API_BASE}/compare-upload`;

let expertUrl = null;
let learnerUrl = null;
let isComparing = false;
let selectedPipeline = "";

function revoke(url) {
  if (url) URL.revokeObjectURL(url);
}

function setPreview(input, videoEl, hintEl, which) {
  const file = input.files?.[0];
  revoke(which === "expert" ? expertUrl : learnerUrl);
  if (!file) {
    hintEl.textContent = "No file selected";
    videoEl.hidden = true;
    videoEl.removeAttribute("src");
    if (which === "expert") expertUrl = null;
    else learnerUrl = null;
    updateCompareButton();
    return;
  }

  hintEl.textContent = file.name;
  const url = URL.createObjectURL(file);
  if (which === "expert") expertUrl = url;
  else learnerUrl = url;
  videoEl.src = url;
  videoEl.hidden = false;
  updateCompareButton();
}

function updateCompareButton() {
  const ready =
    expertInput.files?.length > 0 &&
    learnerInput.files?.length > 0 &&
    Boolean(selectedPipeline) &&
    !isComparing;
  compareBtn.disabled = !ready;
  if (isComparing) {
    compareNote.textContent = "Uploading videos and running the comparison...";
    return;
  }
  if (!selectedPipeline) {
    compareNote.textContent = "Select a pipeline, then choose both videos.";
    return;
  }
  compareNote.textContent = ready
    ? "Pipeline and videos are ready."
    : "Select both videos, then run the comparison.";
}

function setPipelineOptions(pipelines) {
  pipelineSelect.innerHTML = "";
  pipelines.forEach((pipeline) => {
    const option = document.createElement("option");
    option.value = pipeline;
    option.textContent = pipeline;
    pipelineSelect.appendChild(option);
  });
}

async function loadPipelines() {
  pipelineSelect.disabled = true;
  pipelineSelect.innerHTML = "";
  const loadingOption = document.createElement("option");
  loadingOption.value = "";
  loadingOption.textContent = "Loading pipelines...";
  pipelineSelect.appendChild(loadingOption);
  selectedPipeline = "";
  updateCompareButton();

  try {
    const response = await fetch(PIPELINES_ENDPOINT);
    const payload = await response.json().catch(() => ({}));

    if (!response.ok || !Array.isArray(payload?.pipelines)) {
      throw new Error("Failed to load pipelines from backend.");
    }

    const pipelines = payload.pipelines
      .filter((item) => typeof item === "string")
      .map((item) => item.trim())
      .filter(Boolean);

    if (pipelines.length === 0) {
      throw new Error("No pipelines are available.");
    }

    setPipelineOptions(pipelines);
    selectedPipeline = pipelines.includes("vlm_only") ? "vlm_only" : pipelines[0];
    pipelineSelect.value = selectedPipeline;
    pipelineSelect.disabled = false;
  } catch (error) {
    pipelineSelect.innerHTML = "";
    const unavailableOption = document.createElement("option");
    unavailableOption.value = "";
    unavailableOption.textContent = "Pipeline list unavailable";
    pipelineSelect.appendChild(unavailableOption);
    pipelineSelect.disabled = true;
    selectedPipeline = "";
    renderError(
      error instanceof Error
        ? error.message
        : "Unable to load pipeline list from backend."
    );
  } finally {
    updateCompareButton();
  }
}

function resetList(listEl, items, emptyText) {
  listEl.innerHTML = "";
  const values = Array.isArray(items) ? items.filter(Boolean) : [];
  if (values.length === 0) {
    const li = document.createElement("li");
    li.textContent = emptyText;
    listEl.appendChild(li);
    return;
  }

  values.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    listEl.appendChild(li);
  });
}

function renderMeta(result) {
  const parts = [];
  if (result?.run?.run_id) parts.push(`Run ID: ${result.run.run_id}`);
  if (result?.run?.processing_time_sec != null) {
    parts.push(`Processing time: ${result.run.processing_time_sec}s`);
  }
  if (result?.expert_video?.filename) {
    parts.push(`Expert: ${result.expert_video.filename}`);
  }
  if (result?.learner_video?.filename) {
    parts.push(`Learner: ${result.learner_video.filename}`);
  }
  resultMeta.textContent = parts.join(" | ");
}

function formatMetricValue(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(4)
    : "--";
}

function renderResult(result) {
  resultPanel.hidden = false;
  const pipelineName = result?.run?.pipeline_name || "unknown";
  resultPipeline.textContent = pipelineName;
  resultStatus.textContent = "Comparison finished.";
  const warnings = Array.isArray(result?.warnings) ? result.warnings : [];
  const isVjepaOnly = pipelineName === "vjepa_only";
  const extraMetrics = result?.metrics?.extra;
  resultScoreLabel.textContent = isVjepaOnly
    ? "Global semantic closeness"
    : "Overall score";

  if (typeof result?.overall_score === "number") {
    resultScore.hidden = false;
    resultScoreValue.textContent = `${Math.round(result.overall_score)}/100`;
  } else {
    resultScore.hidden = true;
    resultScoreValue.textContent = "--";
  }

  if (isVjepaOnly) {
    resultMetrics.hidden = false;
    const sim = result?.metrics?.semantic_similarity;
    resultSemanticSimilarity.textContent = formatMetricValue(sim);

    const meanSegmentSimilarity = extraMetrics?.mean_segment_similarity;
    const minSegmentSimilarity = extraMetrics?.min_segment_similarity;
    const segmentConsistency = extraMetrics?.segment_consistency;
    const segmentVariance = extraMetrics?.segment_similarity_variance;
    const weightedSimilarity = extraMetrics?.weighted_similarity;
    const worstSegmentIndex = extraMetrics?.worst_segment_index;
    const hasSegmentMetrics =
      typeof meanSegmentSimilarity === "number" ||
      typeof minSegmentSimilarity === "number" ||
      typeof segmentConsistency === "number";
    const hasDebugMetrics =
      typeof segmentVariance === "number" ||
      typeof weightedSimilarity === "number" ||
      Number.isInteger(worstSegmentIndex);

    resultSegmentMetrics.hidden = !hasSegmentMetrics;
    resultMeanSegmentSimilarity.textContent = formatMetricValue(meanSegmentSimilarity);
    resultMinSegmentSimilarity.textContent = formatMetricValue(minSegmentSimilarity);
    resultSegmentConsistency.textContent = formatMetricValue(segmentConsistency);
    resultDebugMetrics.hidden = !hasDebugMetrics;
    resultSegmentVariance.textContent = formatMetricValue(segmentVariance);
    resultWeightedSimilarity.textContent = formatMetricValue(weightedSimilarity);
    resultWorstSegmentIndex.textContent = Number.isInteger(worstSegmentIndex)
      ? String(worstSegmentIndex)
      : "--";
  } else {
    resultMetrics.hidden = true;
    resultSemanticSimilarity.textContent = "--";
    resultSegmentMetrics.hidden = true;
    resultMeanSegmentSimilarity.textContent = "--";
    resultMinSegmentSimilarity.textContent = "--";
    resultSegmentConsistency.textContent = "--";
    resultDebugMetrics.hidden = true;
    resultSegmentVariance.textContent = "--";
    resultWeightedSimilarity.textContent = "--";
    resultWorstSegmentIndex.textContent = "--";
  }

  resultFallback.hidden = !warnings.includes("temporary_embedding_fallback");
  resultText.textContent =
    result?.explanation?.text || "The comparison completed without a detailed explanation.";
  resetList(resultStrengths, result?.explanation?.strengths, "No strengths were returned.");
  resetList(resultWeaknesses, result?.explanation?.weaknesses, "No weaknesses were returned.");
  resetList(resultWarnings, warnings, "No warnings.");
  renderMeta(result);
}

function renderError(message) {
  resultPanel.hidden = false;
  resultPipeline.textContent = "error";
  resultStatus.textContent = "Comparison failed.";
  resultScoreLabel.textContent = "Overall score";
  resultScore.hidden = true;
  resultScoreValue.textContent = "--";
  resultMetrics.hidden = true;
  resultSemanticSimilarity.textContent = "--";
  resultSegmentMetrics.hidden = true;
  resultMeanSegmentSimilarity.textContent = "--";
  resultMinSegmentSimilarity.textContent = "--";
  resultSegmentConsistency.textContent = "--";
  resultDebugMetrics.hidden = true;
  resultSegmentVariance.textContent = "--";
  resultWeightedSimilarity.textContent = "--";
  resultWorstSegmentIndex.textContent = "--";
  resultFallback.hidden = true;
  resultText.textContent = message;
  resetList(resultStrengths, [], "No strengths available.");
  resetList(resultWeaknesses, [], "No weaknesses available.");
  resetList(resultWarnings, [], "No warnings.");
  resultMeta.textContent = "";
}

expertInput.addEventListener("change", () =>
  setPreview(expertInput, expertPreview, expertHint, "expert")
);
learnerInput.addEventListener("change", () =>
  setPreview(learnerInput, learnerPreview, learnerHint, "learner")
);
pipelineSelect.addEventListener("change", () => {
  selectedPipeline = pipelineSelect.value;
  updateCompareButton();
});

function wireDropZone(label, input) {
  label.addEventListener("dragover", (e) => {
    e.preventDefault();
    label.style.borderColor = "var(--accent)";
  });
  label.addEventListener("dragleave", () => {
    label.style.borderColor = "";
  });
  label.addEventListener("drop", (e) => {
    e.preventDefault();
    label.style.borderColor = "";
    const file = e.dataTransfer.files?.[0];
    if (!file || !file.type.startsWith("video/")) return;
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });
}

document.querySelectorAll(".drop-zone").forEach((label) => {
  const input = label.querySelector('input[type="file"]');
  if (input) wireDropZone(label, input);
});

compareBtn.addEventListener("click", async () => {
  if (compareBtn.disabled) return;

  const expertFile = expertInput.files?.[0];
  const learnerFile = learnerInput.files?.[0];
  if (!expertFile || !learnerFile) {
    renderError("Please select both videos before running the comparison.");
    return;
  }

  const formData = new FormData();
  formData.append("expert_video", expertFile);
  formData.append("learner_video", learnerFile);
  formData.append("pipeline_name", selectedPipeline);

  isComparing = true;
  resultPanel.hidden = false;
  resultPipeline.textContent = selectedPipeline;
  resultStatus.textContent = "Running comparison...";
  resultScoreLabel.textContent =
    selectedPipeline === "vjepa_only"
      ? "Global semantic closeness"
      : "Overall score";
  resultScore.hidden = true;
  resultMetrics.hidden = true;
  resultSemanticSimilarity.textContent = "--";
  resultSegmentMetrics.hidden = true;
  resultMeanSegmentSimilarity.textContent = "--";
  resultMinSegmentSimilarity.textContent = "--";
  resultSegmentConsistency.textContent = "--";
  resultDebugMetrics.hidden = true;
  resultSegmentVariance.textContent = "--";
  resultWeightedSimilarity.textContent = "--";
  resultWorstSegmentIndex.textContent = "--";
  resultFallback.hidden = true;
  resultText.textContent = "Uploading the selected videos and waiting for the backend response.";
  resultStrengths.innerHTML = "";
  resultWeaknesses.innerHTML = "";
  resultWarnings.innerHTML = "";
  resultMeta.textContent = "";
  updateCompareButton();

  try {
    const response = await fetch(COMPARE_ENDPOINT, {
      method: "POST",
      body: formData,
    });
    const payload = await response.json().catch(() => ({}));

    if (!response.ok) {
      const detail =
        typeof payload?.detail === "string"
          ? payload.detail
          : "The backend could not complete the comparison.";
      throw new Error(detail);
    }

    renderResult(payload);
  } catch (error) {
    const message =
      error instanceof Error
        ? error.message
        : "Unable to reach the backend. Make sure the API is running on the same host/port as API_BASE in app.js (default http://127.0.0.1:8001).";
    renderError(message);
  } finally {
    isComparing = false;
    updateCompareButton();
  }
});

loadPipelines();
updateCompareButton();
