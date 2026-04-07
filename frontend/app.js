const expertInput = document.getElementById("expert-video");
const learnerInput = document.getElementById("learner-video");
const expertPreview = document.getElementById("expert-preview");
const learnerPreview = document.getElementById("learner-preview");
const expertHint = document.getElementById("expert-filename-hint");
const learnerHint = document.getElementById("learner-filename-hint");
const compareBtn = document.getElementById("compare-btn");
const compareNote = document.getElementById("compare-note");
const resultPanel = document.getElementById("result-panel");
const resultPipeline = document.getElementById("result-pipeline");
const resultStatus = document.getElementById("result-status");
const resultScore = document.getElementById("result-score");
const resultScoreValue = document.getElementById("result-score-value");
const resultText = document.getElementById("result-text");
const resultStrengths = document.getElementById("result-strengths");
const resultWeaknesses = document.getElementById("result-weaknesses");
const resultMeta = document.getElementById("result-meta");
const resultWarnings = document.getElementById("result-warnings");

const COMPARE_ENDPOINT = "http://127.0.0.1:8000/compare-upload";

let expertUrl = null;
let learnerUrl = null;
let isComparing = false;

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
    expertInput.files?.length > 0 && learnerInput.files?.length > 0 && !isComparing;
  compareBtn.disabled = !ready;
  compareNote.textContent = isComparing
    ? "Uploading videos and running the comparison..."
    : ready
      ? "Both videos are ready."
      : "Select both videos, then run the comparison.";
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

function renderResult(result) {
  resultPanel.hidden = false;
  resultPipeline.textContent = result?.run?.pipeline_name || "unknown";
  resultStatus.textContent = "Comparison finished.";

  if (typeof result?.overall_score === "number") {
    resultScore.hidden = false;
    resultScoreValue.textContent = `${Math.round(result.overall_score)}/100`;
  } else {
    resultScore.hidden = true;
    resultScoreValue.textContent = "--";
  }

  resultText.textContent =
    result?.explanation?.text || "The comparison completed without a detailed explanation.";
  resetList(resultStrengths, result?.explanation?.strengths, "No strengths were returned.");
  resetList(resultWeaknesses, result?.explanation?.weaknesses, "No weaknesses were returned.");
  resetList(resultWarnings, result?.warnings, "No warnings.");
  renderMeta(result);
}

function renderError(message) {
  resultPanel.hidden = false;
  resultPipeline.textContent = "error";
  resultStatus.textContent = "Comparison failed.";
  resultScore.hidden = true;
  resultScoreValue.textContent = "--";
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
  formData.append("pipeline_name", "vlm_only");

  isComparing = true;
  resultPanel.hidden = false;
  resultPipeline.textContent = "vlm_only";
  resultStatus.textContent = "Running comparison...";
  resultScore.hidden = true;
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
        : "Unable to reach the backend. Make sure the API is running on http://127.0.0.1:8000.";
    renderError(message);
  } finally {
    isComparing = false;
    updateCompareButton();
  }
});

updateCompareButton();
