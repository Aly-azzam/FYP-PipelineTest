const expertInput = document.getElementById("expert-video");
const learnerInput = document.getElementById("learner-video");
const expertPreview = document.getElementById("expert-preview");
const learnerPreview = document.getElementById("learner-preview");
const expertHint = document.getElementById("expert-filename-hint");
const learnerHint = document.getElementById("learner-filename-hint");
const compareBtn = document.getElementById("compare-btn");
const resultEl = document.getElementById("result");

let expertUrl = null;
let learnerUrl = null;

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
    expertInput.files?.length > 0 && learnerInput.files?.length > 0;
  compareBtn.disabled = !ready;
}

function showResult(text) {
  if (!resultEl) return;
  resultEl.hidden = false;
  resultEl.textContent = text;
}

function setBusy(isBusy) {
  compareBtn.disabled = isBusy || !(expertInput.files?.length > 0 && learnerInput.files?.length > 0);
  compareBtn.textContent = isBusy ? "Running..." : "Run comparison";
}

function setProcessedPreview(videoEl, path) {
  if (!path) return;
  videoEl.src = path;
  videoEl.hidden = false;
  videoEl.load();
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

compareBtn.addEventListener("click", () => {
  const expertFile = expertInput.files?.[0];
  const learnerFile = learnerInput.files?.[0];
  if (!expertFile || !learnerFile) return;

  setBusy(true);
  showResult("Uploading videos and running MediaPipe (wrist only)...");

  const form = new FormData();
  form.append("pipeline_name", "mediapipe_vlm");
  form.append("expert_video", expertFile, expertFile.name);
  form.append("learner_video", learnerFile, learnerFile.name);

  const endpoints = [
    "http://127.0.0.1:8010/compare_upload",
    "http://127.0.0.1:8000/compare_upload",
  ];

  const tryEndpoint = async (index) => {
    if (index >= endpoints.length) {
      throw new Error(
        "Could not reach backend. Start API on port 8010 or 8000 and try again."
      );
    }

    try {
      const res = await fetch(endpoints[index], {
        method: "POST",
        body: form,
      });
      const text = await res.text();
      if (!res.ok) throw new Error(text);
      return text;
    } catch {
      return tryEndpoint(index + 1);
    }
  };

  tryEndpoint(0)
    .then((text) => {
      try {
        const json = JSON.parse(text);
        setProcessedPreview(expertPreview, json?.expert_video?.path);
        setProcessedPreview(learnerPreview, json?.learner_video?.path);
        showResult(JSON.stringify(json, null, 2));
      } catch {
        showResult(text);
      }
    })
    .catch((err) => {
      showResult(`Error: ${String(err?.message || err)}`);
    })
    .finally(() => {
      setBusy(false);
    });
});
