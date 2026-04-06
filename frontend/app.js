const expertInput = document.getElementById("expert-video");
const learnerInput = document.getElementById("learner-video");
const expertPreview = document.getElementById("expert-preview");
const learnerPreview = document.getElementById("learner-preview");
const expertHint = document.getElementById("expert-filename-hint");
const learnerHint = document.getElementById("learner-filename-hint");
const compareBtn = document.getElementById("compare-btn");

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
  // Placeholder until pipeline is integrated
  console.log("Compare:", {
    expert: expertInput.files?.[0]?.name,
    learner: learnerInput.files?.[0]?.name,
  });
});
