"""Backend entrypoint for the AugMentor pipeline comparison demo."""

import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas.pipeline_schema import PipelineInput, PipelineName
from backend.schemas.result_schema import PipelineResult
from backend.services.pipeline_dispatcher import dispatch_pipeline

app = FastAPI(
    title="AugMentor Pipeline Demo",
    description="Benchmark demo API for comparing expert vs learner video pipelines.",
)

UPLOAD_DIR = Path(__file__).resolve().parent / "outputs" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/pipelines")
def list_pipelines():
    return {"pipelines": [p.value for p in PipelineName]}


@app.post("/compare", response_model=PipelineResult)
def compare(pipeline_input: PipelineInput):
    try:
        return dispatch_pipeline(pipeline_input)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _save_upload(upload: UploadFile, prefix: str) -> Path:
    suffix = Path(upload.filename or "").suffix or ".mp4"
    target = UPLOAD_DIR / f"{prefix}-{uuid4().hex}{suffix}"
    with target.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return target


@app.post("/compare-upload", response_model=PipelineResult)
def compare_upload(
    expert_video: UploadFile = File(...),
    learner_video: UploadFile = File(...),
    pipeline_name: PipelineName = Form(PipelineName.VLM_ONLY),
):
    saved_paths: list[Path] = []
    try:
        expert_path = _save_upload(expert_video, "expert")
        learner_path = _save_upload(learner_video, "learner")
        saved_paths.extend([expert_path, learner_path])

        pipeline_input = PipelineInput(
            pipeline_name=pipeline_name,
            expert_video_path=str(expert_path),
            learner_video_path=str(learner_path),
        )
        return dispatch_pipeline(pipeline_input)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        for upload in (expert_video, learner_video):
            upload.file.close()
        for path in saved_paths:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
