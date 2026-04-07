"""Backend entrypoint for the AugMentor pipeline comparison demo."""

import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi import File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.schemas.pipeline_schema import PipelineInput, PipelineName
from backend.schemas.result_schema import PipelineResult
from backend.services.pipeline_dispatcher import dispatch_pipeline

BACKEND_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="AugMentor Pipeline Demo",
    description="Benchmark demo API for comparing expert vs learner video pipelines.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media", StaticFiles(directory=BACKEND_DIR), name="media")


def _to_public_media_url(path_value: str | None, request: Request) -> str | None:
    if not path_value:
        return path_value

    path_obj = Path(path_value).resolve()
    try:
        relative_path = path_obj.relative_to(BACKEND_DIR)
    except ValueError:
        return path_value

    return f"{str(request.base_url).rstrip('/')}/media/{relative_path.as_posix()}"


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


@app.post("/compare_upload", response_model=PipelineResult)
def compare_upload(
    request: Request,
    pipeline_name: str = Form(...),
    expert_video: UploadFile = File(...),
    learner_video: UploadFile = File(...),
):
    try:
        parsed_name = PipelineName(pipeline_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    upload_dir = BACKEND_DIR / ".tmp_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    expert_path = upload_dir / f"{uuid.uuid4().hex}_{Path(expert_video.filename or 'expert').name}"
    learner_path = upload_dir / f"{uuid.uuid4().hex}_{Path(learner_video.filename or 'learner').name}"

    with expert_path.open("wb") as f:
        shutil.copyfileobj(expert_video.file, f)
    with learner_path.open("wb") as f:
        shutil.copyfileobj(learner_video.file, f)

    try:
        result = dispatch_pipeline(
            PipelineInput(
                pipeline_name=parsed_name,
                expert_video_path=str(expert_path),
                learner_video_path=str(learner_path),
            )
        )
        result.expert_video.path = _to_public_media_url(result.expert_video.path, request)
        result.learner_video.path = _to_public_media_url(result.learner_video.path, request)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
