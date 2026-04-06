"""Backend entrypoint for the AugMentor pipeline comparison demo."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas.pipeline_schema import PipelineInput, PipelineName
from backend.schemas.result_schema import PipelineResult
from backend.services.pipeline_dispatcher import dispatch_pipeline

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
