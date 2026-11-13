"""FastAPI backend.   Run:  .venv\\Scripts\\python.exe -m uvicorn main:app --port 8000"""
import json
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

import agent as agent_module
import classifier
import config
import llm

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

MAX_CHARS = 20_000


class AnalyzeRequest(BaseModel):
    log: str

    @field_validator("log")
    @classmethod
    def check(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("The log text is empty.")
        if len(value) > MAX_CHARS:
            raise ValueError(f"The log text is too long ({len(value)} characters; the limit is {MAX_CHARS}).")
        return value


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agent = None
    app.state.model_error = None
    try:
        slm = classifier.SlmClassifier()  # loaded once, reused for every request
        app.state.agent = agent_module.build_agent(slm.classify)
    except Exception as error:  # noqa: BLE001 - the API must still start; /api/health reports the problem
        app.state.model_error = str(error)
        log.error("Model not loaded: %s", error)
    yield


app = FastAPI(title="Log Anomaly Classifier with Diagnostic Agent", lifespan=lifespan)
origins = [o.strip() for o in os.getenv("LOGAGENT_CORS_ORIGINS", "http://localhost:5173").split(",")]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": app.state.agent is not None, "model_error": app.state.model_error,
            "llm_configured": llm.is_configured()}


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest):
    if app.state.agent is None:
        raise HTTPException(503, detail=app.state.model_error or "The classifier model is not loaded.")
    try:
        return agent_module.run_agent(app.state.agent, request.log)
    except ValueError as error:
        raise HTTPException(422, detail=str(error)) from error
    except Exception as error:  # noqa: BLE001
        log.exception("analysis failed")
        raise HTTPException(500, detail="Analysis failed. See the server log.") from error


@app.get("/api/samples")
def samples():
    return json.loads(config.SAMPLES_FILE.read_text(encoding="utf-8")) if config.SAMPLES_FILE.exists() else []


@app.get("/api/evaluation")
def evaluation():
    path = config.RESULTS_DIR / "evaluation.json"
    if not path.exists():
        raise HTTPException(404, detail="No evaluation results yet. Run make_report.py first.")
    return json.loads(path.read_text(encoding="utf-8"))