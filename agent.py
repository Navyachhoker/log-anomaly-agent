"""LangGraph diagnostic agent: classify -> (only if Anomaly) diagnose -> report.

Advisory only: the graph has no tools and no code path that executes anything. The LLM is asked
once and returns both the likely cause and suggested fixes as JSON.
"""
import time
from datetime import datetime, timezone
from typing import Callable, List, Literal, Optional, Tuple, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

import llm

ADVISORY = "Advisory only. These suggestions are for a human operator to review; nothing was executed."


class Diagnosis(BaseModel):
    cause: str = Field(min_length=1)
    suggested_fix: List[str] = Field(min_length=1)


class Report(BaseModel):
    label: Literal["Normal", "Anomaly"]
    confidence: float
    p_anomaly: float
    threshold: float
    cause: Optional[str] = None
    suggested_fix: Optional[List[str]] = None
    diagnosis_error: Optional[str] = None
    timestamp: str
    latency_ms: dict
    llm_usage: Optional[dict] = None
    snippet: str
    truncated: bool = False
    advisory: str = ADVISORY


SYSTEM_PROMPT = (
    "You are a senior site-reliability engineer who knows Hadoop HDFS. You only analyse and advise; "
    "you never run commands. Answer with one JSON object and nothing else."
)

USER_TEMPLATE = (
    "A classifier flagged this HDFS log snippet (all log lines of one block, with timestamps, block IDs "
    "and addresses replaced by placeholders) as ANOMALOUS:\n\n{snippet}\n\n"
    "Reply with JSON of exactly this shape:\n"
    '{{"cause": "1-3 sentences on the most likely root cause, based only on these lines", '
    '"suggested_fix": ["1 to 4 short steps for a human operator to review"]}}\n'
    "If the evidence is weak, say so in the cause."
)


def request_diagnosis(snippet: str) -> Tuple[Optional[Diagnosis], Optional[dict], Optional[str]]:
    """One structured LLM call (one retry only if the JSON is invalid). Never raises.
    Returns (diagnosis, usage, error)."""
    if not llm.is_configured():
        return None, None, "Diagnostic LLM is not configured (fill LOGAGENT_LLM_* in .env)."
    usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "latency_s": 0.0}
    error = None
    for _ in range(2):
        try:
            reply = llm.chat(SYSTEM_PROMPT, USER_TEMPLATE.format(snippet=snippet), json_mode=True)
        except llm.LLMError as exc:
            return None, usage, str(exc)
        usage["calls"] += 1
        usage["input_tokens"] += reply.input_tokens
        usage["output_tokens"] += reply.output_tokens
        usage["latency_s"] += reply.latency_s
        try:
            return Diagnosis.model_validate(llm.parse_json(reply.text)), usage, None
        except ValueError as exc:  # includes JSON errors and pydantic validation errors
            error = f"The LLM returned an invalid answer: {exc}"
    return None, usage, error


class AgentState(TypedDict, total=False):
    log: str
    started: float
    classification: dict
    diagnosis: Optional[Diagnosis]
    usage: Optional[dict]
    diagnosis_error: Optional[str]
    diagnose_seconds: Optional[float]
    report: dict


def build_agent(classify_fn: Callable[[str], dict], diagnose_fn: Callable[[str], tuple] = request_diagnosis):
    """Compile the graph. The two functions are passed in so tests can use fakes."""

    def classify_node(state: AgentState) -> AgentState:
        return {"started": time.perf_counter(), "classification": classify_fn(state["log"])}

    def route(state: AgentState) -> str:
        return "diagnose" if state["classification"]["label"] == "Anomaly" else "report"

    def diagnose_node(state: AgentState) -> AgentState:
        start = time.perf_counter()
        diagnosis, usage, error = diagnose_fn(state["classification"]["snippet"])
        return {"diagnosis": diagnosis, "usage": usage, "diagnosis_error": error,
                "diagnose_seconds": time.perf_counter() - start}

    def report_node(state: AgentState) -> AgentState:
        c = state["classification"]
        diagnosis = state.get("diagnosis")
        diagnose_seconds = state.get("diagnose_seconds")
        report = Report(
            label=c["label"], confidence=c["confidence"], p_anomaly=c["p_anomaly"], threshold=c["threshold"],
            cause=diagnosis.cause if diagnosis else None,
            suggested_fix=diagnosis.suggested_fix if diagnosis else None,
            diagnosis_error=state.get("diagnosis_error"),
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            latency_ms={
                "classify": round(c["latency_s"] * 1000, 1),
                "diagnose": round(diagnose_seconds * 1000, 1) if diagnose_seconds is not None else None,
                "total": round((time.perf_counter() - state["started"]) * 1000, 1),
            },
            llm_usage=state.get("usage"), snippet=c["snippet"], truncated=c.get("truncated", False),
        )
        return {"report": report.model_dump()}

    graph = StateGraph(AgentState)
    graph.add_node("classify", classify_node)
    graph.add_node("diagnose", diagnose_node)
    graph.add_node("report", report_node)
    graph.set_entry_point("classify")
    graph.add_conditional_edges("classify", route, {"diagnose": "diagnose", "report": "report"})
    graph.add_edge("diagnose", "report")
    graph.add_edge("report", END)
    return graph.compile()


def run_agent(agent, log_text: str) -> dict:
    return agent.invoke({"log": log_text})["report"]