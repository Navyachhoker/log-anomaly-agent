"""Agent routing, API and LLM-client tests, using fakes (no model, no network, no API key)."""
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

import agent
import classifier
import llm
import main


def fake_classifier(label, p_anomaly=0.9):
    calls = []

    def classify(log):
        calls.append(log)
        time.sleep(0.02)  # the claimed latency below is real
        return {"label": label, "confidence": p_anomaly if label == "Anomaly" else 1 - p_anomaly, "p_anomaly": p_anomaly,
                "threshold": 0.5, "snippet": "INFO a: b", "truncated": False, "latency_s": 0.02}
    classify.calls = calls
    return classify


def fake_diagnose(result):
    calls = []

    def diagnose(snippet):
        calls.append(snippet)
        return result
    diagnose.calls = calls
    return diagnose


GOOD = (agent.Diagnosis(cause="DataNode lost a block.", suggested_fix=["Check DataNode status"]),
        {"calls": 1, "input_tokens": 100, "output_tokens": 30, "latency_s": 0.5}, None)


def test_normal_path_never_calls_the_llm():
    diagnose = fake_diagnose(GOOD)
    report = agent.run_agent(agent.build_agent(fake_classifier("Normal", 0.02), diagnose), "some log")
    assert report["label"] == "Normal" and report["cause"] is None and report["latency_ms"]["diagnose"] is None
    assert diagnose.calls == []
    assert "Advisory" in report["advisory"]


def test_anomaly_path_calls_the_llm_once_and_returns_both_fields():
    diagnose = fake_diagnose(GOOD)
    report = agent.run_agent(agent.build_agent(fake_classifier("Anomaly"), diagnose), "some log")
    assert len(diagnose.calls) == 1
    assert report["cause"] == "DataNode lost a block." and report["suggested_fix"] == ["Check DataNode status"]
    assert report["diagnosis_error"] is None and report["llm_usage"]["input_tokens"] == 100
    assert report["latency_ms"]["total"] >= report["latency_ms"]["classify"]


def test_llm_failure_still_returns_a_valid_report():
    report = agent.run_agent(agent.build_agent(fake_classifier("Anomaly"), fake_diagnose((None, None, "timeout"))), "log")
    assert report["label"] == "Anomaly" and report["cause"] is None and report["diagnosis_error"] == "timeout"


def test_request_diagnosis_without_configuration(monkeypatch):
    for name in ("LOGAGENT_LLM_PROVIDER", "LOGAGENT_LLM_API_KEY", "LOGAGENT_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    diagnosis, usage, error = agent.request_diagnosis("x")
    assert diagnosis is None and "not configured" in error


def _llm_env(monkeypatch):
    monkeypatch.setenv("LOGAGENT_LLM_PROVIDER", "openai")
    monkeypatch.setenv("LOGAGENT_LLM_API_KEY", "test-key")
    monkeypatch.setenv("LOGAGENT_LLM_MODEL", "test-model")


def test_request_diagnosis_retries_once_on_invalid_json_then_succeeds(monkeypatch):
    _llm_env(monkeypatch)
    replies = iter(["not json at all", '```json\n{"cause": "Disk full", "suggested_fix": ["Free space"]}\n```'])
    monkeypatch.setattr(llm, "chat", lambda *a, **k: llm.LLMResponse(next(replies), 10, 5, 0.1))
    diagnosis, usage, error = agent.request_diagnosis("x")
    assert error is None and diagnosis.cause == "Disk full" and usage["calls"] == 2 and usage["input_tokens"] == 20


def test_request_diagnosis_gives_up_after_two_invalid_answers_and_on_llm_errors(monkeypatch):
    _llm_env(monkeypatch)
    monkeypatch.setattr(llm, "chat", lambda *a, **k: llm.LLMResponse('{"cause": ""}', 1, 1, 0.1))
    diagnosis, _, error = agent.request_diagnosis("x")
    assert diagnosis is None and "invalid" in error

    def boom(*a, **k):
        raise llm.LLMError("HTTP 401")
    monkeypatch.setattr(llm, "chat", boom)
    assert agent.request_diagnosis("x")[2] == "HTTP 401"


def test_llm_chat_parses_openai_style_response_and_hides_nothing_secret(monkeypatch):
    _llm_env(monkeypatch)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={
        "choices": [{"message": {"content": '{"a": 1}'}}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}))
    real_client = httpx.Client
    monkeypatch.setattr(llm.httpx, "Client", lambda **kw: real_client(transport=transport, **kw))
    reply = llm.chat("sys", "user", json_mode=True)
    assert (reply.text, reply.input_tokens, reply.output_tokens) == ('{"a": 1}', 7, 3)


def test_llm_chat_http_error_becomes_llmerror(monkeypatch):
    _llm_env(monkeypatch)
    transport = httpx.MockTransport(lambda request: httpx.Response(401, text="bad key"))
    real_client = httpx.Client
    monkeypatch.setattr(llm.httpx, "Client", lambda **kw: real_client(transport=transport, **kw))
    with pytest.raises(llm.LLMError, match="401"):
        llm.chat("s", "u")


def test_parse_json_and_cost(monkeypatch):
    assert llm.parse_json('Here: {"x": 2} thanks') == {"x": 2}
    with pytest.raises(ValueError):
        llm.parse_json("no braces")
    monkeypatch.setenv("LOGAGENT_PRICE_IN_PER_MTOK", "2")
    monkeypatch.setenv("LOGAGENT_PRICE_OUT_PER_MTOK", "10")
    assert llm.cost_usd(1_000_000, 500_000) == pytest.approx(7.0)
    monkeypatch.delenv("LOGAGENT_PRICE_IN_PER_MTOK")
    assert llm.cost_usd(1, 1) is None


def test_anomaly_probability_is_a_two_way_softmax():
    assert classifier.anomaly_probability(0.0, 0.0) == pytest.approx(0.5)
    assert classifier.anomaly_probability(0.0, 10.0) > 0.999
    assert classifier.anomaly_probability(10.0, 0.0) < 0.001
    assert classifier.anomaly_probability(3.0, 1.0) + classifier.anomaly_probability(1.0, 3.0) == pytest.approx(1.0)
    assert classifier.anomaly_probability(5000.0, 0.0) == 0.0


# ---------------------------------------------------------------- API
@pytest.fixture
def client():
    main.app.state.agent = agent.build_agent(fake_classifier("Anomaly"), fake_diagnose(GOOD))
    main.app.state.model_error = None
    return TestClient(main.app)  # no `with`: the real model is not loaded


def test_analyze_returns_a_report(client):
    response = client.post("/api/analyze", json={"log": "081109 203615 148 INFO dfs.DataNode: hello blk_1"})
    assert response.status_code == 200 and response.json()["label"] == "Anomaly" and response.json()["suggested_fix"]


def test_analyze_validates_input(client):
    assert client.post("/api/analyze", json={"log": "   "}).status_code == 422
    assert client.post("/api/analyze", json={"log": "x" * 20_001}).status_code == 422
    assert client.post("/api/analyze", json={}).status_code == 422


def test_analyze_without_a_model_gives_503(client):
    main.app.state.agent = None
    main.app.state.model_error = "No model found in X"
    response = client.post("/api/analyze", json={"log": "abc"})
    assert response.status_code == 503 and "No model found" in response.json()["detail"]
    assert client.get("/api/health").json()["model_loaded"] is False


def test_health_samples_and_evaluation_when_files_are_missing(client, tmp_path, monkeypatch):
    monkeypatch.setattr(main.config, "SAMPLES_FILE", tmp_path / "none.json")
    monkeypatch.setattr(main.config, "RESULTS_DIR", tmp_path)
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/api/samples").json() == []
    assert client.get("/api/evaluation").status_code == 404
    (tmp_path / "evaluation.json").write_text(json.dumps({"models": []}))
    assert client.get("/api/evaluation").json() == {"models": []}