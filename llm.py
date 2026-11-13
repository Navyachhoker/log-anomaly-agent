"""Small client for the API-based LLM (used for diagnosis and for the zero-shot baseline).

The provider is chosen with environment variables (see .env.example), so switching provider means
editing .env, not code. Supported: openai (and OpenAI-compatible services via LOGAGENT_LLM_BASE_URL),
anthropic, gemini. Uses plain HTTP calls through httpx.
"""
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import httpx

import config  # noqa: F401  (loads .env)

RETRY_STATUS = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """Any failure talking to the LLM API (missing config, network, HTTP error, bad response)."""


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float


def settings() -> dict:
    return {
        "provider": os.getenv("LOGAGENT_LLM_PROVIDER", "").strip().lower(),
        "api_key": os.getenv("LOGAGENT_LLM_API_KEY", "").strip(),
        "model": os.getenv("LOGAGENT_LLM_MODEL", "").strip(),
        "base_url": os.getenv("LOGAGENT_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
    }


def is_configured() -> bool:
    s = settings()
    return bool(s["provider"] and s["api_key"] and s["model"])


def prices_per_million_tokens() -> Tuple[Optional[float], Optional[float]]:
    """(input, output) USD per 1M tokens from .env, or None when not set."""
    def read(name):
        value = os.getenv(name, "").strip()
        return float(value) if value else None
    return read("LOGAGENT_PRICE_IN_PER_MTOK"), read("LOGAGENT_PRICE_OUT_PER_MTOK")


def cost_usd(input_tokens: int, output_tokens: int) -> Optional[float]:
    price_in, price_out = prices_per_million_tokens()
    if price_in is None or price_out is None:
        return None
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def parse_json(text: str) -> dict:
    """Extract the first JSON object from model output (tolerates ``` fences and extra prose)."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found in the model output")
    return json.loads(cleaned[start : end + 1])


def _call_openai(s, system, user, json_mode, client):
    body = {"model": s["model"], "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    r = client.post(f"{s['base_url']}/chat/completions", json=body, headers={"Authorization": f"Bearer {s['api_key']}"})
    return r, lambda d: (d["choices"][0]["message"]["content"] or "", d["usage"]["prompt_tokens"], d["usage"]["completion_tokens"])


def _call_anthropic(s, system, user, json_mode, client):
    body = {"model": s["model"], "max_tokens": 1024, "system": system, "messages": [{"role": "user", "content": user}]}
    headers = {"x-api-key": s["api_key"], "anthropic-version": "2023-06-01"}
    r = client.post("https://api.anthropic.com/v1/messages", json=body, headers=headers)
    return r, lambda d: ("".join(b.get("text", "") for b in d["content"] if b.get("type") == "text"),
                         d["usage"]["input_tokens"], d["usage"]["output_tokens"])


def _call_gemini(s, system, user, json_mode, client):
    body = {"systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}]}
    if json_mode:
        body["generationConfig"] = {"responseMimeType": "application/json"}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{s['model']}:generateContent"
    r = client.post(url, json=body, headers={"x-goog-api-key": s["api_key"]})

    def extract(d):
        parts = d["candidates"][0]["content"]["parts"]
        usage = d.get("usageMetadata", {})
        out_tokens = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
        return "".join(p.get("text", "") for p in parts), usage.get("promptTokenCount", 0), out_tokens
    return r, extract


PROVIDERS = {"openai": _call_openai, "anthropic": _call_anthropic, "gemini": _call_gemini}


def chat(system: str, user: str, json_mode: bool = False, timeout: float = 60.0) -> LLMResponse:
    """One LLM call with up to 3 retries on rate limits / server errors. Raises LLMError on failure."""
    s = settings()
    if not is_configured():
        raise LLMError("LLM is not configured: set LOGAGENT_LLM_PROVIDER, LOGAGENT_LLM_API_KEY and LOGAGENT_LLM_MODEL in .env")
    if s["provider"] not in PROVIDERS:
        raise LLMError(f"Unknown provider '{s['provider']}'. Use one of: {', '.join(PROVIDERS)}")

    start = time.perf_counter()
    last_problem = ""
    with httpx.Client(timeout=timeout) as client:
        for attempt in range(3):
            try:
                response, extract = PROVIDERS[s["provider"]](s, system, user, json_mode, client)
            except httpx.HTTPError as error:
                last_problem = f"network error: {type(error).__name__}"
            else:
                if response.status_code in RETRY_STATUS:
                    last_problem = f"HTTP {response.status_code}"
                elif response.status_code >= 400:
                    raise LLMError(f"HTTP {response.status_code} from {s['provider']}: {response.text[:300]}")
                else:
                    try:
                        text, n_in, n_out = extract(response.json())
                    except (KeyError, IndexError, TypeError, ValueError) as error:
                        raise LLMError(f"Unexpected response format from {s['provider']}: {error!r}") from error
                    return LLMResponse(text, int(n_in), int(n_out), time.perf_counter() - start)
            time.sleep(2 ** attempt)
    raise LLMError(f"{s['provider']} request failed after 3 attempts ({last_problem})")