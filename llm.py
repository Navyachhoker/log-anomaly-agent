
"""Small client for API-based LLMs used for diagnosis and zero-shot evaluation."""

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import httpx

import config  # noqa: F401 - loads .env


RETRY_STATUS = {429, 500, 502, 503, 504}
RETRY_WAITS = (5, 20, 60)


class LLMError(RuntimeError):
    """Failure communicating with an LLM API."""

    def __init__(self, message, attempts=0, latency_s=0.0):
        super().__init__(message)
        self.attempts = attempts
        self.latency_s = latency_s


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    attempts: int = 1


def settings() -> dict:
    return {
        "provider": os.getenv("LOGAGENT_LLM_PROVIDER", "").strip().lower(),
        "api_key": os.getenv("LOGAGENT_LLM_API_KEY", "").strip(),
        "model": os.getenv("LOGAGENT_LLM_MODEL", "").strip(),
        "base_url": os.getenv(
            "LOGAGENT_LLM_BASE_URL",
            "https://api.openai.com/v1"
        ).rstrip("/"),
    }


def is_configured() -> bool:
    s = settings()
    return bool(s["provider"] and s["api_key"] and s["model"])


def request_timeout() -> float:
    """Read request timeout from environment."""
    try:
        value = float(os.getenv("LOGAGENT_LLM_TIMEOUT", "60"))
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        raise LLMError(
            "LOGAGENT_LLM_TIMEOUT must be a positive number."
        ) from None


def prices_per_million_tokens() -> Tuple[Optional[float], Optional[float]]:
    def read(name):
        value = os.getenv(name, "").strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            raise LLMError(
                f"{name} in .env must be a number."
            ) from None

    return (
        read("LOGAGENT_PRICE_IN_PER_MTOK"),
        read("LOGAGENT_PRICE_OUT_PER_MTOK"),
    )


def cost_usd(input_tokens: int, output_tokens: int) -> Optional[float]:
    price_in, price_out = prices_per_million_tokens()

    if price_in is None or price_out is None:
        return None

    return (
        input_tokens * price_in + output_tokens * price_out
    ) / 1_000_000


def parse_json(text: str) -> dict:
    cleaned = re.sub(
        r"^```(?:json)?|```$",
        "",
        text.strip(),
        flags=re.MULTILINE,
    ).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start < 0 or end <= start:
        raise ValueError("no JSON object found in model output")

    return json.loads(cleaned[start:end + 1])


def _call_openai(s, system, user, json_mode, client):
    body = {
        "model": s["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    if json_mode:
        body["response_format"] = {"type": "json_object"}

    response = client.post(
        f"{s['base_url']}/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {s['api_key']}"},
    )

    def extract(data):
        usage = data["usage"]
        return (
            data["choices"][0]["message"]["content"] or "",
            usage["prompt_tokens"],
            usage["completion_tokens"],
        )

    return response, extract


def _call_anthropic(s, system, user, json_mode, client):
    body = {
        "model": s["model"],
        "max_tokens": 1024,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }

    response = client.post(
        "https://api.anthropic.com/v1/messages",
        json=body,
        headers={
            "x-api-key": s["api_key"],
            "anthropic-version": "2023-06-01",
        },
    )

    def extract(data):
        return (
            "".join(
                block.get("text", "")
                for block in data["content"]
                if block.get("type") == "text"
            ),
            data["usage"]["input_tokens"],
            data["usage"]["output_tokens"],
        )

    return response, extract


def _call_gemini(s, system, user, json_mode, client):
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
    }

    generation_config = {}

    if json_mode:
        generation_config["responseMimeType"] = "application/json"

    # Gemini 3.x uses thinkingLevel.
    # Gemini 2.5 uses thinkingBudget.
    level = os.getenv(
        "LOGAGENT_GEMINI_THINKING_LEVEL", ""
    ).strip().lower()

    budget = os.getenv(
        "LOGAGENT_GEMINI_THINKING_BUDGET", ""
    ).strip()

    if level:
        if level not in {"low", "medium", "high"}:
            raise LLMError(
                "LOGAGENT_GEMINI_THINKING_LEVEL must be low, medium or high."
            )

        generation_config["thinkingConfig"] = {
            "thinkingLevel": level
        }

    elif budget:
        generation_config["thinkingConfig"] = {
            "thinkingBudget": int(budget)
        }

    if generation_config:
        body["generationConfig"] = generation_config

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{s['model']}:generateContent"
    )

    response = client.post(
        url,
        json=body,
        headers={"x-goog-api-key": s["api_key"]},
    )

    def extract(data):
        parts = data["candidates"][0]["content"]["parts"]
        usage = data.get("usageMetadata", {})

        output_tokens = (
            usage.get("candidatesTokenCount", 0)
            + usage.get("thoughtsTokenCount", 0)
        )

        return (
            "".join(part.get("text", "") for part in parts),
            usage.get("promptTokenCount", 0),
            output_tokens,
        )

    return response, extract


PROVIDERS = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "gemini": _call_gemini,
}


def chat(
    system: str,
    user: str,
    json_mode: bool = False,
    timeout: Optional[float] = None,
) -> LLMResponse:
    """Call an LLM API with retries on transient errors."""

    s = settings()

    if not is_configured():
        raise LLMError(
            "LLM is not configured. Check LOGAGENT_LLM_* in .env."
        )

    if s["provider"] not in PROVIDERS:
        raise LLMError(
            f"Unknown provider '{s['provider']}'. "
            f"Supported: {', '.join(PROVIDERS)}"
        )

    timeout = timeout if timeout is not None else request_timeout()

    start = time.perf_counter()
    last_problem = ""
    attempts = len(RETRY_WAITS) + 1
    completed_attempts = 0

    with httpx.Client(timeout=timeout) as client:
        for attempt in range(attempts):
            wait = None
            completed_attempts = attempt + 1

            try:
                response, extract = PROVIDERS[s["provider"]](
                    s, system, user, json_mode, client
                )

            except httpx.HTTPError as error:
                last_problem = (
                    f"network error: {type(error).__name__}"
                )

            else:
                if response.status_code in RETRY_STATUS:
                    last_problem = (
                        f"HTTP {response.status_code}: "
                        f"{response.text[:300]}"
                    )

                    retry_after = response.headers.get(
                        "retry-after", ""
                    )

                    if retry_after.replace(".", "", 1).isdigit():
                        wait = float(retry_after)

                elif response.status_code >= 400:
                    raise LLMError(
                        f"HTTP {response.status_code} from "
                        f"{s['provider']}: {response.text[:300]}",
                        attempts=completed_attempts,
                        latency_s=time.perf_counter() - start,
                    )

                else:
                    try:
                        text, n_in, n_out = extract(response.json())

                    except (
                        KeyError,
                        IndexError,
                        TypeError,
                        ValueError,
                    ) as error:
                        raise LLMError(
                            f"Unexpected response format from "
                            f"{s['provider']}: {error!r}",
                            attempts=completed_attempts,
                            latency_s=time.perf_counter() - start,
                        ) from error

                    return LLMResponse(
                        text=text,
                        input_tokens=int(n_in),
                        output_tokens=int(n_out),
                        latency_s=time.perf_counter() - start,
                        attempts=completed_attempts,
                    )

            if attempt < attempts - 1:
                delay = min(
                    wait if wait is not None else RETRY_WAITS[attempt],
                    90,
                )
                time.sleep(delay)

    raise LLMError(
        f"{s['provider']} request failed after "
        f"{completed_attempts} attempts ({last_problem})",
        attempts=completed_attempts,
        latency_s=time.perf_counter() - start,
    )
