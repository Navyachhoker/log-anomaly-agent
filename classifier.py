"""The fine-tuned (merged) Qwen classifier for local CPU inference.

Loaded once and reused. One forward pass per snippet: the next-token logits of the two label words
("Normal", "Anomaly") are turned into a probability with a softmax over just those two values.
"""
import json
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Optional

import config
from logtext import build_messages, build_snippet, make_token_counter, normalize_line

log = logging.getLogger("classifier")


class ModelLoadError(RuntimeError):
    pass


def anomaly_probability(normal_logit: float, anomaly_logit: float) -> float:
    """P(Anomaly) from the logits of the two label tokens (a 2-way softmax)."""
    difference = normal_logit - anomaly_logit  # softmax over two values = sigmoid of their difference
    if difference > 700:
        return 0.0
    return 1.0 / (1.0 + math.exp(difference))


def load_threshold() -> float:
    """Anomaly threshold: LOGAGENT_THRESHOLD, else results/threshold.json (tuned on validation), else 0.5."""
    if os.getenv("LOGAGENT_THRESHOLD"):
        return float(os.environ["LOGAGENT_THRESHOLD"])
    if config.THRESHOLD_FILE.exists():
        return float(json.loads(config.THRESHOLD_FILE.read_text())["threshold"])
    return 0.5


class SlmClassifier:
    def __init__(self, model_dir: Path = config.MODEL_DIR, dtype: Optional[str] = None,
                 threshold: Optional[float] = None, max_snippet_tokens: int = config.MAX_SNIPPET_TOKENS):
        try:  # imported here so the rest of the project (and its tests) work without PyTorch
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise ModelLoadError(f"PyTorch/Transformers are not installed in this environment: {error}") from error

        self._torch = torch
        model_dir = Path(model_dir)
        if not (model_dir / "config.json").exists():
            raise ModelLoadError(f"No model found in {model_dir}. Merge the adapter on Colab and download it there (see README).")
        dtype = (dtype or os.getenv("LOGAGENT_DTYPE", "float32")).lower()
        if dtype not in ("float32", "bfloat16", "int8"):
            raise ModelLoadError(f"Unsupported dtype {dtype!r} (use float32, bfloat16 or int8)")
        threads = os.getenv("LOGAGENT_TORCH_THREADS")
        if threads:
            torch.set_num_threads(int(threads))
        start = time.perf_counter()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir, torch_dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float32)
            if dtype == "int8":
                self.model = torch.ao.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        except Exception as error:
            raise ModelLoadError(f"Could not load the model from {model_dir}: {error}") from error
        self.model.eval()
        self.normal_id = self.tokenizer.encode("Normal", add_special_tokens=False)[0]
        self.anomaly_id = self.tokenizer.encode("Anomaly", add_special_tokens=False)[0]
        if self.normal_id == self.anomaly_id:
            raise ModelLoadError("The label words 'Normal' and 'Anomaly' share their first token; scoring is not possible.")
        self.count_tokens = make_token_counter(self.tokenizer)
        self.max_snippet_tokens = max_snippet_tokens
        self.threshold = threshold if threshold is not None else load_threshold()
        self.dtype = dtype
        self.load_seconds = time.perf_counter() - start
        self._lock = threading.Lock()  # one forward pass at a time
        log.info("Model loaded from %s (%s) in %.1fs, threshold=%.3f", model_dir, dtype, self.load_seconds, self.threshold)

    def classify_snippet(self, snippet: str, truncated: bool = False) -> dict:
        """Classify an already-cleaned snippet."""
        prompt = self.tokenizer.apply_chat_template(build_messages(snippet), tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        start = time.perf_counter()
        with self._lock, self._torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1]
        latency = time.perf_counter() - start
        p_anomaly = anomaly_probability(float(logits[self.normal_id]), float(logits[self.anomaly_id]))
        is_anomaly = p_anomaly >= self.threshold
        return {
            "label": "Anomaly" if is_anomaly else "Normal",
            "confidence": p_anomaly if is_anomaly else 1.0 - p_anomaly,
            "p_anomaly": p_anomaly, "threshold": self.threshold, "snippet": snippet, "truncated": truncated,
            "latency_s": latency, "input_tokens": int(inputs["input_ids"].shape[1]),
        }

    def classify(self, log_text: str) -> dict:
        """Classify raw pasted log text (one or many lines, treated as one block's snippet)."""
        lines = [normalize_line(line) for line in log_text.splitlines() if line.strip()]
        if not lines:
            raise ValueError("The log text is empty.")
        snippet = build_snippet(lines, self.max_snippet_tokens, self.count_tokens)
        return self.classify_snippet(snippet.text, snippet.truncated)