"""Log cleaning, snippet building and prompt text.

Used by data preparation now, and by evaluation and the web app later, so that training and
serving see exactly the same text format.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable, Dict, List, NamedTuple, Optional

LABEL_NORMAL = "Normal"
LABEL_ANOMALY = "Anomaly"
LABELS = (LABEL_NORMAL, LABEL_ANOMALY)

# Raw HDFS_v1 line:  081109 203615 148 INFO dfs.DataNode$PacketResponder: <message>
HDFS_LINE = re.compile(
    r"^(?P<date>\d{6})\s+(?P<time>\d{6})\s+(?P<pid>\d+)\s+"
    r"(?P<level>[A-Z]+)\s+(?P<component>[^\s:]+):\s*(?P<content>.*)$"
)
# Leading ISO timestamp, e.g. "2025-06-14 12:01:32 ERROR ..." (logs pasted into the dashboard)
ISO_TIMESTAMP = re.compile(
    r"^\[?\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?\]?\s*"
)
BLOCK_ID = re.compile(r"blk_-?\d+")
IP_ADDRESS = re.compile(r"/?\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b")
LONG_NUMBER = re.compile(r"(?<![A-Za-z<])\d{3,}(?![A-Za-z>])")  # 3+ digits, not glued to letters
WHITESPACE = re.compile(r"\s+")


# ----------------------------------------------------------------------------- cleaning
def extract_block_ids(raw_line: str) -> List[str]:
    """Distinct block IDs in a raw line, in order of appearance."""
    return list(dict.fromkeys(BLOCK_ID.findall(raw_line)))


def normalize_message(text: str) -> str:
    text = BLOCK_ID.sub("<BLK>", text)
    text = IP_ADDRESS.sub("<IP>", text)
    text = LONG_NUMBER.sub("<NUM>", text)
    return WHITESPACE.sub(" ", text).strip()


def normalize_line(raw_line: str) -> str:
    """Drop date/time/pid (or an ISO timestamp); keep 'LEVEL component: message';
    replace block IDs, IP addresses/ports and long numbers with placeholders."""
    raw_line = raw_line.strip()
    match = HDFS_LINE.match(raw_line)
    if match:
        body = f"{match['level']} {match['component']}: {match['content']}"
    else:
        body = ISO_TIMESTAMP.sub("", raw_line, count=1)
    return normalize_message(body)


# ----------------------------------------------------------------------------- snippets
class Snippet(NamedTuple):
    text: str
    total_lines: int  # lines before collapsing/truncation
    kept_lines: int   # lines present in `text`
    truncated: bool
    tokens: int       # token count of `text`


def collapse_repeats(lines: List[str]) -> List[str]:
    """Collapse runs of identical consecutive lines into 'line [xN]'."""
    out: List[str] = []
    previous: Optional[str] = None
    run = 0
    for line in lines:
        if line == previous:
            run += 1
            continue
        if previous is not None:
            out.append(previous if run == 1 else f"{previous} [x{run}]")
        previous, run = line, 1
    if previous is not None:
        out.append(previous if run == 1 else f"{previous} [x{run}]")
    return out


def build_snippet(lines: List[str], max_tokens: int, count_tokens: Callable[[str], int]) -> Snippet:
    """Collapse repeated lines, then fit the snippet into `max_tokens`.

    If it is too long, WARN/ERROR (non-INFO) lines are kept first, then INFO lines from the start
    fill the remaining budget. The original line order is always preserved.
    """
    total = len(lines)
    lines = collapse_repeats(lines)
    costs = [count_tokens(line) for line in lines]
    if sum(costs) <= max_tokens:
        return Snippet("\n".join(lines), total, len(lines), False, sum(costs))

    keep = [False] * len(lines)
    budget = max_tokens
    for routine_pass in (False, True):  # first non-INFO lines, then INFO lines
        for i, line in enumerate(lines):
            if not keep[i] and line.startswith("INFO") == routine_pass and costs[i] <= budget:
                keep[i] = True
                budget -= costs[i]
    kept = [line for line, k in zip(lines, keep) if k]
    if not kept:  # a single enormous line: cut it by characters (about 3 characters per token)
        kept = [lines[0][: max_tokens * 3]]
        return Snippet(kept[0], total, 1, True, count_tokens(kept[0]))
    return Snippet("\n".join(kept), total, len(kept), True, max_tokens - budget)


# ----------------------------------------------------------------------------- tokenizer
def load_tokenizer(name: str):
    from transformers import AutoTokenizer  # imported here so tests do not need to download anything

    return AutoTokenizer.from_pretrained(name)


def make_token_counter(tokenizer) -> Callable[[str], int]:
    """Tokens of one snippet line (+1 for the newline that joins lines). Results are cached."""

    @lru_cache(maxsize=500_000)
    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False)) + 1

    return count


def prompt_length(tokenizer, messages: List[Dict[str, str]], generation_prompt: bool) -> int:
    """Token length of the full chat prompt (system + user [+ assistant answer])."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation_prompt)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


# ----------------------------------------------------------------------------- prompt
SYSTEM_PROMPT = "You are an expert in analysing Hadoop HDFS logs."

INSTRUCTION = (
    "Classify the following HDFS log snippet as Normal or Anomaly. "
    "The snippet contains the ordered log lines of a single HDFS block "
    "(timestamps and block IDs removed). "
    "Answer with exactly one word: Normal or Anomaly."
)


def build_messages(snippet: str, label: Optional[str] = None) -> List[Dict[str, str]]:
    """Chat messages for Qwen. With `label` the answer is included (training); without it,
    the messages end at the user turn (inference)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{INSTRUCTION}\n\nLog snippet:\n{snippet}"},
    ]
    if label is not None:
        if label not in LABELS:
            raise ValueError(f"Unknown label: {label!r}")
        messages.append({"role": "assistant", "content": label})
    return messages


def parse_label(text: str) -> Optional[str]:
    """Map raw model output to 'Normal' / 'Anomaly'; None if it is neither."""
    head = (text or "").strip().lower()[:40]
    anomaly, normal = head.find("anomal"), head.find("normal")
    if anomaly >= 0 and (normal < 0 or anomaly < normal):
        return LABEL_ANOMALY
    if normal >= 0:
        return LABEL_NORMAL
    return None