"""Phase 2 tests. Run from the project folder:  .venv\\Scripts\\python.exe -m pytest -q

NOTE: the log lines below follow the documented HDFS_v1 format and the logs are SYNTHETIC. They test
the code only. They say nothing about the real dataset; `prepare_data.py inspect` does that.
"""
import random
from array import array

import prepare_data
from logtext import (
    build_messages, build_snippet, collapse_repeats, extract_block_ids, normalize_line, parse_label,
)

RAW = "081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating"


def test_raw_line_is_cleaned():
    assert normalize_line(RAW) == "INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block <BLK> terminating"


def test_ip_port_number_and_negative_block_id():
    raw = ("081109 203518 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: "
           "10.250.19.102:50010 is added to blk_-1608999687919862906 size 67108864")
    out = normalize_line(raw)
    assert out.endswith("blockMap updated: <IP> is added to <BLK> size <NUM>")


def test_task_ids_collapse_to_one_pattern():
    a = normalize_line("081109 1 35 INFO dfs.FSNamesystem: x /_task_200811092030_0001_m_000590_0/part-00590. blk_1")
    b = normalize_line("081109 1 35 INFO dfs.FSNamesystem: x /_task_200811092030_0001_m_000123_0/part-00123. blk_2")
    assert a == b


def test_iso_timestamp_from_the_report_example():
    assert normalize_line("2025-06-14 12:01:32 ERROR dfs.DataNode: Block blk_12345 missing") == \
        "ERROR dfs.DataNode: Block <BLK> missing"


def test_multiple_block_ids_in_one_line():
    line = "081109 1 35 INFO dfs.FSNamesystem: ask <x> to delete blk_1 blk_-2 blk_1"
    assert extract_block_ids(line) == ["blk_1", "blk_-2"]


def test_collapse_and_truncation_keep_warnings():
    assert collapse_repeats(["a", "a", "a", "b"]) == ["a [x3]", "b"]
    lines = [f"INFO comp: routine message number {i}" for i in range(200)] + ["WARN comp: something bad"]
    result = build_snippet(lines, 100, lambda s: len(s) // 3 + 1)
    assert result.truncated and "WARN comp: something bad" in result.text and result.tokens <= 100
    short = build_snippet(["INFO a: b"], 100, lambda s: 5)
    assert not short.truncated and short.text == "INFO a: b"


def test_prompt_and_label_parsing():
    roles = [m["role"] for m in build_messages("x", "Normal")]
    assert roles == ["system", "user", "assistant"]
    assert [m["role"] for m in build_messages("x")] == ["system", "user"]
    assert parse_label("Anomaly") == "Anomaly" and parse_label(" normal.") == "Normal" and parse_label("hmm") is None


def _write_synthetic(tmp_path, n=300, n_anomalous=60):
    rng = random.Random(0)
    lines, labels = [], []
    for i in range(n):
        blk, bad = f"blk_{-1000000 - i * 7919}", i < n_anomalous
        t = f"081109 20{i % 60:02d}00 {100 + i % 50}"
        lines.append(f"{t} INFO dfs.DataNode$DataXceiver: Receiving block {blk} src: /10.250.19.{i % 250}:54106 dest: /10.250.19.102:50010")
        for _ in range(rng.randint(1, 3)):
            lines.append(f"{t} INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block {blk} terminating")
        if bad:
            lines.append(f"{t} WARN dfs.DataNode$DataXceiver: 10.251.43.115:50010:Got exception while serving {blk} to /10.251.43.115:")
        labels.append((blk, "Anomaly" if bad else "Normal"))
    log = tmp_path / "HDFS.log"
    log.write_text("\n".join(lines) + "\n")
    csv = tmp_path / "anomaly_label.csv"
    csv.write_text("BlockId,Label\n" + "\n".join(f"{b},{l}" for b, l in labels) + "\n")
    return log, csv


def test_grouping_labels_and_cleaning(tmp_path):
    log, csv = _write_synthetic(tmp_path)
    label_map, rows = prepare_data.load_labels(csv)
    assert rows == 300 and sum(label_map.values()) == 60
    grouped = prepare_data.group_log(log, label_map, show_progress=False)
    assert len(grouped["blocks"]) == 300 and grouped["stats"]["lines_without_block_id"] == 0
    first = grouped["blocks"]["blk_-1000000"]
    texts = [grouped["templates"][t] for t in first]
    assert texts[0].startswith("INFO dfs.DataNode$DataXceiver: Receiving block <BLK>")
    assert any(t.startswith("WARN") for t in texts)               # the anomalous block (i=0) has its WARN line
    assert not any(prepare_data.LEAK.search(t) for t in grouped["templates"])


def test_split_is_disjoint_stratified_and_only_train_is_balanced(tmp_path):
    log, csv = _write_synthetic(tmp_path)
    label_map, _ = prepare_data.load_labels(csv)
    ids = sorted(label_map)
    train, val, test = prepare_data.split_blocks(ids, label_map)
    assert not (set(train) & set(val)) and not (set(train) & set(test)) and not (set(val) & set(test))
    assert len(train) + len(val) + len(test) == 300 and len(val) == 30 and len(test) == 30
    for part in (val, test):                                     # natural ratio (20% anomalies) is kept
        assert sum(label_map[b] for b in part) == 6
    balanced = prepare_data.balance_train(train, label_map, 0, random.Random(1))
    assert sum(label_map[b] for b in balanced) == len(balanced) // 2
    assert set(balanced) <= set(train)
    capped = prepare_data.balance_train(train, label_map, 10, random.Random(1))
    assert len(capped) == 20


def test_duplicate_report_counts_identical_sequences():
    blocks = {"a": array("I", [1, 2]), "b": array("I", [1, 2]), "c": array("I", [3])}
    labels = {"a": 0, "b": 0, "c": 1}
    report = prepare_data.duplicate_report(blocks, labels, ["a"], ["b", "c"])
    assert report["test_blocks_with_identical_sequence_in_train"]["all"] == 0.5
    assert report["of_those_train_copies_have_only_the_same_label"]["normal"] == 1.0