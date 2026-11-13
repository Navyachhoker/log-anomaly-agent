"""Phase 2: download, inspect and build the HDFS_v1 dataset.

    python prepare_data.py download          get HDFS_v1 from LogHub into data/raw
    python prepare_data.py inspect           real statistics + token lengths (no files for training)
    python prepare_data.py build --pilot     small pilot set (for testing the pipeline and Colab)
    python prepare_data.py build             full train / val / test files

Pipeline: group log lines by block ID -> clean + label -> split blocks 80/10/10 (stratified) ->
balance ONLY the training split -> write JSONL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import re
import sys
import urllib.request
import zipfile
from array import array
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config
from logtext import (
    HDFS_LINE, LABEL_ANOMALY, LABEL_NORMAL, build_messages, build_snippet, collapse_repeats,
    extract_block_ids, load_tokenizer, make_token_counter, normalize_line, prompt_length,
)

assert array("I").itemsize == 4, "unexpected platform: array('I') must be 4 bytes"
LEAK = re.compile(r"blk_|\b\d{6} \d{6}\b|\b\d{1,3}(?:\.\d{1,3}){3}\b")  # must never survive cleaning


# ============================================================================ small helpers
def save_json(name: str, data: dict) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / name
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved {path}")


def describe(values) -> dict:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {}
    return {
        "min": float(a.min()), "median": float(np.median(a)), "mean": round(float(a.mean()), 2),
        "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)), "max": float(a.max()),
    }


def find_raw_files() -> Tuple[Path, Path]:
    logs = sorted(config.RAW_DIR.rglob("HDFS.log"))
    labels = sorted(config.RAW_DIR.rglob("anomaly_label.csv"))
    if not logs or not labels:
        sys.exit("HDFS.log / anomaly_label.csv not found under data\\raw. Run: python prepare_data.py download")
    return logs[0], labels[0]


def load_labels(path: Path) -> Tuple[Dict[str, int], int]:
    """Returns ({block_id: 0 (Normal) or 1 (Anomaly)}, number of csv rows)."""
    df = pd.read_csv(path)
    if not {"BlockId", "Label"} <= set(df.columns):
        sys.exit(f"{path} must have columns BlockId and Label, found {list(df.columns)}")
    labels = df["Label"].astype(str).str.strip().str.lower().map({"normal": 0, "anomaly": 1})
    if labels.isna().any():
        sys.exit(f"Unexpected values in the Label column: {sorted(df['Label'][labels.isna()].unique())}")
    return dict(zip(df["BlockId"].astype(str).str.strip(), labels.astype(int))), len(df)


# ============================================================================ download
def cmd_download(args) -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = config.RAW_DIR / "HDFS_v1.zip"
    if not zip_path.exists():
        print(f"Downloading {config.HDFS_URL}")
        request = urllib.request.Request(config.HDFS_URL, headers={"User-Agent": "Mozilla/5.0"})
        part = zip_path.with_suffix(".part")
        try:
            with urllib.request.urlopen(request, timeout=60) as response, open(part, "wb") as out:
                total = int(response.headers.get("Content-Length") or 0)
                with tqdm(total=total or None, unit="B", unit_scale=True) as bar:
                    while chunk := response.read(1 << 20):
                        out.write(chunk)
                        bar.update(len(chunk))
            part.replace(zip_path)
        except Exception as error:
            part.unlink(missing_ok=True)
            sys.exit(
                f"Download failed: {error}\nManual option: open https://github.com/logpai/loghub, "
                f"download the HDFS_v1 file, save it as {zip_path} and run this command again."
            )
    else:
        print(f"{zip_path} already exists; not downloading again")

    sha = hashlib.sha256()
    with open(zip_path, "rb") as f:
        while chunk := f.read(1 << 20):
            sha.update(chunk)

    if not list(config.RAW_DIR.rglob("HDFS.log")):
        print("Extracting ...")
        try:
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(config.RAW_DIR)
        except zipfile.BadZipFile:
            zip_path.unlink()
            sys.exit("The zip file was corrupt and has been deleted. Run the command again.")

    log_path, label_path = find_raw_files()
    info = {
        "url": config.HDFS_URL, "downloaded_or_checked_on": date.today().isoformat(),
        "zip_size_bytes": zip_path.stat().st_size, "zip_sha256": sha.hexdigest(),
        "log_file": str(log_path.relative_to(config.ROOT)), "log_size_bytes": log_path.stat().st_size,
        "label_file": str(label_path.relative_to(config.ROOT)),
    }
    print(json.dumps(info, indent=2))
    save_json("data_download.json", info)


# ============================================================================ grouping
def group_log(log_path: Path, label_map: Dict[str, int], show_progress: bool = True) -> dict:
    """One pass over the log. Lines are cleaned and stored as integer IDs into a table of unique
    cleaned lines ('templates'), grouped per block ID in file order. A line that mentions several
    blocks is added to each of them."""
    templates: List[str] = []
    index: Dict[str, int] = {}
    blocks: Dict[str, array] = {}
    levels: Counter = Counter()
    unlabelled = set()
    stats = {k: 0 for k in ("lines", "blank_lines", "lines_not_matching_hdfs_format",
                            "lines_without_block_id", "lines_with_multiple_block_ids")}

    with open(log_path, encoding="utf-8", errors="replace") as f:
        for raw in tqdm(f, total=config.EXPECTED_LINES, unit=" lines", mininterval=5, disable=not show_progress):
            raw = raw.rstrip("\r\n")
            if not raw.strip():
                stats["blank_lines"] += 1
                continue
            stats["lines"] += 1
            match = HDFS_LINE.match(raw)
            if match:
                levels[match["level"]] += 1
            else:
                stats["lines_not_matching_hdfs_format"] += 1
            ids = extract_block_ids(raw)
            if not ids:
                stats["lines_without_block_id"] += 1
                continue
            if len(ids) > 1:
                stats["lines_with_multiple_block_ids"] += 1
            known = []
            for block_id in ids:
                if block_id in label_map:
                    known.append(block_id)
                else:
                    unlabelled.add(block_id)
            if not known:
                continue
            text = normalize_line(raw)
            template_id = index.get(text)
            if template_id is None:
                template_id = index[text] = len(templates)
                templates.append(text)
            for block_id in known:
                if block_id not in blocks:
                    blocks[block_id] = array("I")
                blocks[block_id].append(template_id)

    stats["blocks_in_log_without_label"] = len(unlabelled)
    return {"blocks": blocks, "templates": templates, "levels": dict(levels), "stats": stats}


def load_grouped(refresh: bool = False) -> dict:
    """Grouped data, cached in data/interim/grouped.pkl because the pass takes several minutes."""
    cache = config.INTERIM_DIR / "grouped.pkl"
    if cache.exists() and not refresh:
        print(f"Loading cached grouping from {cache} (use --refresh to rebuild)")
        with open(cache, "rb") as f:
            return pickle.load(f)
    log_path, label_path = find_raw_files()
    label_map, _ = load_labels(label_path)
    grouped = group_log(log_path, label_map)
    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(grouped, f, protocol=pickle.HIGHEST_PROTOCOL)
    return grouped


# ============================================================================ inspect
def cmd_inspect(args) -> None:
    grouped = load_grouped(args.refresh)
    blocks, templates, stats = grouped["blocks"], grouped["templates"], grouped["stats"]
    _, label_path = find_raw_files()
    label_map, csv_rows = load_labels(label_path)
    rng = random.Random(config.SEED)

    n_anomaly = sum(label_map.values())
    n_normal = len(label_map) - n_anomaly
    block_class = {b: label_map[b] for b in blocks}
    report = {
        "parsing": {
            **stats,
            "lines_expected_by_loghub": config.EXPECTED_LINES,
            "line_count_matches_loghub": stats["lines"] + stats["blank_lines"] == config.EXPECTED_LINES,
            "levels": grouped["levels"],
            "unique_cleaned_lines": len(templates),
        },
        "labels": {
            "csv_rows": csv_rows, "unique_blocks_in_csv": len(label_map),
            "normal": n_normal, "anomaly": n_anomaly,
            "anomaly_rate": round(n_anomaly / len(label_map), 5),
            "csv_blocks_without_log_lines": len(label_map) - len(blocks),
        },
    }

    # lines per block, by class
    per_class = {0: [], 1: []}
    for b, arr in blocks.items():
        per_class[block_class[b]].append(len(arr))
    report["raw_lines_per_block"] = {"normal": describe(per_class[0]), "anomaly": describe(per_class[1])}

    # does a block contain any non-INFO line? (explains how easy/hard the task is)
    non_info = np.array([not t.startswith("INFO") for t in templates])
    has_non_info = {0: [], 1: []}
    for b, arr in blocks.items():
        has_non_info[block_class[b]].append(bool(non_info[np.frombuffer(arr, dtype=np.uint32)].any()))
    report["share_of_blocks_with_a_non_INFO_line"] = {
        "normal": round(float(np.mean(has_non_info[0])), 4), "anomaly": round(float(np.mean(has_non_info[1])), 4),
    }

    # most frequent cleaned lines
    all_ids = np.concatenate([np.frombuffer(a, dtype=np.uint32) for a in blocks.values()])
    counts = np.bincount(all_ids, minlength=len(templates))
    report["top_15_cleaned_lines"] = [
        {"count": int(counts[i]), "line": templates[i]} for i in np.argsort(counts)[::-1][:15]
    ]

    # token lengths with the real Qwen tokenizer
    tokenizer = load_tokenizer(config.BASE_MODEL)
    count = make_token_counter(tokenizer)
    overhead = prompt_length(tokenizer, build_messages(""), generation_prompt=True)
    normal_ids = sorted(b for b in blocks if block_class[b] == 0)
    anomaly_ids = sorted(b for b in blocks if block_class[b] == 1)
    sample = {
        "normal": rng.sample(normal_ids, min(5000, len(normal_ids))),
        "anomaly": rng.sample(anomaly_ids, min(5000, len(anomaly_ids))),
    }
    budgets = [256, 384, 512, 768, 1024]
    token_report = {"prompt_overhead_tokens_without_snippet": overhead, "sampled_blocks_per_class": {k: len(v) for k, v in sample.items()}}
    for name, ids in sample.items():
        lengths = [sum(count(l) for l in collapse_repeats([templates[t] for t in blocks[b]])) for b in tqdm(ids, desc=f"tokens ({name})")]
        token_report[name] = {
            "snippet_tokens": describe(lengths),
            "share_longer_than_budget": {str(x): round(float(np.mean(np.array(lengths) > x)), 4) for x in budgets},
        }
    report["token_lengths_untruncated_after_collapsing_repeats"] = token_report

    # label tokens (needed later for the confidence score)
    label_tokens = {}
    for label in (LABEL_NORMAL, LABEL_ANOMALY):
        ids = tokenizer.encode(label, add_special_tokens=False)
        label_tokens[label] = {"token_ids": ids, "pieces": tokenizer.convert_ids_to_tokens(ids)}
    report["label_tokens"] = label_tokens
    report["label_first_tokens_differ"] = label_tokens[LABEL_NORMAL]["token_ids"][0] != label_tokens[LABEL_ANOMALY]["token_ids"][0]

    # how many anomalies would an evaluation subset contain? (test split = 10% of blocks, stratified)
    test_blocks, test_anomalies = round(0.1 * len(label_map)), round(0.1 * n_anomaly)
    rate = n_anomaly / len(label_map)
    table = []
    for size in (1000, 2000, 3000, 5000, 10000, test_blocks):
        anomalies = size * rate
        table.append({
            "eval_subset_size": size, "expected_anomalies": round(anomalies),
            "rough_95pct_halfwidth_of_recall_if_recall_is_0.9": round(1.96 * (0.9 * 0.1 / anomalies) ** 0.5, 3),
        })
    report["evaluation_subset_options"] = {"test_blocks": test_blocks, "test_anomalies": test_anomalies, "table": table}

    save_json("data_inspection.json", report)
    print(json.dumps(report, indent=2))

    print("\n=== A few cleaned snippets, for your own review (first 10 lines each) ===")
    for name, ids in (("NORMAL", normal_ids), ("ANOMALY", anomaly_ids)):
        for b in rng.sample(ids, 2):
            lines = collapse_repeats([templates[t] for t in blocks[b]])
            print(f"\n--- {name} ({len(blocks[b])} raw lines) ---")
            print("\n".join(lines[:10]))


# ============================================================================ build
def split_blocks(ids: List[str], label_map: Dict[str, int]) -> Tuple[List[str], List[str], List[str]]:
    """80/10/10 by block ID, stratified by label, seeded."""
    y = [label_map[b] for b in ids]
    train, rest, _, y_rest = train_test_split(ids, y, test_size=0.20, stratify=y, random_state=config.SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=y_rest, random_state=config.SEED)
    return sorted(train), sorted(val), sorted(test)


def balance_train(train_ids: List[str], label_map: Dict[str, int], cap_per_class: int, rng: random.Random) -> List[str]:
    """All anomalies plus the same number of random normals (optionally capped), shuffled."""
    anomalies = [b for b in train_ids if label_map[b] == 1]
    normals = [b for b in train_ids if label_map[b] == 0]
    n = min(len(anomalies), len(normals))
    if cap_per_class:
        n = min(n, cap_per_class)
    chosen = rng.sample(anomalies, n) + rng.sample(normals, n)
    rng.shuffle(chosen)
    return chosen


def per_class_sample(ids: List[str], label_map: Dict[str, int], n: int, rng: random.Random) -> List[str]:
    anomalies = [b for b in ids if label_map[b] == 1]
    normals = [b for b in ids if label_map[b] == 0]
    chosen = rng.sample(anomalies, min(n, len(anomalies))) + rng.sample(normals, min(n, len(normals)))
    rng.shuffle(chosen)
    return chosen


def duplicate_report(blocks: Dict[str, array], label_map: Dict[str, int], train_ids: List[str], test_ids: List[str]) -> dict:
    """How many test blocks have a line sequence identical to some training block? (Many HDFS blocks
    follow the same sequence, so this helps interpret test results honestly.)"""
    seen: Dict[bytes, set] = {}
    for b in train_ids:
        seen.setdefault(blocks[b].tobytes(), set()).add(label_map[b])
    totals = {0: 0, 1: 0}
    duplicated = {0: 0, 1: 0}
    same_label = {0: 0, 1: 0}
    for b in test_ids:
        label = label_map[b]
        totals[label] += 1
        found = seen.get(blocks[b].tobytes())
        if found is not None:
            duplicated[label] += 1
            same_label[label] += found == {label}
    share = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
    return {
        "test_blocks_with_identical_sequence_in_train": {
            "all": share(duplicated[0] + duplicated[1], totals[0] + totals[1]),
            "normal": share(duplicated[0], totals[0]), "anomaly": share(duplicated[1], totals[1]),
        },
        "of_those_train_copies_have_only_the_same_label": {
            "normal": share(same_label[0], duplicated[0]), "anomaly": share(same_label[1], duplicated[1]),
        },
    }


def cmd_build(args) -> None:
    grouped = load_grouped(args.refresh)
    blocks, templates = grouped["blocks"], grouped["templates"]
    _, label_path = find_raw_files()
    label_map, _ = load_labels(label_path)
    rng = random.Random(config.SEED)

    ids = sorted(b for b in blocks if b in label_map)
    train_ids, val_ids, test_ids = split_blocks(ids, label_map)
    assert not (set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids))
    duplicates = duplicate_report(blocks, label_map, train_ids, test_ids)
    train_balanced = balance_train(train_ids, label_map, args.max_per_class, rng)

    if args.pilot:
        out_dir = config.PROCESSED_DIR / "pilot"
        splits = {
            "train": per_class_sample(train_balanced, label_map, 250, rng),
            "val": per_class_sample(val_ids, label_map, 100, rng),
        }
    else:
        out_dir = config.PROCESSED_DIR
        splits = {"train": train_balanced, "val": val_ids, "test": test_ids}
        if args.eval_size:
            if args.eval_size >= len(test_ids):
                sys.exit(f"--eval-size must be smaller than the test split ({len(test_ids)} blocks)")
            eval_ids, _ = train_test_split(test_ids, train_size=args.eval_size,
                                           stratify=[label_map[b] for b in test_ids], random_state=config.SEED)
            splits["test_eval"] = sorted(eval_ids)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config.BASE_MODEL)
    count = make_token_counter(tokenizer)
    summary = {"pilot": args.pilot, "max_snippet_tokens": args.max_tokens, "seed": config.SEED,
               "split_blocks_before_balancing": {
                   n: {"blocks": len(s), "anomalies": sum(label_map[b] for b in s)}
                   for n, s in (("train", train_ids), ("val", val_ids), ("test", test_ids))},
               "train_test_duplicate_analysis": duplicates, "files": {}}
    examples: Dict[str, List[str]] = {LABEL_NORMAL: [], LABEL_ANOMALY: []}
    leaks = 0

    for name, split_ids in splits.items():
        path = out_dir / f"{name}.jsonl"
        check = set(rng.sample(range(len(split_ids)), min(1000, len(split_ids))))
        anomalies = truncated = 0
        tokens: List[int] = []
        prompt_tokens: List[int] = []
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for i, b in enumerate(tqdm(split_ids, desc=f"writing {path.name}")):
                snippet = build_snippet([templates[t] for t in blocks[b]], args.max_tokens, count)
                label = LABEL_ANOMALY if label_map[b] else LABEL_NORMAL
                f.write(json.dumps({
                    "block_id": b, "label": label, "snippet": snippet.text, "num_lines": snippet.total_lines,
                    "truncated": snippet.truncated, "snippet_tokens": snippet.tokens,
                }, ensure_ascii=False) + "\n")
                anomalies += label_map[b]
                truncated += snippet.truncated
                tokens.append(snippet.tokens)
                leaks += bool(LEAK.search(snippet.text))
                if i in check:
                    prompt_tokens.append(prompt_length(tokenizer, build_messages(snippet.text, label), generation_prompt=False))
                if name == "train" and len(examples[label]) < 3:
                    examples[label].append(snippet.text)
        n = len(split_ids)
        summary["files"][name] = {
            "path": str(path.relative_to(config.ROOT)), "samples": n, "anomalies": int(anomalies),
            "normals": n - int(anomalies), "anomaly_rate": round(anomalies / n, 5),
            "share_truncated": round(truncated / n, 4), "snippet_tokens": describe(tokens),
            "full_training_prompt_tokens_checked_on_a_sample": {
                "sample_size": len(prompt_tokens), **describe(prompt_tokens),
                "sample_over_512": int(sum(p > 512 for p in prompt_tokens)),
            },
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    summary["snippets_containing_block_ids_timestamps_or_ips_after_cleaning"] = leaks

    save_json("pilot_stats.json" if args.pilot else "dataset_stats.json", summary)
    print(json.dumps(summary, indent=2))
    sample_text = []
    for label, texts in examples.items():
        for k, text in enumerate(texts, 1):
            sample_text.append(f"===== {label} example {k} (from the training file) =====\n{text}\n")
    (config.RESULTS_DIR / ("pilot_samples.txt" if args.pilot else "sample_snippets.txt")).write_text("\n".join(sample_text), encoding="utf-8")


# ============================================================================ main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download").set_defaults(func=cmd_download)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--refresh", action="store_true", help="re-read HDFS.log instead of using the cache")
    inspect.set_defaults(func=cmd_inspect)
    build = sub.add_parser("build")
    build.add_argument("--pilot", action="store_true", help="write a small pilot set to data/processed/pilot")
    build.add_argument("--max-tokens", type=int, default=config.MAX_SNIPPET_TOKENS, help="snippet token budget")
    build.add_argument("--max-per-class", type=int, default=0, help="cap per class in the balanced train set (0 = no cap)")
    build.add_argument("--eval-size", type=int, default=0, help="also write test_eval.jsonl of this size (decide after inspect)")
    build.add_argument("--refresh", action="store_true")
    build.set_defaults(func=cmd_build)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()