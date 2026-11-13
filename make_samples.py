"""Create samples/samples.json for the dashboard's example picker, using RAW log lines of a few
validation blocks (so the demo shows real HDFS lines with timestamps, like a pasted log).

    python make_samples.py --n 3
"""
import argparse
import json
import random

import config
from logtext import extract_block_ids
from prepare_data import find_raw_files


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=3, help="examples per class")
    args = p.parse_args()

    rng = random.Random(config.SEED)
    records = [json.loads(line) for line in open(config.PROCESSED_DIR / "val.jsonl", encoding="utf-8")]
    wanted = {}
    for label in ("Normal", "Anomaly"):
        pool = [r for r in records if r["label"] == label and 4 <= r["num_lines"] <= 25]
        wanted[label] = rng.sample(pool, args.n)
    lines_by_block = {r["block_id"]: [] for group in wanted.values() for r in group}

    log_path, _ = find_raw_files()
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            for block_id in extract_block_ids(raw):
                if block_id in lines_by_block:
                    lines_by_block[block_id].append(raw.rstrip("\r\n"))

    samples = []
    for label, group in wanted.items():
        for k, record in enumerate(group, 1):
            samples.append({"id": f"{label.lower()}-{k}", "title": f"{label} example {k} (validation block)", "true_label": label,
                            "log": "\n".join(lines_by_block[record["block_id"]])})
    config.SAMPLES_FILE.parent.mkdir(exist_ok=True)
    config.SAMPLES_FILE.write_text(json.dumps(samples, indent=2), encoding="utf-8")
    print(f"Wrote {len(samples)} samples to {config.SAMPLES_FILE}")


if __name__ == "__main__":
    main()