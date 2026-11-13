"""Create the evaluation subsets from the full val/test files (natural class ratio preserved).

    python make_eval_sets.py --val-size 6000 --test-size 6000

Sizes are chosen after `prepare_data.py inspect` shows how many anomalies each size would contain.
"""
import argparse
import json
import sys

from sklearn.model_selection import train_test_split

import config


def subset(name: str, size: int) -> dict:
    source = config.PROCESSED_DIR / f"{name}.jsonl"
    if not source.exists():
        sys.exit(f"{source} not found. Run: python prepare_data.py build")
    lines = source.read_text(encoding="utf-8").splitlines()
    labels = [json.loads(line)["label"] for line in lines]
    if size >= len(lines):
        sys.exit(f"--{name}-size must be smaller than the {name} split ({len(lines)})")
    chosen, _ = train_test_split(range(len(lines)), train_size=size, stratify=labels, random_state=config.SEED)
    chosen = sorted(chosen)
    target = config.PROCESSED_DIR / f"{name}_eval.jsonl"
    target.write_text("\n".join(lines[i] for i in chosen) + "\n", encoding="utf-8", newline="\n")
    anomalies = sum(labels[i] == "Anomaly" for i in chosen)
    print(f"{target.name}: {size} samples, {anomalies} anomalies ({anomalies / size:.2%}); source has {len(lines)} samples")
    return {"size": size, "anomalies": anomalies, "source_size": len(lines)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val-size", type=int, required=True, help="validation subset (threshold tuning, provider comparison)")
    p.add_argument("--test-size", type=int, required=True, help="test subset used for the three-model comparison")
    args = p.parse_args()
    info = {"val_eval": subset("val", args.val_size), "test_eval": subset("test", args.test_size), "seed": config.SEED}
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "eval_sets.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()