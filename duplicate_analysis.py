"""Do the test scores depend on test blocks that repeat a training sequence?

Many HDFS blocks have exactly the same sequence of log lines. This script scores the test examples
separately for blocks WITH an identical sequence in the training data and blocks WITHOUT one.
It only analyses existing predictions; nothing is tuned.

    python duplicate_analysis.py            (needs data\\interim\\grouped.pkl from prepare_data.py)
"""
import argparse
import json
import pickle

import numpy as np

import config
import make_report as mr
from prepare_data import find_raw_files, load_labels, split_blocks


def trained_block_ids(train_jsonl, cap_per_class):
    """The blocks colab/train.py really used: the first `cap_per_class` of each class in train.jsonl."""
    counts, ids = {}, set()
    with open(train_jsonl, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            label = record["label"]
            if cap_per_class and counts.get(label, 0) >= cap_per_class:
                continue
            counts[label] = counts.get(label, 0) + 1
            ids.add(record["block_id"])
    return ids


def with_identical_sequence(blocks, reference_ids, test_ids):
    """Test blocks whose line sequence is identical to a sequence of some reference block."""
    seen = {blocks[b].tobytes() for b in reference_ids}
    return {b for b in test_ids if blocks[b].tobytes() in seen}


def analyze(blocks, label_map, test_preds, references):
    """test_preds: {model_key: {block_id: prediction record}}; references: {name: set of block ids}."""
    test_ids = sorted(set.intersection(*[set(p) for p in test_preds.values()]))
    result = {"test_examples": len(test_ids), "references": {}}
    for ref_name, ref_ids in references.items():
        twins = with_identical_sequence(blocks, ref_ids, test_ids)
        entry = {"reference_blocks": len(ref_ids), "test_blocks_with_identical_sequence": len(twins), "models": {}}
        for model, preds in test_preds.items():
            subsets = {"all": test_ids,
                       "with_identical_sequence": [b for b in test_ids if b in twins],
                       "without_identical_sequence": [b for b in test_ids if b not in twins]}
            rows = {}
            for name, ids in subsets.items():
                y_true = np.array([label_map[b] for b in ids], dtype=np.int8)
                records = [preds[b] for b in ids]
                if not ids:
                    rows[name] = {"n": 0}
                    continue
                y_pred, _ = mr.predictions_from_text(records, y_true)
                rows[name] = {"n": len(ids), "anomalies": int(y_true.sum()), **mr.score(y_true, y_pred)}
            entry["models"][model] = rows
        result["references"][ref_name] = entry
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-per-class", type=int, default=2000, help="the training cap you used (colab/train.py --max-per-class)")
    args = p.parse_args()

    with open(config.INTERIM_DIR / "grouped.pkl", "rb") as f:
        blocks = pickle.load(f)["blocks"]
    _, label_path = find_raw_files()
    label_map, _ = load_labels(label_path)
    train_split, _, _ = split_blocks(sorted(b for b in blocks if b in label_map), label_map)
    trained = trained_block_ids(config.PROCESSED_DIR / "train.jsonl", args.max_per_class)
    assert trained <= set(train_split), "the trained blocks must come from the training split"

    test_preds = {}
    for key in ("finetuned", "untuned"):
        loaded = mr.load_predictions(config.PREDICTIONS_DIR / f"{key}_test_eval.jsonl")
        if loaded:
            test_preds[key] = loaded
    result = analyze(blocks, label_map, test_preds, {
        f"the {len(trained)} blocks actually trained on": trained,
        f"the full training split ({len(train_split)} blocks)": set(train_split)})

    for ref_name, entry in result["references"].items():
        print(f"\nReference: {ref_name}\n  test blocks with an identical sequence: {entry['test_blocks_with_identical_sequence']} of {result['test_examples']}")
        print(f"  {'model':10s} {'subset':28s} {'n':>5s} {'anom':>5s} {'acc':>6s} {'prec':>6s} {'rec':>6s} {'f1':>6s}  FP FN")
        for model, rows in entry["models"].items():
            for name, r in rows.items():
                if r["n"]:
                    c = r["confusion"]
                    print(f"  {model:10s} {name:28s} {r['n']:5d} {r['anomalies']:5d} {r['accuracy']:6.3f} {r['precision']:6.3f} {r['recall']:6.3f} {r['f1']:6.3f}  {c['fp']:3d} {c['fn']:3d}")
    out = config.RESULTS_DIR / "duplicate_analysis.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nSaved", out)


if __name__ == "__main__":
    main()