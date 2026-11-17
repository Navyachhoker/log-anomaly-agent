"""List the test blocks the fine-tuned model got wrong (for the error discussion in the report).

    python error_analysis.py            (writes results\\error_analysis.txt)
Only reads existing predictions; nothing is tuned.
"""
import json

import config
import make_report as mr
from logtext import parse_label


def find_errors(preds, snippets, threshold=0.5):
    """False alarms and misses of one model: [(kind, record, snippet record)]."""
    errors = []
    for block, rec in sorted(preds.items()):
        predicted = parse_label(rec["generated"])
        if predicted is None:
            predicted = "Anomaly" if rec["label"] == "Normal" else "Normal"  # unparseable counts as an error
        if predicted != rec["label"]:
            errors.append(("FALSE ALARM (normal flagged as anomaly)" if predicted == "Anomaly" else "MISSED ANOMALY", rec, snippets.get(block)))
    return errors


def describe(kind, rec, snippet):
    lines = snippet["snippet"].split("\n") if snippet else []
    warn = [line for line in lines if not line.startswith("INFO")]
    out = [f"--- {kind}: {rec['block_id']}  p_anomaly={rec['p_anomaly']:.4f}  raw lines={snippet['num_lines'] if snippet else '?'}  truncated={snippet['truncated'] if snippet else '?'}",
           f"    WARN/ERROR lines: {len(warn)}"]
    out += [f"      {line}" for line in warn[:6]]
    out.append("    last 5 lines:")
    out += [f"      {line}" for line in lines[-5:]]
    return "\n".join(out)


def main():
    preds = mr.load_predictions(config.PREDICTIONS_DIR / "finetuned_test_eval.jsonl")
    snippets = {r["block_id"]: r for r in map(json.loads, open(config.PROCESSED_DIR / "test_eval.jsonl", encoding="utf-8"))}
    errors = find_errors(preds, snippets)
    text = f"Fine-tuned model, test_eval: {len(errors)} errors out of {len(preds)} examples\n\n" + "\n\n".join(describe(*e) for e in errors)
    print(text)
    (config.RESULTS_DIR / "error_analysis.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()