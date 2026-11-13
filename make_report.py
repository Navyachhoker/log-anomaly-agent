"""Compute every evaluation number from the prediction files -> results/evaluation.json.

Inputs (results/predictions/, produced by colab/evaluate_slm.py and evaluate_api.py):
    finetuned_test_eval.jsonl  untuned_test_eval.jsonl  zeroshot_test_eval.jsonl   (same test examples)
    finetuned_val_eval.jsonl                                                       (threshold tuning, validation only)
    finetuned_test_full.jsonl                                                      (optional, supplementary)
Also reads results/e2e.json and results/diagnosis_ratings.csv when they exist.

    python make_report.py --gpu-hourly-usd 0.35
Nothing is invented here: every number comes from the files above.
"""
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

import config
from logtext import parse_label

MODELS = [("finetuned", "Fine-tuned Qwen2.5-1.5B (QLoRA)"),
          ("untuned", "Untuned Qwen2.5-1.5B-Instruct"),
          ("zeroshot", "Zero-shot API LLM")]


# ------------------------------------------------------------------ metrics
def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Binary metrics with Anomaly (1) as the positive class."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": (tp + tn) / len(y_true), "precision": precision, "recall": recall, "f1": f1,
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}}


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000, seed: int = config.SEED) -> dict:
    """95% bootstrap intervals (resampling examples) for accuracy, precision, recall and F1."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), size=(n_boot, len(y_true)), dtype=np.int32)
    yt, yp = y_true[idx], y_pred[idx]
    tp = ((yt == 1) & (yp == 1)).sum(1).astype(float)
    fp = ((yt == 0) & (yp == 1)).sum(1).astype(float)
    fn = ((yt == 1) & (yp == 0)).sum(1).astype(float)
    tn = ((yt == 0) & (yp == 0)).sum(1).astype(float)
    safe = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b > 0)  # noqa: E731
    precision, recall = safe(tp, tp + fp), safe(tp, tp + fn)
    values = {"accuracy": (tp + tn) / len(y_true), "precision": precision, "recall": recall,
              "f1": safe(2 * precision * recall, precision + recall)}
    return {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for k, v in values.items()}


def full_metrics(y_true, y_pred, n_boot=1000) -> dict:
    return {**score(y_true, y_pred), "ci95": bootstrap_ci(y_true, y_pred, n_boot)}


def tune_threshold(y_true: np.ndarray, p_anomaly: np.ndarray) -> float:
    """Threshold in 0.01..0.99 with the best F1 (ties: closest to 0.5). Use validation data only."""
    best_key, best_t = None, 0.5
    for t in np.round(np.arange(0.01, 1.0, 0.01), 2):
        key = (score(y_true, (p_anomaly >= t).astype(np.int8))["f1"], -abs(float(t) - 0.5))
        if best_key is None or key > best_key:
            best_key, best_t = key, float(t)
    return best_t


def predictions_from_text(records, y_true):
    """Parse generated text. Unparseable answers are counted as errors (the opposite of the truth)."""
    parsed = [parse_label(r["generated"]) for r in records]
    y_pred = np.array([(1 - t) if p is None else int(p == "Anomaly") for p, t in zip(parsed, y_true)], dtype=np.int8)
    return y_pred, sum(p is None for p in parsed)


# ------------------------------------------------------------------ files
def load_predictions(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return {r["block_id"]: r for r in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


def load_meta(path: Path) -> dict:
    meta = Path(str(path) + ".meta.json")
    return json.loads(meta.read_text()) if meta.exists() else {}


def latency_stats(records) -> dict:
    values = np.array([r["latency_s"] for r in records if not r.get("warmup")])
    if values.size == 0:
        return {}
    return {"n": int(values.size), "mean_s": round(float(values.mean()), 4), "median_s": round(float(np.median(values)), 4),
            "p95_s": round(float(np.percentile(values, 95)), 4), "total_s": round(float(values.sum()), 2)}


def model_cost(key: str, records, meta: dict, latency: dict, gpu_hourly_usd, actual_spend_usd) -> dict:
    n = len(records)
    if key == "zeroshot":
        tokens_in = sum(r.get("input_tokens", 0) for r in records)
        tokens_out = sum(r.get("output_tokens", 0) for r in records)
        price_in, price_out = meta.get("price_in_per_mtok_usd"), meta.get("price_out_per_mtok_usd")
        estimated = None
        if price_in is not None and price_out is not None:
            estimated = (tokens_in * price_in + tokens_out * price_out) / 1e6
        return {"input_tokens": tokens_in, "output_tokens": tokens_out,
                "estimated_commercial_usd_total": estimated,
                "estimated_commercial_usd_per_1000": estimated / n * 1000 if estimated is not None else None,
                "actual_spend_usd": actual_spend_usd,
                "assumptions": f"list prices from .env at run time ({price_in} in / {price_out} out USD per 1M tokens); actual spend as supplied by the user (free tiers may make it 0)"}
    estimated = None
    if gpu_hourly_usd is not None and latency:
        estimated = latency["total_s"] / 3600 * gpu_hourly_usd
    per_1000 = estimated / latency["n"] * 1000 if estimated is not None else None
    return {"estimated_commercial_usd_total": estimated, "estimated_commercial_usd_per_1000": per_1000,
            "actual_spend_usd": 0.0,
            "assumptions": f"estimate = measured inference seconds x assumed GPU price of {gpu_hourly_usd} USD/hour (inference time only); actual spend 0 on the free Colab tier"}


# ------------------------------------------------------------------ report
def build_report(pred_dir: Path, results_dir: Path, gpu_hourly_usd=None, actual_spend_zeroshot_usd=None) -> dict:
    files = {key: pred_dir / f"{key}_test_eval.jsonl" for key, _ in MODELS}
    preds = {key: load_predictions(path) for key, path in files.items()}
    missing = [f"{key}_test_eval.jsonl" for key, p in preds.items() if p is None]
    if missing:
        sys.exit(f"Missing prediction files in {pred_dir}: {', '.join(missing)}")

    common = sorted(set.intersection(*[set(p) for p in preds.values()]))
    dropped = {key: len(p) - len(common) for key, p in preds.items()}
    if not common:
        sys.exit("The prediction files share no examples.")
    for block in common:
        if len({preds[k][block]["label"] for k in preds}) != 1:
            sys.exit(f"Ground-truth labels disagree between files for {block}")
    y_true = np.array([preds["finetuned"][b]["label"] == "Anomaly" for b in common], dtype=np.int8)

    # threshold tuned on VALIDATION predictions only
    threshold_info = None
    val = load_predictions(pred_dir / "finetuned_val_eval.jsonl")
    if val:
        ids = sorted(val)
        yv = np.array([val[b]["label"] == "Anomaly" for b in ids], dtype=np.int8)
        pv = np.array([val[b]["p_anomaly"] for b in ids])
        tuned = tune_threshold(yv, pv)
        threshold_info = {"threshold": tuned, "tuned_on": "validation subset (val_eval)", "val_examples": len(ids),
                          "val_anomalies": int(yv.sum()), "val_f1_at_0.5": score(yv, (pv >= 0.5).astype(np.int8))["f1"],
                          "val_f1_at_tuned": score(yv, (pv >= tuned).astype(np.int8))["f1"],
                          "tuned_at": datetime.now().isoformat(timespec="seconds")}
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "threshold.json").write_text(json.dumps(threshold_info, indent=2))

    models = []
    for key, name in MODELS:
        records = [preds[key][b] for b in common]
        meta = load_meta(files[key])
        y_text, unparseable = predictions_from_text(records, y_true)
        entry = {"key": key, "name": name, "n": len(records), "hardware": meta.get("hardware"),
                 "precision": meta.get("precision") or meta.get("model"),
                 "unparseable": {"count": int(unparseable), "rate": unparseable / len(records)},
                 "as_generated": full_metrics(y_true, y_text)}
        if "p_anomaly" in records[0]:
            p = np.array([r["p_anomaly"] for r in records])
            entry["scored_at_0.5"] = full_metrics(y_true, (p >= 0.5).astype(np.int8))
            if key == "finetuned" and threshold_info:
                entry["scored_at_tuned_threshold"] = {"threshold": threshold_info["threshold"],
                                                      **full_metrics(y_true, (p >= threshold_info["threshold"]).astype(np.int8))}
        entry["latency"] = latency_stats(records)
        entry["cost"] = model_cost(key, records, meta, entry["latency"], gpu_hourly_usd, actual_spend_zeroshot_usd)
        models.append(entry)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "test_examples": {"n": len(common), "anomalies": int(y_true.sum()), "anomaly_rate": float(y_true.mean()),
                          "dropped_because_not_in_all_three_files": dropped},
        "threshold": threshold_info, "models": models, "supplementary": {}, "end_to_end": None,
        "diagnosis_ratings": None,
        "notes": ["Anomaly is the positive class. Unparseable answers are counted as errors.",
                  "95% intervals come from a 1000-sample bootstrap over the test examples.",
                  "Latency is measured at batch size 1 excluding warm-up calls and is NOT comparable across hardware: T4 GPU vs API vs laptop CPU.",
                  "The decision threshold was tuned on validation data only; the test examples were never used for tuning."],
    }

    full = load_predictions(pred_dir / "finetuned_test_full.jsonl")
    if full:
        ids = sorted(full)
        yf = np.array([full[b]["label"] == "Anomaly" for b in ids], dtype=np.int8)
        pf = np.array([full[b]["p_anomaly"] for b in ids])
        sup = {"n": len(ids), "anomalies": int(yf.sum()), "scored_at_0.5": full_metrics(yf, (pf >= 0.5).astype(np.int8), 200)}
        if threshold_info:
            sup["scored_at_tuned_threshold"] = full_metrics(yf, (pf >= threshold_info["threshold"]).astype(np.int8), 200)
        report["supplementary"]["finetuned_on_full_test_set"] = sup

    e2e_path = results_dir / "e2e.json"
    if e2e_path.exists():
        e2e = json.loads(e2e_path.read_text())
        ft = next(m for m in models if m["key"] == "finetuned")
        key_metrics = ft.get("scored_at_tuned_threshold") or ft["scored_at_0.5"]
        flagged_fraction = (key_metrics["confusion"]["tp"] + key_metrics["confusion"]["fp"]) / len(common)
        slm_per_log = (ft["cost"]["estimated_commercial_usd_per_1000"] or 0) / 1000 if ft["cost"]["estimated_commercial_usd_per_1000"] is not None else None
        diag = e2e.get("diagnosis_cost_per_call_usd")
        total = slm_per_log + flagged_fraction * diag if slm_per_log is not None and diag is not None else None
        e2e["cost_formula"] = {"fraction_flagged_as_anomaly": flagged_fraction, "classification_cost_per_log_usd": slm_per_log,
                               "diagnosis_cost_per_llm_call_usd": diag, "total_cost_per_log_usd": total,
                               "total_cost_per_1000_logs_usd": total * 1000 if total is not None else None,
                               "formula": "Total = classification cost + fraction of anomalous logs x diagnosis cost per LLM call"}
        report["end_to_end"] = e2e

    ratings = results_dir / "diagnosis_ratings.csv"
    if ratings.exists():
        rows = []
        for row in csv.DictReader(ratings.open(encoding="utf-8")):
            try:
                rows.append({k: float(v) for k, v in row.items() if k.startswith("r1_") or k.startswith("r2_")})
            except (ValueError, TypeError):
                continue
        if rows:
            criteria = ["plausibility", "relevance", "actionability"]
            report["diagnosis_ratings"] = {
                "rated_examples": len(rows),
                "mean_by_criterion": {c: {"reviewer1": float(np.mean([r[f"r1_{c}"] for r in rows])),
                                          "reviewer2": float(np.mean([r[f"r2_{c}"] for r in rows]))} for c in criteria},
                "mean_absolute_difference_between_reviewers": float(np.mean([abs(r[f"r1_{c}"] - r[f"r2_{c}"]) for r in rows for c in criteria])),
                "exact_agreement_share": float(np.mean([r[f"r1_{c}"] == r[f"r2_{c}"] for r in rows for c in criteria]))}

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "evaluation.json").write_text(json.dumps(report, indent=2))
    return report


def print_summary(report: dict) -> None:
    t = report["test_examples"]
    print(f"\nTest examples: {t['n']} ({t['anomalies']} anomalies). Dropped (not in all three files): {t['dropped_because_not_in_all_three_files']}")
    print(f"{'model':36s} {'acc':>6s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'unparse':>8s} {'med.lat(s)':>11s}  hardware")
    for m in report["models"]:
        g = m["as_generated"]
        print(f"{m['name']:36s} {g['accuracy']:6.3f} {g['precision']:6.3f} {g['recall']:6.3f} {g['f1']:6.3f} "
              f"{m['unparseable']['rate']:8.3f} {m['latency'].get('median_s', float('nan')):11.3f}  {m['hardware']}")
    if report["threshold"]:
        print(f"\nTuned threshold (validation only): {report['threshold']['threshold']}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu-hourly-usd", type=float, default=None, help="assumed commercial price of a T4-class GPU hour, for the estimated SLM cost")
    p.add_argument("--actual-spend-zeroshot-usd", type=float, default=None, help="what you actually paid the API provider (0 on a free tier)")
    args = p.parse_args()
    report = build_report(config.PREDICTIONS_DIR, config.RESULTS_DIR, args.gpu_hourly_usd, args.actual_spend_zeroshot_usd)
    print_summary(report)
    print(f"\nSaved {config.RESULTS_DIR / 'evaluation.json'}")


if __name__ == "__main__":
    main()