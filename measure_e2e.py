"""End-to-end measurement on the laptop: classifier (CPU) + LangGraph agent + diagnostic LLM (API).

    python measure_e2e.py --n-anomaly 20 --n-normal 10

Uses VALIDATION examples. Writes results/e2e.json (latency split by component, diagnosis cost per call),
results/diagnosis_outputs.jsonl, and a blank rating sheet results/diagnosis_ratings.csv for the manual
1-5 rating by two reviewers (plausibility, relevance, actionability) required by the report.
"""
import argparse
import csv
import json
import platform
import time

import numpy as np

import agent
import config
import llm
from classifier import SlmClassifier


def stats(values):
    a = np.array([v for v in values if v is not None], dtype=float)
    return {} if a.size == 0 else {"n": int(a.size), "mean_ms": round(float(a.mean()), 1), "median_ms": round(float(np.median(a)), 1),
                                    "p95_ms": round(float(np.percentile(a, 95)), 1)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(config.PROCESSED_DIR / "val_eval.jsonl"))
    p.add_argument("--n-anomaly", type=int, default=20)
    p.add_argument("--n-normal", type=int, default=10)
    p.add_argument("--sleep", type=float, default=0.0)
    args = p.parse_args()

    records = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    chosen = [r for r in records if r["label"] == "Anomaly"][: args.n_anomaly] + [r for r in records if r["label"] == "Normal"][: args.n_normal]
    classifier = SlmClassifier()
    graph = agent.build_agent(classifier.classify)
    graph.invoke({"log": chosen[0]["snippet"]})  # warm-up (not counted)

    reports = []
    for record in chosen:
        report = agent.run_agent(graph, record["snippet"])
        reports.append({"block_id": record["block_id"], "true_label": record["label"], **report})
        print(f"{record['label']:8s} -> {report['label']:8s} total {report['latency_ms']['total']:8.1f} ms  "
              f"{'(diagnosis error: ' + report['diagnosis_error'] + ')' if report['diagnosis_error'] else ''}")
        time.sleep(args.sleep)

    flagged = [r for r in reports if r["label"] == "Anomaly"]
    diagnosed = [r for r in flagged if r["cause"]]
    per_call_costs = [llm.cost_usd(r["llm_usage"]["input_tokens"], r["llm_usage"]["output_tokens"]) / r["llm_usage"]["calls"]
                      for r in diagnosed if r["llm_usage"] and r["llm_usage"]["calls"]]
    summary = {
        "hardware_classifier": f"laptop CPU ({platform.processor()}), dtype {classifier.dtype}", "diagnosis_provider": llm.settings()["provider"],
        "diagnosis_model": llm.settings()["model"], "examples_run": len(reports), "flagged_as_anomaly": len(flagged),
        "diagnosed_successfully": len(diagnosed), "latency_anomalous_flagged": {
            "classify": stats(r["latency_ms"]["classify"] for r in flagged), "diagnose": stats(r["latency_ms"]["diagnose"] for r in flagged),
            "total": stats(r["latency_ms"]["total"] for r in flagged)},
        "latency_normal_flagged_total": stats(r["latency_ms"]["total"] for r in reports if r["label"] == "Normal"),
        "diagnosis_cost_per_call_usd": float(np.mean(per_call_costs)) if per_call_costs and None not in per_call_costs else None,
        "mean_llm_input_tokens": float(np.mean([r["llm_usage"]["input_tokens"] for r in diagnosed])) if diagnosed else None,
        "mean_llm_output_tokens": float(np.mean([r["llm_usage"]["output_tokens"] for r in diagnosed])) if diagnosed else None,
        "note": "Classifier on laptop CPU and diagnosis over the internet; these two latencies are reported separately because they run on different systems.",
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "e2e.json").write_text(json.dumps(summary, indent=2))
    (config.RESULTS_DIR / "diagnosis_outputs.jsonl").write_text("\n".join(json.dumps(r) for r in reports) + "\n", encoding="utf-8")

    sheet = config.RESULTS_DIR / "diagnosis_ratings.csv"
    if sheet.exists():
        sheet = config.RESULTS_DIR / "diagnosis_ratings_new.csv"
        print(f"{config.RESULTS_DIR / 'diagnosis_ratings.csv'} already exists (may contain your ratings); writing {sheet.name} instead")
    columns = ["block_id", "snippet", "cause", "suggested_fix"] + [f"r{i}_{c}" for i in (1, 2) for c in ("plausibility", "relevance", "actionability")]
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for r in diagnosed:
            writer.writerow([r["block_id"], r["snippet"], r["cause"], " | ".join(r["suggested_fix"])] + [""] * 6)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()