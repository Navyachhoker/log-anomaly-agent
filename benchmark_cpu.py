"""Benchmark the merged model on THIS laptop's CPU: load time, memory, latency, agreement with Colab.

    python benchmark_cpu.py --dtype float32
    python benchmark_cpu.py --dtype bfloat16
    python benchmark_cpu.py --dtype int8        (experimental: dynamic int8 quantization)

Plug the laptop in and close heavy programs first. Results are merged into results/cpu_benchmark.json.
"""
import argparse
import json
import platform
import time

import numpy as np
import psutil
import torch

import config
from classifier import SlmClassifier


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default=str(config.MODEL_DIR))
    p.add_argument("--data", default=str(config.PROCESSED_DIR / "val_eval.jsonl"))
    p.add_argument("--reference", default=str(config.PREDICTIONS_DIR / "finetuned_val_eval.jsonl"),
                   help="Colab predictions on the same file, used to check that the merged CPU model agrees")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "int8"])
    p.add_argument("--n-anomaly", type=int, default=25)
    p.add_argument("--n-normal", type=int, default=25)
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    process = psutil.Process()
    rss_start = process.memory_info().rss
    classifier = SlmClassifier(args.model_dir, dtype=args.dtype, threshold=0.5)
    rss_loaded = process.memory_info().rss
    print(f"Loaded in {classifier.load_seconds:.1f}s; memory after loading: {rss_loaded / 2**30:.2f} GB")

    reference = {}
    try:
        reference = {r["block_id"]: r for r in map(json.loads, open(args.reference, encoding="utf-8"))}
    except FileNotFoundError:
        print("No reference predictions found: agreement with Colab will not be checked.")
    records = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    chosen = [r for r in records if r["label"] == "Anomaly"][: args.n_anomaly] + [r for r in records if r["label"] == "Normal"][: args.n_normal]

    for _ in range(3):  # warm-up, not counted
        classifier.classify_snippet(chosen[0]["snippet"])
    latencies, peak_rss, diffs, agree, compared = [], rss_loaded, [], 0, 0
    for record in chosen:
        result = classifier.classify_snippet(record["snippet"])
        latencies.append(result["latency_s"])
        peak_rss = max(peak_rss, process.memory_info().rss)
        ref = reference.get(record["block_id"])
        if ref:
            compared += 1
            diffs.append(abs(result["p_anomaly"] - ref["p_anomaly"]))
            agree += (result["p_anomaly"] >= 0.5) == (ref["p_anomaly"] >= 0.5)

    lat = np.array(latencies)
    summary = {
        "dtype": args.dtype, "cpu": platform.processor(), "logical_cores": psutil.cpu_count(), "torch_threads": torch.get_num_threads(),
        "torch": torch.__version__, "model_dir": args.model_dir, "load_seconds": round(classifier.load_seconds, 2),
        "memory_gb_after_load": round(rss_loaded / 2**30, 2), "peak_memory_gb": round(peak_rss / 2**30, 2),
        "memory_gb_before_load": round(rss_start / 2**30, 2), "n_measured": len(lat),
        "latency_s": {"mean": round(float(lat.mean()), 3), "median": round(float(np.median(lat)), 3),
                      "p95": round(float(np.percentile(lat, 95)), 3), "max": round(float(lat.max()), 3)},
        "agreement_with_colab": {"compared": compared, "label_agreement": agree / compared if compared else None,
                                 "mean_abs_p_anomaly_difference": float(np.mean(diffs)) if diffs else None,
                                 "max_abs_p_anomaly_difference": float(np.max(diffs)) if diffs else None},
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(json.dumps(summary, indent=2))
    path = config.RESULTS_DIR / "cpu_benchmark.json"
    config.RESULTS_DIR.mkdir(exist_ok=True)
    all_runs = json.loads(path.read_text()) if path.exists() else {}
    all_runs[args.dtype] = summary
    path.write_text(json.dumps(all_runs, indent=2))
    print("Saved", path)


if __name__ == "__main__":
    main()