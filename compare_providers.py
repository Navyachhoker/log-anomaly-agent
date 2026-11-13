"""Compare diagnostic-LLM providers on VALIDATION anomalies (never the test set).

Run once per provider: edit LOGAGENT_LLM_* in .env (or set them in the PowerShell session), then
    python compare_providers.py --n 10
Each run is stored in results/provider_comparison.json under "<provider>:<model>".
"""
import argparse
import json
import time

import numpy as np

import agent
import config
import llm


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(config.PROCESSED_DIR / "val_eval.jsonl"))
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--sleep", type=float, default=0.0)
    args = p.parse_args()
    if not llm.is_configured():
        raise SystemExit("LLM not configured. Fill LOGAGENT_LLM_* in .env first.")

    records = [r for r in map(json.loads, open(args.data, encoding="utf-8")) if r["label"] == "Anomaly"][: args.n]
    rows = []
    for record in records:
        diagnosis, usage, error = agent.request_diagnosis(record["snippet"])
        rows.append({"block_id": record["block_id"], "valid": diagnosis is not None, "first_try_valid": diagnosis is not None and usage["calls"] == 1,
                     "error": error, "usage": usage, "cause": diagnosis.cause if diagnosis else None,
                     "suggested_fix": diagnosis.suggested_fix if diagnosis else None})
        print(("OK   " if diagnosis else "FAIL ") + record["block_id"], error or "")
        time.sleep(args.sleep)

    used = [r["usage"] for r in rows if r["usage"] and r["usage"]["calls"]]
    latency = np.array([u["latency_s"] for u in used]) if used else np.array([0.0])
    costs = [llm.cost_usd(u["input_tokens"], u["output_tokens"]) for u in used]
    s = llm.settings()
    summary = {"provider": s["provider"], "model": s["model"], "n": len(rows),
               "valid_json_rate": sum(r["valid"] for r in rows) / len(rows),
               "valid_on_first_try_rate": sum(r["first_try_valid"] for r in rows) / len(rows),
               "latency_s": {"mean": round(float(latency.mean()), 2), "median": round(float(np.median(latency)), 2), "p95": round(float(np.percentile(latency, 95)), 2)},
               "mean_input_tokens": float(np.mean([u["input_tokens"] / u["calls"] for u in used])) if used else None,
               "mean_output_tokens": float(np.mean([u["output_tokens"] / u["calls"] for u in used])) if used else None,
               "mean_cost_per_call_usd": float(np.mean(costs)) if costs and None not in costs else None,
               "prices_per_mtok_usd": list(llm.prices_per_million_tokens()),
               "examples": [{k: r[k] for k in ("cause", "suggested_fix")} for r in rows[:2]],
               "measured_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(json.dumps(summary, indent=2))
    path = config.RESULTS_DIR / "provider_comparison.json"
    config.RESULTS_DIR.mkdir(exist_ok=True)
    runs = json.loads(path.read_text()) if path.exists() else {}
    runs[f"{s['provider']}:{s['model']}"] = summary
    path.write_text(json.dumps(runs, indent=2))
    print("Saved", path)


if __name__ == "__main__":
    main()