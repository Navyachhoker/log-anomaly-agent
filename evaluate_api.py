"""Zero-shot baseline: classify the evaluation snippets with the API LLM (same prompt as the SLMs).

    python evaluate_api.py --data data/processed/test_eval.jsonl --out results/predictions/zeroshot_test_eval.jsonl

Resumable: snippets already in --out are skipped, failed calls are retried on the next run.
Use --sleep to stay inside free-tier rate limits.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

import config
import llm
from logtext import build_messages


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(config.PROCESSED_DIR / "test_eval.jsonl"))
    p.add_argument("--out", default=str(config.PREDICTIONS_DIR / "zeroshot_test_eval.jsonl"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--sleep", type=float, default=0.0, help="seconds to wait between calls")
    args = p.parse_args()

    if not llm.is_configured():
        sys.exit("LLM not configured. Fill LOGAGENT_LLM_* in .env first (see .env.example).")
    records = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    if args.limit:
        records = records[: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["block_id"] for line in open(out, encoding="utf-8")} if out.exists() else set()
    todo = [r for r in records if r["block_id"] not in done]
    print(f"{len(done)} already done, {len(todo)} to do; provider={llm.settings()['provider']} model={llm.settings()['model']}")

    failures = 0
    with open(out, "a", encoding="utf-8") as f:
        for record in tqdm(todo):
            system, user = (m["content"] for m in build_messages(record["snippet"]))
            try:
                reply = llm.chat(system, user)
            except llm.LLMError as error:
                failures += 1
                print(f"\nfailed {record['block_id']}: {error}")
                continue
            f.write(json.dumps({"block_id": record["block_id"], "label": record["label"], "generated": reply.text,
                                "latency_s": reply.latency_s, "input_tokens": reply.input_tokens,
                                "output_tokens": reply.output_tokens}) + "\n")
            f.flush()
            if args.sleep:
                time.sleep(args.sleep)

    price_in, price_out = llm.prices_per_million_tokens()
    meta = {**{k: v for k, v in llm.settings().items() if k not in ("api_key",)},
            "hardware": "API (remote provider)", "price_in_per_mtok_usd": price_in, "price_out_per_mtok_usd": price_out,
            "data": args.data, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Finished with {failures} failed call(s). Run again to retry failures." if failures else "Finished.")


if __name__ == "__main__":
    main()