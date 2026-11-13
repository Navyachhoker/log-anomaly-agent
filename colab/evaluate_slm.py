"""Evaluate Qwen2.5-1.5B-Instruct (4-bit) on a JSONL file, with or without the LoRA adapter.

One greedy generation per snippet at batch size 1. For every snippet it stores the generated text,
the probability of "Anomaly" (softmax over the first-token logits of the words Normal and Anomaly),
and the latency. Resumable: running it again continues where it stopped.

    python colab/evaluate_slm.py --data /content/data/processed/test_eval.jsonl --adapter <adapter dir> --out <drive>/predictions/finetuned_test_eval.jsonl
    python colab/evaluate_slm.py --data /content/data/processed/test_eval.jsonl --out <drive>/predictions/untuned_test_eval.jsonl
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from logtext import build_messages  # noqa: E402

WARMUP = 3  # the first calls are slower (GPU warm-up) and are flagged so reports can exclude them


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--adapter", default=None, help="LoRA adapter folder; omit for the untuned baseline")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=4)
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("No GPU found. Change the Colab runtime to a GPU runtime.")
    props = torch.cuda.get_device_properties(0)
    dtype = torch.bfloat16 if props.major >= 8 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, quantization_config=bnb,
                                                 device_map={"": 0}, torch_dtype=dtype)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    normal_id = tokenizer.encode("Normal", add_special_tokens=False)[0]
    anomaly_id = tokenizer.encode("Anomaly", add_special_tokens=False)[0]
    assert normal_id != anomaly_id, "the two label words share their first token"

    records = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    if args.limit:
        records = records[: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["block_id"] for line in open(out, encoding="utf-8")} if out.exists() else set()
    todo = [r for r in records if r["block_id"] not in done]
    print(f"{len(done)} already done, {len(todo)} to do")

    with open(out, "a", encoding="utf-8") as f:
        for i, record in enumerate(tqdm(todo)):
            prompt = tokenizer.apply_chat_template(build_messages(record["snippet"]), tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                result = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                        repetition_penalty=1.0, output_scores=True, return_dict_in_generate=True,
                                        pad_token_id=tokenizer.pad_token_id)
            torch.cuda.synchronize()
            latency = time.perf_counter() - start
            n_in = inputs["input_ids"].shape[1]
            text = tokenizer.decode(result.sequences[0, n_in:], skip_special_tokens=True)
            two = torch.stack([result.scores[0][0, normal_id], result.scores[0][0, anomaly_id]]).float()
            p_anomaly = float(torch.softmax(two, dim=0)[1])
            f.write(json.dumps({"block_id": record["block_id"], "label": record["label"], "generated": text,
                                "p_anomaly": p_anomaly, "latency_s": latency, "input_tokens": n_in,
                                "warmup": i < WARMUP}) + "\n")
            f.flush()

    meta = {"model": config.BASE_MODEL, "adapter": args.adapter, "hardware": f"{props.name} (Colab GPU)",
            "precision": "4-bit NF4 base" + (" + LoRA adapter" if args.adapter else ""),
            "data": args.data, "batch_size": 1, "torch": torch.__version__, "python": platform.python_version(),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2))
    print("Done:", out)


if __name__ == "__main__":
    main()