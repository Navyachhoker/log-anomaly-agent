"""QLoRA fine-tuning of Qwen2.5-1.5B-Instruct (4-bit) for HDFS log anomaly classification.

Runs on Google Colab with a GPU. Checkpoints are written to --output-dir (put it on Google Drive).
Running the same command again resumes from the latest checkpoint.

Pilot (pipeline test, resume test, throughput):
    python colab/train.py --data-dir /content/data/processed/pilot --output-dir /content/drive/MyDrive/logagent/pilot_run --max-steps 60 --save-steps 20
Real run:
    python colab/train.py --data-dir /content/data/processed --output-dir /content/drive/MyDrive/logagent/run1 --max-per-class 3000
"""
import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import peft
import torch
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from logtext import build_messages  # noqa: E402


def check_gpu():
    if not torch.cuda.is_available():
        sys.exit("No GPU found. In Colab: Runtime > Change runtime type > pick a T4 GPU, then run again. "
                 "If none is available, try again later or use Kaggle's free GPU.")
    props = torch.cuda.get_device_properties(0)
    use_bf16 = props.major >= 8  # a T4 (compute capability 7.5) has no bf16; it uses fp16
    print(f"GPU: {props.name} | {props.total_memory / 2**30:.1f} GB | compute dtype: {'bf16' if use_bf16 else 'fp16'}")
    return props.name, round(props.total_memory / 2**30, 1), use_bf16


def read_jsonl(path, cap_per_class=0):
    records, counts = [], {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if cap_per_class and counts.get(record["label"], 0) >= cap_per_class:
                continue
            counts[record["label"]] = counts.get(record["label"], 0) + 1
            records.append(record)
    return records


def encode(tokenizer, record, max_len):
    """Prompt tokens are masked (-100); the loss is only on the answer word and end-of-turn token."""
    prompt = tokenizer.apply_chat_template(build_messages(record["snippet"]), tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(record["label"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(answer_ids) > max_len:
        return None
    return {"input_ids": prompt_ids + answer_ids, "labels": [-100] * len(prompt_ids) + answer_ids}


class Encoded(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collator(pad_id):
    def collate(batch):
        longest = max(len(x["input_ids"]) for x in batch)
        pad = lambda seq, value, n: seq + [value] * (n - len(seq))  # noqa: E731
        return {
            "input_ids": torch.tensor([pad(x["input_ids"], pad_id, longest) for x in batch]),
            "attention_mask": torch.tensor([pad([1] * len(x["input_ids"]), 0, longest) for x in batch]),
            "labels": torch.tensor([pad(x["labels"], -100, longest) for x in batch]),
        }
    return collate


def build_dataset(tokenizer, records, max_len, name):
    items = [e for e in (encode(tokenizer, r, max_len) for r in records) if e is not None]
    skipped = len(records) - len(items)
    print(f"{name}: {len(items)} samples ({skipped} skipped for exceeding {max_len} tokens)")
    return Encoded(items), skipped


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="folder with train.jsonl and val.jsonl")
    p.add_argument("--output-dir", required=True, help="checkpoints + final adapter (use a Google Drive path)")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--max-steps", type=int, default=-1, help="stop after N steps (pilot); -1 = use epochs")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--max-per-class", type=int, default=0, help="use at most N training samples per class (0 = all)")
    p.add_argument("--eval-per-class", type=int, default=100, help="validation samples per class used for eval loss")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--save-steps", type=int, default=100)
    args = p.parse_args()

    gpu_name, gpu_gb, use_bf16 = check_gpu()
    random.seed(config.SEED)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_records = read_jsonl(data_dir / "train.jsonl", args.max_per_class)
    val_all = read_jsonl(data_dir / "val.jsonl")
    val_records = []
    for label in ("Normal", "Anomaly"):
        pool = [r for r in val_all if r["label"] == label]
        val_records += random.Random(config.SEED).sample(pool, min(args.eval_per_class, len(pool)))
    train_ds, skipped = build_dataset(tokenizer, train_records, args.max_seq_len, "train")
    eval_ds, _ = build_dataset(tokenizer, val_records, args.max_seq_len, "eval (balanced subset of validation)")

    dtype = torch.bfloat16 if use_bf16 else torch.float16
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, quantization_config=bnb,
                                                 device_map={"": 0}, torch_dtype=dtype)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    for param in model.parameters():  # trainable (LoRA) weights must be fp32 for fp16 mixed precision
        if param.requires_grad:
            param.data = param.data.float()
    model.config.use_cache = False
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(steps_per_epoch * args.epochs)
    print(f"Optimizer steps per epoch: {steps_per_epoch} | total steps planned: {total_steps}")

    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_steps=20, logging_steps=10, save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
        eval_strategy="steps", eval_steps=args.save_steps, fp16=not use_bf16, bf16=use_bf16,
        optim="paged_adamw_8bit", gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, report_to="none",
        remove_unused_columns=False, seed=config.SEED,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=make_collator(tokenizer.pad_token_id))

    last = get_last_checkpoint(str(out / "checkpoints")) if (out / "checkpoints").exists() else None
    print(f"Resuming from {last}" if last else "Starting a new run (no checkpoint found)")
    started = time.time()
    result = trainer.train(resume_from_checkpoint=last)

    adapter_dir = out / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    (out / "loss_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))

    train_file = data_dir / "train.jsonl"
    manifest = {
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "resumed_from_checkpoint": last,
        "args": vars(args), "gpu": gpu_name, "gpu_memory_gb": gpu_gb,
        "compute_dtype": "bf16" if use_bf16 else "fp16",
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__, "peft": peft.__version__},
        "train_samples": len(train_ds), "skipped_too_long": skipped, "eval_samples": len(eval_ds),
        "train_file_sha256": hashlib.sha256(train_file.read_bytes()).hexdigest(),
        "total_steps_planned": total_steps, "this_session_seconds": round(time.time() - started, 1),
        "train_metrics_this_session": result.metrics,
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    (out / "train_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    sps = result.metrics.get("train_samples_per_second")
    if sps:
        print(f"\nMeasured throughput: {sps:.2f} samples/s -> about {len(train_ds) * 3 / sps / 60:.0f} minutes for 3 epochs of this training set.")
    print(f"Adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()