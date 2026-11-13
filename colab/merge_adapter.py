"""Merge the LoRA adapter into the full-precision base model and save it for CPU inference.

    python colab/merge_adapter.py --adapter <drive>/run1/adapter --out <drive>/merged
"""
import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    model = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, torch_dtype=torch.float16)  # on CPU, not 4-bit
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(config.BASE_MODEL).save_pretrained(args.out)
    print("Merged model saved to", args.out)


if __name__ == "__main__":
    main()