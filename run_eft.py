#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified entry script for EFT (synthetic data) experiments.

This script dispatches to the underlying task-specific training
and evaluation scripts for Amazon, Yelp, Yahoo and MNLI, and for
both LLaMA and Qwen model families.

Example usages:

- Train Amazon sentiment (LLaMA):
  python run_eft.py --task amazon --model_family llama --mode train

- Evaluate Yahoo topics (Qwen) with custom dataset:
  python run_eft.py --task yahoo --model_family qwen --mode eval \
    --script_args "--test_dataset ./yahoo_test_custom.json --sample_size 200"

The --script_args string is forwarded as-is to the underlying
script, so you can override its CLI options when needed.
"""

import argparse
import os
import shlex
import subprocess
import sys
from typing import Dict, Tuple


# Mapping: (task, model_family, mode) -> underlying script filename
_SCRIPT_MAP: Dict[Tuple[str, str, str], str] = {
    # Amazon sentiment
    ("amazon", "llama", "train"): "amazon_eft_training_llama.py",
    ("amazon", "llama", "eval"): "amazon_evaluate_llama.py",
    ("amazon", "qwen", "train"): "amazon_eft_training_qwen.py",
    ("amazon", "qwen", "eval"): "amazon_evaluate_qwen.py",

    # Yelp synthetic / sentiment
    ("yelp", "llama", "train"): "yelp_eft_training_llama.py",
    ("yelp", "llama", "eval"): "yelp_evaluate_llama.py",
    ("yelp", "qwen", "train"): "yelp_eft_training_qwen.py",
    ("yelp", "qwen", "eval"): "yelp_evaluate_qwen.py",

    # Yahoo topic classification
    ("yahoo", "llama", "train"): "yahoo_eft_training_llama.py",
    ("yahoo", "llama", "eval"): "yahoo_evaluate_llama.py",
    ("yahoo", "qwen", "train"): "yahoo_eft_training_qwen.py",
    ("yahoo", "qwen", "eval"): "yahoo_evaluate_qwen.py",

    # MNLI natural language inference
    ("mnli", "llama", "train"): "MNLI_eft_training_llama.py",
    ("mnli", "llama", "eval"): "MNLI_evaluate_llama.py",
    ("mnli", "qwen", "train"): "MNLI_eft_training_qwen.py",
    ("mnli", "qwen", "eval"): "MNLI_evaluate_qwen.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified entry for EFT (synthetic data) training and evaluation"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["amazon", "yelp", "yahoo", "mnli"],
        required=True,
        help="Task name: amazon, yelp, yahoo, or mnli.",
    )
    parser.add_argument(
        "--model_family",
        type=str,
        choices=["llama", "qwen"],
        required=True,
        help="Model family to use: llama or qwen.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval"],
        required=True,
        help="Whether to run training or evaluation.",
    )
    parser.add_argument(
        "--script_args",
        type=str,
        default="",
        help=(
            "Extra arguments to pass to the underlying script, "
            "e.g. --script_args \"--dataset_path ./data.json --num_train_epochs 3\""
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the resolved command without executing it.",
    )
    return parser.parse_args()


def resolve_script(task: str, model_family: str, mode: str) -> str:
    key = (task, model_family, mode)
    if key not in _SCRIPT_MAP:
        raise ValueError(f"No script mapping for task={task}, model_family={model_family}, mode={mode}")
    script_name = _SCRIPT_MAP[key]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(base_dir, "EFT", script_name)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Resolved script not found: {script_path}")
    return script_path


def main() -> None:
    args = parse_args()

    script_path = resolve_script(args.task, args.model_family, args.mode)

    extra_args = shlex.split(args.script_args) if args.script_args else []

    cmd = [sys.executable, script_path] + extra_args

    print("=== EFT Unified Entry ===")
    print(f"Task        : {args.task}")
    print(f"Model family: {args.model_family}")
    print(f"Mode        : {args.mode}")
    print(f"Script      : {script_path}")
    if extra_args:
        print(f"Script args : {' '.join(extra_args)}")

    if args.dry_run:
        print("[DRY RUN] Command not executed.")
        return

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
