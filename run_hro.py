#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified entry script for HRO (reward-enhanced GRPO) experiments.

This script dispatches to the task- and model-specific HRO training scripts
for Amazon, Yelp, Yahoo and MNLI, and for both LLaMA and Qwen.

The underlying scripts expect that base model paths (MERGED_MODEL_PATH or
--base-model-path) and data paths are already configured appropriately.

Example usages:

- Train Amazon HRO with LLaMA:
  python run_hro.py --task amazon --model_family llama \
    --script_args "--enable-dynamic-reward"

- Train Yelp HRO with Qwen, providing base model path:
  python run_hro.py --task yelp --model_family qwen \
    --script_args "--base-model-path /path/to/qwen_merged_model"

The --script_args string is forwarded as-is to the underlying script.
"""

import argparse
import os
import shlex
import subprocess
import sys
from typing import Dict, Tuple


# Mapping: (task, model_family) -> HRO script filename
_HRO_SCRIPT_MAP: Dict[Tuple[str, str], str] = {
    ("amazon", "llama"): "amazon_HRO_llama.py",
    ("amazon", "qwen"): "amazon_HRO_qwen.py",
    ("yelp", "llama"): "yelp_HRO_llama.py",
    ("yelp", "qwen"): "yelp_HRO_qwen.py",
    ("yahoo", "llama"): "yahoo_HRO_llama.py",
    ("yahoo", "qwen"): "yahoo_HRO_qwen.py",
    ("mnli", "llama"): "MNLI_HRO_llama.py",
    ("mnli", "qwen"): "MNLI_HRO_qwen.py.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified entry for HRO (reward-enhanced GRPO) training"
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
        help="Model family: llama or qwen.",
    )
    parser.add_argument(
        "--script_args",
        type=str,
        default="",
        help=(
            "Extra arguments for the underlying HRO script, e.g. "
            "--script_args \"--base-model-path /path/to/model --override-max-train-samples 1000\""
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the resolved command without executing it.",
    )
    return parser.parse_args()


def resolve_hro_script(task: str, model_family: str) -> str:
    key = (task, model_family)
    if key not in _HRO_SCRIPT_MAP:
        raise ValueError(f"No HRO script mapping for task={task}, model_family={model_family}")
    script_name = _HRO_SCRIPT_MAP[key]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(base_dir, "HRO", script_name)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Resolved HRO script not found: {script_path}")
    return script_path


def main() -> None:
    args = parse_args()

    script_path = resolve_hro_script(args.task, args.model_family)

    extra_args = shlex.split(args.script_args) if args.script_args else []

    cmd = [sys.executable, script_path] + extra_args

    print("=== HRO Unified Entry ===")
    print(f"Task        : {args.task}")
    print(f"Model family: {args.model_family}")
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
